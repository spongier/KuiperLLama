#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {

// ======================warp测试=========================
static __global__ void row_rmsnorm_f32_warp(const float* in, const float* wei, float* out,
                                            const int size, const float eps) {
  const int lane_id = threadIdx.x;
  float sum = 0.0f;
  for (int i = lane_id; i < size; i += 32) {
    sum += in[i] * in[i];
  }

  using WarpReduce = cub::WarpReduce<float, 32>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = WarpReduce(temp).Sum(sum);
  if (lane_id == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = lane_id; i < size; i += 32) {
    out[i] = scale * in[i] * wei[i];
  }
}

void rmsnorm_kernel_cu_warp(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.size());
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 32;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_warp<<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32_warp<<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}



//  =======================float4优化========================
template<int32_t BLOCK_THREADS>
static __global__ void row_rmsnorm_f32_block_float4(const float* in, const float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  auto in_pack = reinterpret_cast<const float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  auto wei_pack = reinterpret_cast<const float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cu_block_float4(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.size());
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_block_float4<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32_block_float4<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_block_float4<1024><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32_block_float4<1024><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}



// ======================多block测试========================
template<int32_t BLOCK_THREADS>
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size, int size, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (bid >= dim_size) {
    return;
  }

  float* block_in = in + bid * size;
  float* block_out = out + bid * size;
  constexpr int pack_size = 4;   // 16B 对齐，4*4刚好16B
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(block_out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);  // dims是一个数组，记录各个维度数值
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim<threads_num><<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim<threads_num><<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32_dim<threads_num><<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    } else {
      row_rmsnorm_f32_dim<threads_num><<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
  }
}


template<int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
                                       const int size, const float eps) {
  const int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
  for (int i = tid; i < size; i += blockDim.x) {
    out[i] = scale * in[i] * wei[i];
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-5f;
  const int32_t size = static_cast<int32_t>(input.size());
  const float* in_ptr = input.ptr<float>();
  const float* wei_ptr = weight.ptr<float>();
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  if (size < 1024) {
    constexpr int threads_num = 128;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<threads_num><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<threads_num><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  } else {
    constexpr int threads_num = 1024;
    if (stream) {
      cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
      row_rmsnorm_f32<threads_num><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    } else {
      row_rmsnorm_f32<threads_num><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
  }
}

}  // namespace kernel
