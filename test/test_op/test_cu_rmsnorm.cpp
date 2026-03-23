#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <random>
#include "../source/op/kernels/cuda/rmsnorm_kernel.cuh"
#include "../source/op/kernels/kernels_interface.h"
TEST(test_rmsnorm_cu, rmsnorm_nostream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }
  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, nullptr);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
}

TEST(test_rmsnorm_cu, rmsnorm_stream) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

TEST(test_rmsnorm_cu, rmsnorm_stream2) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu, stream);
  out_cu.to_cpu();

  kernel::get_rmsnorm_kernel(base::DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu, nullptr);

  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

// Nsight Compute 测试入口
TEST(test_rmsnorm_cu, rmsnorm_profile_single_kernel) {
  auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

  // 固定参数，便于 Nsight Compute 对比两次报告。
  // hidden 必须是 4 的倍数，否则 dim kernel 的 float4 访问会因行首地址非 16B 对齐而崩溃。
  constexpr int batch = 32, seq_len = 151, hidden = 1600;
  constexpr int size = batch * seq_len * hidden;
  constexpr int warmup = 10;
  constexpr int iters = 10;

  tensor::Tensor in_cpu(base::DataType::kDataTypeFp32, std::vector{batch, seq_len, hidden}, true, alloc_cpu);
  tensor::Tensor wei_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);
  tensor::Tensor out_cpu(base::DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::mt19937 mt(42);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i) {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  tensor::Tensor in_cu = in_cpu.clone();
  tensor::Tensor wei_cu = wei_cpu.clone();
  tensor::Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 只保留一套代码：切换下面一行来生成两份 Nsight 报告。
  // kernel::RMSNormKernel kernel_fn = kernel::rmsnorm_kernel_cu_warp;
  // kernel::RMSNormKernel kernel_fn = kernel::rmsnorm_kernel_cu;
  // kernel::RMSNormKernel kernel_fn = kernel::rmsnorm_kernel_cu_block_float4;
  kernel::RMSNormKernel kernel_fn = kernel::rmsnorm_kernel_cu_dim;

  for (int i = 0; i < warmup; ++i) {
    kernel_fn(in_cu, wei_cu, out_cu, stream);
  }
  for (int i = 0; i < iters; ++i) {
    kernel_fn(in_cu, wei_cu, out_cu, stream);
  }
  cudaStreamSynchronize(stream);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  cudaStreamDestroy(stream);
}
