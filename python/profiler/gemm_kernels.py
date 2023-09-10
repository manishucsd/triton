# Contains collection of GEMM kernels

import torch
from torch.testing import assert_close

import triton
import triton.language as tl

"""
Triton GEMM kernel for Tensor Core operations using Multistaged-pipeline for 
A100 Tensor Cores. This kernel should handle the following data types:
1. F16 <= F16 * F16 + F16 (F16 inputs, F16 accumulation, F16 output)      | Half-precision Tensor Cores
2. F32 <= F32 * F32 + F32 (F32 inputs, F32 accumulation, F32 output)      | Single-precision Tensor Cores
3. F16 <= F16 * F16 + F32 (F16 inputs, F32 accumulation, F16 output)      | Mixed-precision Tensor Cores
4. F32 <= BF16 * BF16 + F32 (BF16 inputs, F32 accumulation, F32 output)   | Mixed-precision Tensor Cores
5. BF16 <= BF16 * BF16 + F32 (BF16 inputs, F32 accumulation, F16 output)  | Mixed-precision Tensor Cores
"""
@triton.jit
def gemm_tensor_op_multistage_kernel(tensor_a, tensor_b, tensor_c, tensor_d,
                  stride_am, stride_ak, 
                  stride_bk, stride_bn, 
                  stride_cm, stride_cn,
                  stride_dm, stride_dn,
                  problem_shape_m, problem_shape_n, problem_shape_k,
                  kCtaTileM : tl.constexpr, kCtaTileN : tl.constexpr, kCtaTileK : tl.constexpr, 
                  dtype_accumulator : tl.constexpr):
  pid = tl.program_id(0)
  grid_shape_m = tl.cdiv(problem_shape_m, kCtaTileM)
  grid_shape_n = tl.cdiv(problem_shape_n, kCtaTileN)
  # Threadblock swizzle (re-order threadblocks for better L2 re-use)
  group_size_m = 8
  width = group_size_m * grid_shape_n
  group_id = pid // width
  group_size = min(grid_shape_m - group_id * group_size_m, group_size_m)
  cta_offset_m = group_id * group_size_m + (pid % group_size)
  cta_offset_n = (pid % width) // (group_size)

  # Element offsets a threadblock tile of kkNumStagesM x kCtaTileN x kCtaTileK
  element_offsets_m = ((cta_offset_m * kCtaTileM) + 
                        tl.arange(0, kCtaTileM)) % problem_shape_m
  element_offsets_n = ((cta_offset_n * kCtaTileN) +
                        tl.arange(0, kCtaTileN)) % problem_shape_n
  element_offsets_k = tl.arange(0, kCtaTileK)

  # 2D block of pointers to tile kCtaTileM x kCtaTileK of
  a_block_ptrs = tensor_a + (element_offsets_m[:,None] * stride_am +
                            element_offsets_k[None,:] * stride_ak)
  # 2D block of pointers to tile kCtaTileK x kCtaTileN of
  b_block_ptrs = tensor_b + (element_offsets_k[:,None] * stride_bk +
                            element_offsets_n[None,:] * stride_bn)
  
  # Mainloop
  accumulator = tl.zeros((kCtaTileM, kCtaTileN), dtype=dtype_accumulator)
    
  for gemm_k_iteration in range(0, tl.cdiv(problem_shape_k, kCtaTileK)):
    gemm_k = gemm_k_iteration * kCtaTileK
    # Load a tile of A and B into shared memory
    a_tile = tl.load(a_block_ptrs,
                     mask=element_offsets_k[None, :] < problem_shape_k - gemm_k, 
                     other=0.0)
    b_tile = tl.load(b_block_ptrs, 
                     mask=element_offsets_k[:, None] < problem_shape_k - gemm_k, 
                     other=0.0)
    
    # Mma
    accumulator += tl.dot(a_tile, b_tile, out_dtype=dtype_accumulator) 

    # Advance A and B pointers
    a_block_ptrs += kCtaTileK * stride_ak
    b_block_ptrs += kCtaTileK * stride_bk

  # Cast the accumulator to the destination type and store in global memory
  accumulator = accumulator.to(tensor_d.dtype.element_ty)
  element_offsets_m = cta_offset_m * kCtaTileM + tl.arange(0, kCtaTileM)
  element_offsets_n = cta_offset_n * kCtaTileN + tl.arange(0, kCtaTileN)
  d_block_ptrs = tensor_d + (element_offsets_m[:, None] * stride_dm + 
                             element_offsets_n[None, :] * stride_dn)
  tl.store(d_block_ptrs, accumulator, 
           mask=((element_offsets_m[:, None] < problem_shape_m) & 
                 (element_offsets_n[None, :] < problem_shape_n)))



"""
Triton GEMM kernel for Tensor Core operations using Multistaged-pipeline for 
A100 Tensor Cores. This kernel should handle the following data types:
1. F32 <= F16 * I8 + F32 (F16 input A, I8, input B, F16 accumulation, F16 output C/D)   | Mixed-input Tensor Cores
2. F32 <= BF16 * I8 + F32 (BF16 input A, I8, input B, F16 accumulation, F16 output C/D) | Mixed-input Tensor Cores
"""
@triton.jit
def gemm_mixed_input_tensor_op_multistage_kernel(tensor_a, tensor_b, tensor_c, tensor_d,
                  stride_am, stride_ak, 
                  stride_bk, stride_bn, 
                  stride_cm, stride_cn,
                  stride_dm, stride_dn,
                  problem_shape_m, problem_shape_n, problem_shape_k,
                  kCtaTileM : tl.constexpr, kCtaTileN : tl.constexpr, kCtaTileK : tl.constexpr, 
                  dtype_accumulator : tl.constexpr):
  pid = tl.program_id(0)
  grid_shape_m = tl.cdiv(problem_shape_m, kCtaTileM)
  grid_shape_n = tl.cdiv(problem_shape_n, kCtaTileN)
  # Threadblock swizzle (re-order threadblocks for better L2 re-use)
  group_size_m = 8
  width = group_size_m * grid_shape_n
  group_id = pid // width
  group_size = min(grid_shape_m - group_id * group_size_m, group_size_m)
  cta_offset_m = group_id * group_size_m + (pid % group_size)
  cta_offset_n = (pid % width) // (group_size)

  # Element offsets a threadblock tile of kkNumStagesM x kCtaTileN x kCtaTileK
  element_offsets_m = ((cta_offset_m * kCtaTileM) + 
                        tl.arange(0, kCtaTileM)) % problem_shape_m
  element_offsets_n = ((cta_offset_n * kCtaTileN) +
                        tl.arange(0, kCtaTileN)) % problem_shape_n
  element_offsets_k = tl.arange(0, kCtaTileK)

  # 2D block of pointers to tile kCtaTileM x kCtaTileK of
  a_block_ptrs = tensor_a + (element_offsets_m[:,None] * stride_am +
                            element_offsets_k[None,:] * stride_ak)
  # 2D block of pointers to tile kCtaTileK x kCtaTileN of
  b_block_ptrs = tensor_b + (element_offsets_k[:,None] * stride_bk +
                            element_offsets_n[None,:] * stride_bn)
  
  # Mainloop
  accumulator = tl.zeros((kCtaTileM, kCtaTileN), dtype=dtype_accumulator)
    
  for gemm_k_iteration in range(0, tl.cdiv(problem_shape_k, kCtaTileK)):
    gemm_k = gemm_k_iteration * kCtaTileK
    # Load a tile of A and B into shared memory
    a_tile = tl.load(a_block_ptrs,
                     mask=element_offsets_k[None, :] < problem_shape_k - gemm_k, 
                     other=0.0)
    b_tile = tl.load(b_block_ptrs, 
                     mask=element_offsets_k[:, None] < problem_shape_k - gemm_k, 
                     other=0.0)
    b_tile_in_dot_operand_dtype = b_tile.to(tensor_a.dtype.element_ty)
    
    # Mma
    accumulator += tl.dot(a_tile, b_tile_in_dot_operand_dtype, out_dtype=dtype_accumulator) 

    # Advance A and B pointers
    a_block_ptrs += kCtaTileK * stride_ak
    b_block_ptrs += kCtaTileK * stride_bk

  # Cast the accumulator to the destination type and store in global memory
  accumulator = accumulator.to(tensor_d.dtype.element_ty)
  element_offsets_m = cta_offset_m * kCtaTileM + tl.arange(0, kCtaTileM)
  element_offsets_n = cta_offset_n * kCtaTileN + tl.arange(0, kCtaTileN)
  d_block_ptrs = tensor_d + (element_offsets_m[:, None] * stride_dm + 
                             element_offsets_n[None, :] * stride_dn)
  tl.store(d_block_ptrs, accumulator, 
           mask=((element_offsets_m[:, None] < problem_shape_m) & 
                 (element_offsets_n[None, :] < problem_shape_n)))