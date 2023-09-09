from typing import Any

import enum, re
from enum import auto

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


torch_to_triton_type = {
  torch.float32: tl.float32,
  torch.float16: tl.float16,
  torch.bfloat16: tl.bfloat16,
  torch.int32: tl.int32,   
}

class ProblemShape():
  def __init__(self, m, n, k) -> None:
    self.m = m
    self.n = n
    self.k = k

  def print(self) -> None:
    """Print the problem shape"""
    print("Problem shape")
    print("  M = %s, N = %s, K = %s" % (self.m, self.n, self.k))

class Params():
  """Matmul tensor/matrix parameters"""
  def __init__(self, tensor_a, tensor_b, tensor_c, tensor_d) -> None:
    self.tensor_a = tensor_a
    self.tensor_b = tensor_b
    self.tensor_c = tensor_c
    self.tensor_d = tensor_d

class MatrixLayout(enum.Enum):
    """Matrix layout"""
    ColumnMajor = auto()  # cuBLAS `N` layout
    RowMajor = auto()     # cuBLAS `T` layout (Transposed)
    
class MatmulDescription():
  """Matmul description (Functional)"""
  def __init__(self, 
               dtype_a: tl.constexpr, layout_a, 
               dtype_b: tl.constexpr, layout_b, 
               dtype_c: tl.constexpr, layout_c, 
               dtype_d: tl.constexpr, 
               dtype_accumulator: tl.constexpr
              ) -> None:
    self.dtype_a = dtype_a
    self.layout_a = layout_a
    self.dtype_b = dtype_b
    self.layout_b = layout_b
    self.dtype_c = dtype_c
    self.layout_c = layout_c
    self.dtype_d = dtype_d
    self.dtype_accumulator = dtype_accumulator
    pass
  
  def print(self):
    """Print the matmul description"""
    print("GEMM description")
    print("  DataType : a_type = %s, b_type = %s,  accumulator_type = %s, c_type = %s" % (self.dtype_a, self.dtype_b, self.dtype_accumulator, self.dtype_c))
    print("  Layout   : a_layout = %s, b_layout = %s, c_layout = %s" % (self.layout_a, self.layout_b, self.layout_c))

class TileConfiguration():
  """Kernel tile configuration (Performance)"""
  def __init__(self,
               kCtaTileM : tl.constexpr, # Threadblock tile M in number of elements
               kCtaTileN : tl.constexpr, # Threadblock tile N in number of elements
               kCtaTileK : tl.constexpr, # Threadblock tile K in number of elements
               kNumStages : tl.constexpr # Number of stages in gemm_k_iteration
              ) -> None:
    self.kCtaTileM = kCtaTileM
    self.kCtaTileN = kCtaTileN
    self.kCtaTileK = kCtaTileK
    self.kNumStages = kNumStages

  def print(self) -> None:
    """Print the tile configuration"""
    print("Tile configuration")
    print("  TileM x TileN x TileK (%s x %s x %s), NumStages (%s)" \
          % (self.kCtaTileM, self.kCtaTileN, self.kCtaTileK, self.kNumStages))

#@triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=5, num_warps=4)
#    ],
#    key=['problem_shape_m', 'problem_shape_n', 'problem_shape_k'],
#)
@triton.jit
def matmul_kernel(tensor_a, tensor_b, tensor_c, tensor_d,
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
    b_tile_upcast = b_tile.to(tensor_a.dtype.element_ty)
    
    # Mma
    accumulator += tl.dot(a_tile, b_tile_upcast, out_dtype=dtype_accumulator) 

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


class TritonMatmul():
  """class for triton matmul which sets up the problem, allocates torch tensors, and launches the kernel"""
  def __init__(self, matmul_description, problem_shape, tile_config):
    self.matmul_description = matmul_description
    self.problem_shape = problem_shape
    self.tile_config = tile_config

    # Initialize the grid shape and parameters
    self.grid_shape = self.get_grid_shape()
    self.params = self.get_matmul_parameters() 

    # Profiling data
    self.ms = None
    self.tflops = None

  def get_grid_shape(self) -> Any:
    """Compute the grid shape"""
    return (triton.cdiv(self.problem_shape.m, self.tile_config.kCtaTileM) * 
            triton.cdiv(self.problem_shape.n, self.tile_config.kCtaTileN),)
  
  def initialize_torch_tensor(self, num_rows, num_cols, dtype, layout) -> torch.Tensor:
    """Initialize a torch tensor"""
    int_valued_tensor = torch.randint(-1, 2, (num_rows, num_cols), dtype=torch.int32, device="cuda")
    tensor = int_valued_tensor.to(dtype=dtype)
    """
    if layout == MatrixLayout.ColumnMajor:
      return tensor.t()
    else:
      return tensor
    """
    return tensor
  
  def get_flops(self) -> int:
    """Compute the number of flops"""
    return self.problem_shape.m * self.problem_shape.n * self.problem_shape.k * 2
  
  def get_matmul_parameters(self) -> Params:
    """Allocate/initialize torch tensors and returns in a `params` object"""
    tensor_a = self.initialize_torch_tensor(self.problem_shape.m, self.problem_shape.k, 
                                            self.matmul_description.dtype_a, 
                                            self.matmul_description.layout_a)
    tensor_b = self.initialize_torch_tensor(self.problem_shape.k, self.problem_shape.n, 
                                            self.matmul_description.dtype_b, 
                                            self.matmul_description.layout_b)
    tensor_c = self.initialize_torch_tensor(self.problem_shape.m, self.problem_shape.n, 
                                            self.matmul_description.dtype_c, 
                                            self.matmul_description.layout_c)
    tensor_d = self.initialize_torch_tensor(self.problem_shape.m, self.problem_shape.n, 
                                            self.matmul_description.dtype_d, 
                                            self.matmul_description.layout_c)
    return Params(tensor_a, tensor_b, tensor_c, tensor_d)


  def __call__(self) -> torch.Tensor:
    """Launch the kernel"""   
    matmul_kernel[self.grid_shape](self.params.tensor_a, self.params.tensor_b, 
                                   self.params.tensor_c, self.params.tensor_d,
                                   self.params.tensor_a.stride(0), self.params.tensor_a.stride(1),
                                   self.params.tensor_b.stride(0), self.params.tensor_b.stride(1),
                                   self.params.tensor_c.stride(0), self.params.tensor_c.stride(1),
                                   self.params.tensor_d.stride(0), self.params.tensor_d.stride(1),
                                   self.problem_shape.m, self.problem_shape.n, self.problem_shape.k,
                                   self.tile_config.kCtaTileM, self.tile_config.kCtaTileN, self.tile_config.kCtaTileK,
                                   torch_to_triton_type[self.matmul_description.dtype_accumulator],
                                   num_stages = self.tile_config.kNumStages)
    return self.params.tensor_d
  
  def verify(self):
    """Verify the result against torch.matmul"""
    d_triton = self()
    d_torch_ref = torch.matmul(self.params.tensor_a, self.params.tensor_b.to(self.matmul_description.dtype_a))
    passed = torch.equal(d_triton, d_torch_ref)
    if not passed:
      print("d_triton: ", d_triton)
      print("d_torch_ref: ", d_torch_ref)
    return passed
  

  def profile(self):
    """Profile the kernel"""
    self.ms = round(triton.testing.do_bench(lambda: self(), warmup=10, rep=100), 4)
    self.tflops = int(self.get_flops() / (self.ms * 1e9))
    self.print_profile()

  def print_profile(self):
    """Print the profile"""
    print("----------------------------------------")
    self.problem_shape.print()
    self.matmul_description.print()
    self.tile_config.print()
    print("Performance on NVIDIA A100-SXM4-40GB")
    print("  Runtime: %s ms" % self.ms)
    print("  TFLOPS: %s" % self.tflops)
  
# Create a matmul problem, description, and tile configuration
# problem_shape = ProblemShape(128, 256, 64)
problem_shape = ProblemShape(3456, 4096, 8192)

##################################################################################################
###                          F32 <= F16 * I8 + F32 (Mixed-input Tensor Cores)
##################################################################################################
matmul_description_mixed_input_f16_i8 = MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.RowMajor, 
                                                   dtype_b=torch.int8, layout_b=MatrixLayout.ColumnMajor,
                                                   dtype_c=torch.float32, layout_c=MatrixLayout.RowMajor,
                                                   dtype_d=torch.float32, 
                                                   dtype_accumulator=torch.float32)
tile_config_mixed_input = TileConfiguration(kCtaTileM=128, kCtaTileN=128, kCtaTileK=64, kNumStages=4)

# Create a triton matmul for mixed-precision Tensor Cores 
triton_matmul_mixed_input_f16_i8 = TritonMatmul(matmul_description_mixed_input_f16_i8, problem_shape, tile_config_mixed_input)

# Verify and profile triton matmul
if(triton_matmul_mixed_input_f16_i8.verify()):
  print("triton_matmul_mixed_input_f16_i8 matmul verified")
else:
  print("triton_matmul_mixed_input_f16_i8 matmul failed verification")

triton_matmul_mixed_input_f16_i8.profile()


##################################################################################################
###                          F32 <= BF16 * I8 + F32 (Mixed-input Tensor Cores)
##################################################################################################
matmul_description_mixed_input_bf16_i8 = MatmulDescription(dtype_a=torch.bfloat16, layout_a=MatrixLayout.RowMajor,
                                                   dtype_b=torch.int8, layout_b=MatrixLayout.ColumnMajor,
                                                   dtype_c=torch.float32, layout_c=MatrixLayout.RowMajor,
                                                   dtype_d=torch.float32, 
                                                   dtype_accumulator=torch.float32)

# Create a triton matmul for mixed-precision Tensor Cores 
triton_matmul_mixed_input_bf16_i8 = TritonMatmul(matmul_description_mixed_input_bf16_i8, problem_shape, tile_config_mixed_input)

triton_matmul_mixed_input_bf16_i8.profile()