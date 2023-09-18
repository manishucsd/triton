from library import *
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

class GemmProfiler():
  """class for triton matmul which sets up the problem, allocates torch tensors, and launches the kernel"""
  def __init__(self, matmul_description, problem_shape, tile_config, gemm_kernel):
    self.matmul_description = matmul_description
    self.problem_shape = problem_shape
    self.tile_config = tile_config
    self.gemm_kernel = gemm_kernel

    # Initialize the grid shape and parameters
    self.grid_shape = self.get_grid_shape()
    self.params = self.get_matmul_parameters() 

    # Verfication data
    self.verification = "Not verified"

    # Profiling data
    self.ms = None
    self.tflops = None

  def get_grid_shape(self) -> Any:
    """Compute the grid shape"""
    return (triton.cdiv(self.problem_shape.m, self.tile_config.kCtaTileM) * 
            triton.cdiv(self.problem_shape.n, self.tile_config.kCtaTileN),)
  
  def initialize_torch_tensor(self, num_rows, num_cols, dtype, layout) -> torch.Tensor:
    """Initialize a torch 2D tensor of rows x cols with integer random values centered around zero"""
    # Generate a random tensor with values in the range [-2, 2] with RowMajor layout in GPU memory
    matrix_shape = (num_rows, num_cols) if layout == MatrixLayout.RowMajor else (num_cols, num_rows)
    tensor_integer_valued_init = torch.randint(low=-2, high=3, size=matrix_shape, device="cuda")
    tensor = tensor_integer_valued_init.to(dtype=dtype)
    # Transpose the tensor if the layout is ColumnMajor
    return tensor if layout == MatrixLayout.RowMajor else tensor.T


  def get_flops(self) -> int:
    """Compute the number of flops"""
    return self.problem_shape.m * self.problem_shape.n * self.problem_shape.k * 2
  
  def get_matmul_parameters(self) -> Params:
    """Allocate/initialize torch tensors and returns in a `params` object"""
    torch.manual_seed(0)
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
    self.gemm_kernel[self.grid_shape](self.params.tensor_a, self.params.tensor_b, 
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
  
  def verify(self) -> bool:
    """Verify the result against torch.matmul"""

    # Run triton gemm kernel
    d_triton = self()

    # Find the wider of the two operands dtypes
    dot_operand_dtype = self.matmul_description.dtype_a \
      if self.params.tensor_a.element_size() > self.params.tensor_b.element_size() \
        else self.matmul_description.dtype_b

    # Operands for reference torch.matmul
    torch_tensor_a = self.params.tensor_a
    torch_tensor_b = self.params.tensor_b

    # Cast the operands to the wider dtype if dtype_a != dtype_b
    if self.matmul_description.dtype_a != dot_operand_dtype:
      torch_tensor_a = torch_tensor_a.to(dtype=dot_operand_dtype)
    if self.matmul_description.dtype_b != dot_operand_dtype:
      torch_tensor_b = torch_tensor_b.to(dtype=dot_operand_dtype)

    # Run torch.matmul
    d_torch_ref = torch.matmul(torch_tensor_a, torch_tensor_b)

    # Verify the result
    self.passed = torch.equal(d_triton, d_torch_ref)
    self.verification = "Passed" if self.passed else "Failed"

    # Print the result if verification failed
    if not self.passed:
      # print tensor_a shape and stride
      print("tensor_a: ", self.params.tensor_a)
      print("tensor_a.shape: ", self.params.tensor_a.shape)
      print("tensor_a.stride: ", self.params.tensor_a.stride())

      # print tensor_b shape and stride
      print("tensor_b: ", self.params.tensor_b)
      print("tensor_b.shape: ", self.params.tensor_b.shape)
      print("tensor_b.stride: ", self.params.tensor_b.stride())

      print("d_triton: ", d_triton)
      print("d_triton.shape: ", d_triton.shape)
      print("d_triton.stride: ", d_triton.stride())

      print("d_torch_ref: ", d_torch_ref)
      print("d_torch_ref.shape: ", d_torch_ref.shape)
      print("d_torch_ref.stride: ", d_torch_ref.stride())

    return self.passed

  def profile(self):
    """Profile the kernel"""
    self.ms = round(triton.testing.do_bench(lambda: self(), warmup=10, rep=100), 4)
    self.tflops = int(self.get_flops() / (self.ms * 1e9)) if self.ms else None

  def performance_report(self):
    """Print the Triton kernel performance report"""
    print("=================== Performance Report ===================")
    self.problem_shape.print()
    self.matmul_description.print()
    self.tile_config.print()
    print("Performance on NVIDIA A100-SXM4-40GB")
    print("  Verification: %s" % self.verification)
    print("  Runtime: %s ms" % self.ms)
    print("  TFLOPS: %s" % self.tflops)
    print("----------------------------------------------------------")