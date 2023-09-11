from typing import Any
import torch
import enum, re
from enum import auto
import triton.language as tl

"""
Contains helper functions and classes for verifing and profiling Triton GEMMs.
"""
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
    print("  DataType : %s <= %s * %s + %s" % (self.dtype_c, self.dtype_a, self.dtype_b, self.dtype_accumulator))
    print("  Layout   : %s <= %s * %s" % (self.layout_c, self.layout_a, self.layout_b))

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