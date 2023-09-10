from library import *
from gemm_profiler import *
from gemm_kernels import gemm_tensor_op_multistage_kernel
import torch

# Matmul problem shape to fill 108 SMs on A100 and test instruction mix and scheduling 
# (decoupling auto-tuning)
problem_shape = ProblemShape(3456, 4096, 8192)

# Create a list of tile configurations 
tile_configurations = []
tile_configurations.append(TileConfiguration(kCtaTileM=128, kCtaTileN=128, kCtaTileK=64, kNumStages=4))

# Create a list of matmul descriptions for F16 Tensor Cores
matmul_description = []
##################################################################################################
###                          F16 <= F16 * F16 + F16 (F16 Tensor Cores)
##################################################################################################
matmul_description.append(MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.RowMajor,
                                            dtype_b=torch.float16, layout_b=MatrixLayout.RowMajor,
                                            dtype_c=torch.float16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.float16, 
                                            dtype_accumulator=torch.float16))

matmul_description.append(MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.RowMajor,
                                            dtype_b=torch.float16, layout_b=MatrixLayout.ColumnMajor,
                                            dtype_c=torch.float16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.float16, 
                                            dtype_accumulator=torch.float16))

matmul_description.append(MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.ColumnMajor,
                                            dtype_b=torch.float16, layout_b=MatrixLayout.RowMajor,
                                            dtype_c=torch.float16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.float16, 
                                            dtype_accumulator=torch.float16))

matmul_description.append(MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.ColumnMajor,
                                            dtype_b=torch.float16, layout_b=MatrixLayout.ColumnMajor,
                                            dtype_c=torch.float16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.float16, 
                                            dtype_accumulator=torch.float16))

for tile_config in tile_configurations:
  for description in matmul_description:
    triton_gemm = GemmProfiler(description, problem_shape, tile_config, gemm_tensor_op_multistage_kernel)
    triton_gemm.verify()
    triton_gemm.profile()
    triton_gemm.performance_report()

