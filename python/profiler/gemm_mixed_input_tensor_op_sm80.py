from library import *
from gemm_profiler import *
from gemm_kernels import gemm_mixed_input_tensor_op_multistage_kernel
import torch



# Matmul problem shape to fill 108 SMs on A100 and test instruction mix and scheduling 
# (decoupling auto-tuning)
problem_shape = ProblemShape(3456, 4096, 8192)
#problem_shape = ProblemShape(128, 256, 512)


# Create a list of tile configurations 
tile_configurations = []
tile_configurations.append(TileConfiguration(kCtaTileM=128, kCtaTileN=128, kCtaTileK=64, kNumStages=3))

# Create a list of matmul descriptions for F16 Tensor Cores
matmul_description = []
##################################################################################################
###                          F32 <= F16 * I8 + F32 
##################################################################################################
# RowMajor * ColumnMajor / TN GEMM with F16 operand A and I8 operand B
matmul_description.append(MatmulDescription(dtype_a=torch.float16, layout_a=MatrixLayout.RowMajor,
                                            dtype_b=torch.int8, layout_b=MatrixLayout.ColumnMajor,
                                            dtype_c=torch.float16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.float16, 
                                            dtype_accumulator=torch.float32))


##################################################################################################
###                          F32 <= BF16 * I8 + F32 
##################################################################################################
# RowMajor * ColumnMajor / TN GEMM with BF16 operand A and I8 operand B
matmul_description.append(MatmulDescription(dtype_a=torch.bfloat16, layout_a=MatrixLayout.RowMajor,
                                            dtype_b=torch.int8, layout_b=MatrixLayout.ColumnMajor,
                                            dtype_c=torch.bfloat16, layout_c=MatrixLayout.RowMajor,
                                            dtype_d=torch.bfloat16, 
                                            dtype_accumulator=torch.float32))

for tile_config in tile_configurations:
  for description in matmul_description:
    triton_gemm = GemmProfiler(description, problem_shape, tile_config, gemm_mixed_input_tensor_op_multistage_kernel)
    triton_gemm.verify()
    triton_gemm.profile()
    triton_gemm.performance_report()