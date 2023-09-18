import torch


tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor_b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11 ,12]])
# Print tensor a details
print("tensor_a.shape: ", tensor_a.shape)
print("tensor_a.stride: ", tensor_a.stride())
print("tensor_a: ", tensor_a)
print("tensor_a.dtype: ", tensor_a.dtype)
print("tensor_a dtype size in number of bits: ", tensor_a.element_size() * 8)
print("tensor_a.layout: ", tensor_a.layout)

# Print tensor a details
print("tensor_a.T.shape: ", tensor_a.T.shape)
print("tensor_a.stride: ", tensor_a.T.stride())
print("tensor_a: ", tensor_a.T)
print("tensor_a.dtype: ", tensor_a.T.dtype)
print("tensor_a dtype size in number of bits: ", tensor_a.T.element_size() * 8)
print("tensor_a.layout: ", tensor_a.T.layout)


# Print tensor b details
print("tensor_b.shape: ", tensor_b.shape)
print("tensor_b.stride: ", tensor_b.stride())
print("tensor_b: ", tensor_b)
print("tensor_b.dtype: ", tensor_b.dtype)
print("tensor_b dtype size in number of bits: ", tensor_b.element_size() * 8)
print("tensor_b.layout: ", tensor_b.layout)

# matmul tensor_a and tensor_b
tensor_c = torch.matmul(tensor_a, tensor_b)
print("tensor_c.shape: ", tensor_c.shape)
print("tensor_c.stride: ", tensor_c.stride())
print("tensor_c: ", tensor_c)
