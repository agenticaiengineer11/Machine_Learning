import torch

# 0D Tensor
scalar = torch.tensor(5)

# 1D Tensor
vector = torch.tensor([1, 2, 3])

# 2D Tensor
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

# 3D Tensor
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:", matrix)
print("3D Tensor:", tensor_3d)

print("Scalar dimensions:", scalar.ndim)
print("Vector dimensions:", vector.ndim)
print("Matrix dimensions:", matrix.ndim)
print("3D dimensions:", tensor_3d.ndim)
