import torch
from vision_mamba import *
x = torch.randn(1, 3, 15, 15)

model = Vim(
    dim=256,  # Dimension of the model
    heads=8,  # Number of attention heads
    dt_rank=32,  # Rank of the dynamic routing tensor
    dim_inner=256,  # Inner dimension of the model
    d_state=256,  # State dimension of the model
    num_classes=1000,  # Number of output classes
    image_size=15,  # Size of the input image
    patch_size=3,  # Size of the image patch
    channels=3,  # Number of input channels
    dropout=0.1,  # Dropout rate
    depth=12,  # Depth of the model
)


out = model(x)

# Print the shape and output of the forward pass
print(out.shape)
# print(out)