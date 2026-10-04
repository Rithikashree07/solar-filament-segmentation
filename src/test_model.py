from model import UNet
import torch

model = UNet()

x = torch.randn(1, 1, 512, 512)

y = model(x)

print("Input :", x.shape)
print("Output:", y.shape)