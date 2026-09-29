import torch

path = "./checkpoints/full/Oh_CNN/fold0_best_model.pth"
data = torch.load(path)

print(type(data))