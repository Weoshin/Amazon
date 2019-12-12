import numpy as np
import torch

labels = torch.ones(10, 10, dtype=float)
output = torch.zeros(10, 10, dtype=float)
print(torch.sum((labels - output) ** 2))
