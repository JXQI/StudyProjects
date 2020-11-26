import numpy as np
import torch

a=np.array([2,3,4])
b=a[:,None,None]
print(b)
c=np.array([[1,2,3],[2,3,4]])
mask=c==b #这块的输出是True和False
print(mask)
print(torch.tensor(mask,dtype=torch.uint8))