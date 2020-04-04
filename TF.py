import torch

# 首先，新建一个三维的tensor
a = torch.tensor([[[3, 3, 3], [3, 3, 3],
                   [3, 3, 3], [3, 3, 3]]])
# 打印出它的形状,这里它的形状为（1，4，3）
print(a.shape)
# 对该tensor进行维度变换，根据我上面文字详细说的那样
b = a.permute(2, 0, 1)
# 打印出变换后a的形状，形状变为（3，1，4）
print(b.shape)

# a.permute(2,0,1)
# b=a.permute(2,0,1)
# #打印出变换后a的形状
# print(b.shape)
# print(a.permute(2,0,1).shape)
