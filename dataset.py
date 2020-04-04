"""
这个项目的代码应该是对于利用Pytorch进行网络训练的，最基础的版本了。
由赵志明写于2020年1月
"""

from torch.utils.data  import Dataset,DataLoader

import os
import torch
import SimpleITK as sitk
import numpy as np




def get_files_list(mra_dir,seg_dir):
    imgs=[]
    masks=[]
    n=len(os.listdir(mra_dir))
    #or file in os.listdir(root)
    for i in range(n):
    #for mra_file in os.listdir(mra_dir):
        img=os.path.join(mra_dir,"data-%d.nii"%i)
        imgs.append(img)
    #for seg_file in os.listdir(seg_dir):
        mask=os.path.join(seg_dir,"seg-%d.nii"%i)
        #img = os.path.join(root, file)
        #mask = os.path.join(root, file)
        masks.append(mask)
    return imgs,masks

class NiiDataset(Dataset):
    def __init__(self,mra_dir,label_dir,mra_transforms=None,label_transforms=None):
        imgs,labels=get_files_list(mra_dir,label_dir)
        self.imgs=imgs
        self.labels=labels
        self.imgs_transforms=mra_transforms
        self.labels_transforms=label_transforms

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index],self.labels[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage(x_path, sitk.sitkInt16)).astype(np.float).transpose((1,2,0))
        img_y = sitk.GetArrayFromImage(sitk.ReadImage(y_path, sitk.sitkInt16)).astype(np.float).transpose((1,2,0))
        if self.imgs_transforms is not None:
            img_x=self.imgs_transforms(img_x)
            img_x=torch.unsqueeze(img_x,dim=0)
        if self.labels_transforms is not None:
            img_y=self.labels_transforms(img_y)

        return img_x,img_y
    def __len__(self):
        return len(self.imgs)