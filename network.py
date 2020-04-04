from torch import nn

import torch
import torch.nn.functional as F

# class MultiViewConv(nn.Module):
#     #用来实例化
#     def __init__(self,in_ch,out_ch):
#         super(MultiViewConv, self).__init__()
#         self.conv_multiview=nn.Conv2d(in_channels=in_ch,out_channels=1,kernel_size=(3,3),stride=1,padding=1)
#         # self.conv_hc=nn.Conv2d(in_channels=w,out_channels=w,kernel_size=(3,3),stride=1,padding=1)
#         # self.conv_wc=nn.Conv2d(in_channels=h,out_channels=h,kernel_size=(3,3),stride=1,padding=1)
#         self.pointwise=nn.Conv3d(in_channels=3,out_channels=out_ch,kernel_size=1,stride=1)
#     #用来执行动作
#     def forward(self,x):
#         x_hw=x
#         x_wd=x.permute([0,1,3,4,2]) #[p,c,w,d,h]
#         x_hd=x.permute([0,1,2,4,3]) #这个函数真让人纠结 [p,c,h,d,w]
#         out_hw=self.conv_multiview(x_hw)
#         out_wd=self.conv_multiview(x_wd)
#         out_hd=self.conv_multiview(x_hd)
#
#         out_hw=out_hw
#         out_wd=out_wd.permute([0,1,4,2,3])
#         out_hd=out_hd.permute([0,1,2,4,3])
#
#         out=torch.cat((out_hd,out_wd,out_hw),dim=3)
#         out=self.pointwise(out)
#         return out



class DoubleConv(nn.Module):
    #类的结构
    def __init__(self,in_ch,out_ch):
        super(DoubleConv, self).__init__()
        self.conv=nn.Sequential(nn.Conv3d(in_ch,out_ch,3,padding=1),
                                nn.BatchNorm3d(out_ch),
                                nn.ReLU(inplace=True),
                                nn.Conv3d(out_ch,out_ch,3,padding=1),
                                nn.BatchNorm3d(out_ch),
                                nn.ReLU(inplace=True))
    #类的动作
    def forward(self,x):
        return self.conv(x)

#赵改
# class DoubleConv(nn.Module):
#     #类的结构
#     def __init__(self,in_ch,out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv=nn.Sequential(MultiViewConv(in_ch,out_ch),
#                                 nn.BatchNorm3d(out_ch),
#                                 nn.ReLU(inplace=True),
#                                 MultiViewConv(in_ch,out_ch),
#                                 nn.BatchNorm3d(out_ch),
#                                 nn.ReLU(inplace=True))
#     #类的动作
#     def forward(self,x):
#         return self.conv(x)

class Unet3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet3d, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool3d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv3d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)#还需要一个softmax
        out = nn.ReLU()(c10)
        # out = nn.Softmax(dim=1)(out)
        # out= torch.max(F.softmax(out, dim=1), dim=1)[1]

        return out

if __name__ == '__main__':
    net=Unet3d(1,2)
    print(net)