from network import Unet3d
from dataset import NiiDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from loss import SoftDiceLoss,dice_coeff


import torch
import os
import sys
import utility


def train(net,epochs,batch_size,lr,mra_transforms,label_transforms):
    dir_imgs="./data/after_slice/copy/data/"
    dir_labels = "./data/after_slice/copy/seg/"
    dir_model="./model"

    utility.sureDir(dir_model)

    #load data
    dataset=NiiDataset(mra_dir=dir_imgs,label_dir=dir_labels,
                       mra_transforms=mra_transforms,
                       label_transforms=label_transforms)

    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=4)

    #loss and optimizer
    criterion=SoftDiceLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=lr)

    #begin train
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print('-' * 10)

        net.train()
        dt_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0

        for img,label in dataloader:
            step+=1
            input = img.type(torch.FloatTensor).cuda() #因为前面已经为它们to tensor了
            label = label.type(torch.FloatTensor).cuda().squeeze()  # .long()

            # zero the parameter gradients
            optimizer.zero_grad()

            output=net(input)


            out = output[:,1, :, :, :].squeeze()#(75,64,64)
            print("dice: %0.3f " % dice_coeff(out, label))

            loss=criterion(out,label)
            loss.backward()
            optimizer.step()
            epoch_loss+=float(loss.item())
            print("%d/%d,train_loss:%0.3f" % (step, dt_size // dataloader.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))

        torch.save(net.state_dict(),dir_model)



if __name__ == '__main__':

    #parameters set
    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs=100
    batch_size=1
    lr=0.001
    x_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))])
    y_transforms = transforms.ToTensor()


    net = Unet3d(in_ch=1,out_ch=2)
    net.cuda()


    try:
        train(net=net,
              epochs=epochs,
              batch_size=batch_size,
              lr=lr,
              mra_transforms=x_transforms,
              label_transforms=y_transforms)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
