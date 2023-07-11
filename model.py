#用一个简单的网络实现位姿回归
import torchvision.models
from torch import nn
import torch
import numpy as np
from dataset import dataset
from torch.utils.data import DataLoader

def piex_exchange(p,z):
    return 1.5*224*p/(1.73*z)

class posnet(nn.Module):
    def __init__(self,p):
        super(posnet, self).__init__()
        self.res_color=torchvision.models.resnet18()
        self.res_depth=nn.Sequential(
            torchvision.models.resnet18()
        )
        self.conv=nn.Conv2d(6,3,kernel_size=3,padding=1)
        self.fusion=nn.Sequential(
            nn.Linear(2000,1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
        )
        self.class_position=nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1000,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32,3)
        )
        self.class_Orientation=nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1000,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(32,3)
        )
    def forward(self,img,depth_img):
        img=self.conv(torch.cat((img,depth_img),dim=1))
        fusion_feature=self.res_color(img)
        position=self.class_position(fusion_feature)
        return position,fusion_feature

class cropnet(nn.Module):
    def __init__(self):
        super(cropnet, self).__init__()
        self.padding=nn.ZeroPad2d(32)
        self.resize=torchvision.transforms.Resize([64,64])

    def forward(self,position,color_img,depth_img):
        x=position[:,0]
        y=position[:,1]
        z=position[:,2]
        x=torch.reshape(x,(-1,1))
        y = torch.reshape(y, (-1, 1))
        z = torch.reshape(z, (-1, 1))
        m=(piex_exchange(x,z)+0.5).int()+112
        n=(piex_exchange(y,z)+0.5).int()+112
        d_half=((piex_exchange(0.1,z))/2).int()+1
        color_pad=nn.ZeroPad2d(padding=(40,40,40,40))(color_img)
        depth_pad=nn.ZeroPad2d(padding=(40,40,40,40))(depth_img)
        m_s = len(position)
        c=torch.ones((len(position),1))
        color_crop=torch.zeros((m_s,3,64,64))
        depth_crop=torch.zeros((m_s,3,64,64))
        for i in range(m_s):
            x1=40+m[i]-d_half[i]
            x2=40+m[i]+d_half[i]
            y1=40+n[i]-d_half[i]
            y2=40+n[i]+d_half[i]
            color=color_pad[i,:,y1:y2,x1:x2]
            depth=depth_pad[i,:,y1:y2,x1:x2]
            if(color.shape[-1]!=0 and color.shape[-2]!=0):
                color_crop[i]=self.resize(color)
                depth_crop[i]=self.resize(depth)
                c[i,0]=1
        return color_crop,depth_crop,c

class refine_net(nn.Module):
    def __init__(self):
        super(refine_net, self).__init__()
        self.color_net=nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.depth_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fusionnet=nn.Sequential(
            nn.Conv2d(32+64+3+3,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classnet_orien = nn.Sequential(
            nn.Linear(512 * 4 * 4+1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
    def forward(self,color_crop,depth_crop,globe_fusion_feature):
        color_feature=torch.cat((self.color_net(color_crop),color_crop),dim=1)
        depth_feature=torch.cat((self.depth_net(depth_crop),depth_crop),dim=1)
        fusion_feature=torch.cat((color_feature,depth_feature),dim=1)
        fusion_feature=self.fusionnet(fusion_feature)
        feature=torch.reshape(fusion_feature,(-1,512*4*4))
        feature=torch.cat((feature,globe_fusion_feature),dim=1)
        orientation=self.classnet_orien(feature)
        return orientation

if __name__ == '__main__':
    posenet=torch.load("model_exp/epoch_38.pth")
    root_vaild = "dataset/vaild"
    dataset_vaild = dataset(root_vaild)
    dataloader_vaild = DataLoader(dataset_vaild, batch_size=1, shuffle=False)
    print("验证集大小为：{}".format(len(dataset_vaild)))
    images, images_depth, labels =dataset_vaild[7]
    print(labels)
    images=torch.reshape(images,(1,-1,224,224)).to(torch.device("cuda")).float()
    images_depth = torch.reshape(images_depth, (1, -1, 224, 224)).to(torch.device("cuda")).float()
    output1=posenet(images,images_depth)
    print(output1)
    crop_net=cropnet().to(torch.device("cuda"))
    color_crop,depth_crop,c=crop_net(output1, images, images_depth)
    a=torch.reshape(color_crop,(3,32,32))
    a=torchvision.transforms.ToPILImage()(a)
    a.show()
