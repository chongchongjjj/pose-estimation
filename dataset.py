import os

import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,Dataset
import cv2
import numpy as np
from argparse import Namespace

#点云转换
def get_point_cloud_from_z(Y,scale=1):
    camera_matrix = {'xc': 111.5, 'zc': 111.5, 'f': 193.9896}
    camera_matrix = Namespace(**camera_matrix)
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for i in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * Y[::scale, ::scale] / camera_matrix.f
    #

    XYZ = np.concatenate((X[..., np.newaxis], Y[::scale, ::scale][..., np.newaxis]
                          ,Z[..., np.newaxis]), axis=X.ndim)
    return XYZ

class dataset(Dataset):
    def __init__(self,dataset_path):
        #与vrep通讯连接 获取通讯ID
        self.dataset_path=dataset_path
        img_path=os.listdir(dataset_path)
        self.label_list=[]
        self.img_path=[]
        for img_name in img_path:
            if(img_name[-1]=="g"):
                label=img_name.split(".p")
                label=label[0].split("_")
                label=list(map(float,label))
                self.label_list.append(label)
                self.img_path.append(os.path.join(dataset_path,img_name))
    def __getitem__(self, idex):
        img=cv2.imread(self.img_path[idex],cv2.IMREAD_COLOR)
        img=ToTensor()(img)
        img_path=self.img_path
        img_path=img_path[idex][0:-3]+"npy"
        depth_img=np.load(img_path)
        cloud_point = get_point_cloud_from_z(depth_img, scale=1)
        cloud=ToTensor()(cloud_point)
        label=self.label_list[idex]
        return img,cloud,torch.Tensor(label)

    def __len__(self):
        return len(self.label_list)


if __name__ == '__main__':
    root="dataset/train"
    dataset_test=dataset(root)
    img,depth_img,label=dataset_test[0]
    print(img.shape)
    print(depth_img.shape)
    print(label.shape)
