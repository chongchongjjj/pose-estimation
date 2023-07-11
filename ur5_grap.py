#数据集的制作
import torch
import torchvision.transforms
from torchvision.transforms import ToTensor
import time
import model
import sim
import numpy as np
import random
from dataset import get_point_cloud_from_z

clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
pi = 3.14152653589793
class ur5_grap():

    def __init__(self):
        #关闭之前的连接
        sim.simxFinish(-1)
        #获取用户ID
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if self.clientID != -1:
            print("Connected to remote API server!")
        else:
            print("Failed connecting to remote API server")
        #启动仿真
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

        #获取对象句柄
        _, self.visionSensorHandle = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor',
                                                                       sim.simx_opmode_oneshot_wait)
        _, self.redHandle = sim.simxGetObjectHandle(self.clientID, 'red', sim.simx_opmode_oneshot_wait)
        _, self.testHandle = sim.simxGetObjectHandle(self.clientID, 'test', sim.simx_opmode_oneshot_wait)

        self.device=torch.device("cuda")

        self.posenet = torch.load("model_exp/epoch_{}.pth".format(38))
        self.cropnet = model.cropnet().to(self.device)
        self.refinenet = torch.load("model_exp/epoch_refine_{}.pth".format(22))

    #随机移动目标体
    def randomchange(self):
        z = random.uniform(0.4, 0.6)
        x = random.uniform(-0.70 * z / 1.25, +0.70 * z / 1.25)
        y = random.uniform(-0.70 * z / 1.25, +0.70 * z / 1.25)
        a = random.uniform(0, pi / 3)
        b = 0
        g = random.uniform(0, pi / 3)
        sim.simxSetObjectPosition(self.clientID, self.redHandle, self.visionSensorHandle, (x, y, z), sim.simx_opmode_oneshot_wait)
        print(a,b,g)
        sim.simxSetObjectOrientation(self.clientID, self.redHandle, self.visionSensorHandle, (a, b, g), sim.simx_opmode_oneshot_wait)
        _, position = sim.simxGetObjectPosition(self.clientID, self.redHandle, self.visionSensorHandle, sim.simx_opmode_oneshot_wait)
        _, orientation = sim.simxGetObjectOrientation(self.clientID, self.redHandle, self.visionSensorHandle,
                                                      sim.simx_opmode_oneshot_wait)
        print(a,b,g)
        return position,orientation
    #预测相机视角中的目标物体
    def prevpose(self):
        sim.simxSetObjectPosition(self.clientID, self.testHandle, self.visionSensorHandle,
                                  (0,0,-10),
                                  sim.simx_opmode_oneshot_wait)
        _, resolution, color_buffer = sim.simxGetVisionSensorImage(self.clientID, self.visionSensorHandle, 0,
                                                                   sim.simx_opmode_oneshot_wait)
        _, _, depth_buffer = sim.simxGetVisionSensorDepthBuffer(self.clientID, self.visionSensorHandle,
                                                                sim.simx_opmode_oneshot_wait)

        color_img = np.asarray(color_buffer)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float32)
        color_img[color_img < 0] += 255
        color_img = np.fliplr(color_img)
        color_img = color_img.astype(np.uint8)
        color_img = ToTensor()(color_img)
        torchvision.transforms.ToPILImage()(color_img).show()
        color_img = torch.reshape(color_img, (1, -1, 224, 224))
        depth_img = np.asarray(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = depth_img.astype(np.float32)
        depth_img = np.fliplr(depth_img)
        zNear = 0.01
        zFar = 1.5
        # 1.5
        # 深度比例转成实际距离
        depth_img = depth_img * (zFar - zNear) + zNear
        depth_img = get_point_cloud_from_z(depth_img)
        depth_img = ToTensor()(depth_img)
        depth_img = torch.reshape(depth_img, (1, -1, 224, 224))

        color_img, depth_img = color_img.to(self.device), depth_img.to(self.device)
        prev_position = self.posenet(color_img.float(), depth_img.float())
        color_crop, depth_crop, c = self.cropnet(prev_position, color_img, depth_img)
        torchvision.transforms.ToPILImage()(color_crop[0]).show()
        color_crop, depth_crop = color_crop.to(self.device), depth_crop.to(self.device)
        orient = self.refinenet(color_crop, depth_crop)
        prev_position=prev_position[0]
        orient=orient[0]
        sim.simxSetObjectPosition(self.clientID, self.testHandle, self.visionSensorHandle,(prev_position[0].item(),prev_position[1].item(),prev_position[2].item()),
                                  sim.simx_opmode_oneshot_wait)
        sim.simxSetObjectOrientation(self.clientID, self.testHandle,self.visionSensorHandle,(orient[0],orient[1],orient[2]),
                                     sim.simx_opmode_oneshot_wait)
        return prev_position

if __name__ == '__main__':
    while(True):
        ur5=ur5_grap()
        ur5.randomchange()
        ur5.prevpose()
        time.sleep(5)
sim.simxFinish(clientID)


