#数据集的制作
import os.path
import sim
import numpy as np
import cv2
import random

#注：限制数据集每张图像的采集尺寸为224*224，如果需要改变，则需更改网络结构
#制作数据集的个数
num_train=3000
num_test=1000
num_vaild=1000
#定义vrep空间中的相机对象的名称和待检测物体对象的名称
camera_name='Vision_sensor'
object_name='red'
#定义相机的最远拍摄距离和最近拍摄距离 用于生成深度图像
zNear = 0.01
zFar = 1.5

#导入vrep
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
if clientID != -1:
    print("Connected to remote API server!")
else:
    print("Failed connecting to remote API server")
sim.simxGetPingTime(clientID)

#导入VisionHandle
errorCode_vision, visionSensorHandle = sim.simxGetObjectHandle(clientID, camera_name, sim.simx_opmode_oneshot_wait)

#导入Handle
errorCode_red, objectHandle = sim.simxGetObjectHandle(clientID, object_name, sim.simx_opmode_oneshot_wait)
pi= 3.14152653589793

#创建数据集文件夹
if not (os.path.exists("dataset/train")):
    os.mkdir("dataset/train")
if not (os.path.exists("dataset/test")):
    os.mkdir("dataset/test")
if not (os.path.exists("dataset/vaild")):
    os.mkdir("dataset/vaild")

#制作大小为2000的数据集
for i in range(num_test+num_train+num_vaild):
    #定义对象平移的空间，若相机张角为60°，此处已经给出x、y与z的投射关系，只需要改变z即可
    #此处决定了数据集的标签 若希望标签用其他旋转表示方式，则可以将其他坐标转换为a b g的形式 但在代码的最后区域改变标签的返还
    z=random.uniform(0.4,0.6)
    x=random.uniform(-0.70*z/1.25,+0.70*z/1.25)
    y=random.uniform(-0.70*z/1.25,+0.70*z/1.25)
    #欧拉角变换范围的定义
    a=random.uniform(0,pi/3)
    b=0
    g=random.uniform(0,pi/3)

    sim.simxSetObjectPosition(clientID, objectHandle, visionSensorHandle, (x, y, z), sim.simx_opmode_oneshot_wait)
    sim.simxSetObjectOrientation(clientID, objectHandle, visionSensorHandle, (a, b, g), sim.simx_opmode_oneshot_wait)

    _, resolution, color_buffer = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, sim.simx_opmode_oneshot_wait)
    _, _, depth_buffer = sim.simxGetVisionSensorDepthBuffer(clientID, visionSensorHandle, sim.simx_opmode_oneshot_wait)

    _,position= sim.simxGetObjectPosition(clientID, objectHandle, visionSensorHandle, sim.simx_opmode_oneshot_wait)
    _,orientation= sim.simxGetObjectOrientation(clientID, objectHandle, visionSensorHandle, sim.simx_opmode_oneshot_wait)

    color_img = np.asarray(color_buffer)
    color_img.shape = (resolution[1], resolution[0], 3)
    color_img = color_img.astype(np.float32)
    color_img[color_img < 0] += 255
    color_img = np.fliplr(color_img)
    color_img = color_img.astype(np.uint8)

    depth_img = np.asarray(depth_buffer)
    depth_img.shape = (resolution[1], resolution[0])
    depth_img = depth_img.astype(np.float32)
    depth_img = np.fliplr(depth_img)
    #深度比例转成实际距离
    depth_img = depth_img * (zFar - zNear) + zNear

    if(i<num_train):
        path="train"
    elif(i>=num_train and i<num_train+num_vaild):
        path="vaild"
    elif (i >= num_train+num_vaild and i < num_train+num_vaild+num_test):
        path = "test"
    cv2.imwrite("dataset/{}/{:.6f}_{:.6f}_{:.6f}_{:.6f}_{:.6f}_{:.6f}.png".format(path,position[0],position[1],position[2],orientation[0],orientation[1],orientation[2]),color_img)
    np.save("dataset/{}/{:.6f}_{:.6f}_{:.6f}_{:.6f}_{:.6f}_{:.6f}.npy".format(path,position[0], position[1], position[2],orientation[0],orientation[1],orientation[2]),depth_img)
sim.simxFinish(clientID)


