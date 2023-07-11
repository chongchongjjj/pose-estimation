import torch.nn
from dataset import dataset
from torch.utils.data import DataLoader
from model import posnet,cropnet,refine_net
from torch.utils.tensorboard import SummaryWriter
#----------------------------------------超参数设置------------------------------------------
learning_rate=0.0001
epoch=30
batchsize=32
p=0
w=0.0
start_epoch=0
load_pose_epoch=0
load_refine_epoch=0
lr_decay=0.3
refine_margin=0.00005
test_best_loss=10000
start_refine_flag=0
num_decay=0
root_train="dataset/train"
root_vaild="dataset/vaild"
#----------------------------------------设备与数据集加载---------------------------------------
if(torch.cuda.is_available()):
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
print("device:{}".format(device))

dataset_train=dataset(root_train)
dataloader_train=DataLoader(dataset_train,batch_size=batchsize,shuffle=True)
print("训练集大小为：{}".format(len(dataset_train)))

dataset_vaild=dataset(root_vaild)
dataloader_vaild=DataLoader(dataset_vaild,batch_size=1,shuffle=False)
print("验证集大小为：{}".format(len(dataset_vaild)))

writer=SummaryWriter("log_train")
#--------------------------------------模型加载---------------------------------------------
#加载或创建posenet
if(load_pose_epoch==0):
    posenet=posnet(p).to(device)
else:
    posenet = torch.load("model_exp/epoch_{}.pth".format(load_pose_epoch)).to(device)
    print("model_exp/epoch_{}.pth----已加载".format(load_pose_epoch))
#加载cropnet
cropnet=cropnet().to(device)
#加载refinenet
if(load_refine_epoch==0):
    refinenet=refine_net().to(device)
else:
    refinenet = torch.load("model_exp/epoch_refine_{}.pth".format(load_pose_epoch)).to(device)
    start_refine_flag=1
    print("model_exp/epoch_refine_{}.pth----已加载".format(load_pose_epoch))
#------------------------------------损失与优化器初始化--------------------------------------
loss_fn=torch.nn.SmoothL1Loss().to(device)
optimizer=torch.optim.Adam(posenet.parameters(),learning_rate,weight_decay=w)
#---------------------------------------模型训练-------------------------------------------
for i in range(epoch):
    train_step=0
    loss_avg=0
    #当损失下降至精化阈值时，冻结目标检测网络参数，启用精化网络的训练
    if(test_best_loss<refine_margin and start_refine_flag==0):
        start_refine_flag=1
        optimizer=torch.optim.Adam(refinenet.parameters(),learning_rate,weight_decay=w)
        #刷新学习率
        learning_rate=0.0001
        print("refine train")
    posenet.train()
    refinenet.train()
    for data in dataloader_train:
        #数据加载
        images,images_depth,labels=data
        images,images_depth,labels=images.to(device),images_depth.to(device),labels.to(device)
        #空间坐标估计
        position_prev,fusion_feature=posenet(images.float(),images_depth.float())
        #启用精化时进行精化结果预测
        if(start_refine_flag==1):
            color_crop,depth_crop,c=cropnet(position_prev,images,images_depth)
            color_crop, depth_crop,c=color_crop.to(device),depth_crop.to(device),c.to(device)
            outputs=refinenet(color_crop,depth_crop,fusion_feature)
            outputs=c*outputs
            loss=loss_fn(c*outputs,c*labels)
        else:
            loss=loss_fn(position_prev,labels[:,0:3])
        #梯度下降 初次迭代不训练 观察当前网络的效果
        optimizer.zero_grad()
        loss.backward()
        if(i!=0):
            optimizer.step()

        train_step+=1
        loss_avg+=loss
        if(train_step%10==0):
            print("train step : {}  loss : {}".format(train_step,loss))
    posenet.eval()
    refinenet.eval()
    with torch.no_grad():
        posenet.eval()
        loss_avg_vaild=0
        for data in dataloader_vaild:
            images, images_depth, labels = data
            images, images_depth, labels = images.to(device), images_depth.to(device), labels.to(device)
            position_prev,fusion_feature = posenet(images.float(), images_depth.float())
            if (start_refine_flag == 1):
                color_crop, depth_crop, c = cropnet(position_prev, images, images_depth)
                color_crop, depth_crop, c = color_crop.to(device), depth_crop.to(device), c.to(device)
                outputs = refinenet(color_crop, depth_crop,fusion_feature)
                outputs = c * outputs
                loss = loss_fn(c * outputs, c * labels)
            else:
                loss = loss_fn(position_prev, labels[:, 0:3])
            loss_avg_vaild+=loss
    loss_avg=(loss_avg/len(dataloader_train)).item()
    loss_avg_vaild=(loss_avg_vaild/len(dataloader_vaild)).item()
    print("Epoch {} train loss: {}".format(start_epoch + i + 1, loss_avg))
    print("Epoch {} vaild loss: {}".format(start_epoch + i + 1, loss_avg_vaild))
    writer.add_scalar("train_loss", loss_avg, start_epoch + i + 1)
    writer.add_scalar("vaild_loss", loss_avg_vaild, start_epoch + i + 1)
    writer.close()

    if(loss_avg_vaild<test_best_loss):
        if(start_refine_flag==1):
            torch.save(refinenet,"model_exp/epoch_refine_{}.pth".format(start_epoch + i + 1))
            torch.save(refinenet, "model_exp/epoch_posenet_best.pth")
        else:
            torch.save(refinenet, "model_exp/epoch_{}.pth".format(start_epoch + i + 1))
            torch.save(refinenet, "model_exp/epoch_refine_best.pth")
        test_best_loss=loss_avg_vaild
        print("best_model_save")
        num_decay=0
    else:
        num_decay+=1
    if(num_decay==3):
        num_decay=0
        learning_rate=learning_rate*lr_decay
        print("learning rate decay------learning rate = {}".format(learning_rate))
        optimizer = torch.optim.Adam(posenet.parameters(), learning_rate, weight_decay=w)
