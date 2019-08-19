import torch
import models
import torch.nn as nn
import os
import syft
import torchvision

from utils import cf_calc
from torchvision import transforms as T
from torchnet import meter
from config import DefaultConfig
from torch.utils.data import DataLoader
from visdom import Visdom

opt = DefaultConfig()
vis = Visdom(env = opt.visname,log_to_filename='./logs/logging.txt')

hook = syft.TorchHook(torch)    #用pysyft的hook增加pytorch的方法库，使其支持联邦学习
os.environ['CUDA_VISIBLE_DEVICES']=opt.Devices_ID   #指定所使用GPU的编号
device = torch.device("cuda"if opt.use_gpu else "cpu")  #指定进行模型优化的设备
torch.manual_seed(opt.random_seed)
def train(**kwargs):
    #使用命令行参数更新默认参数配置
    opt.parse(kwargs)
    
    #定义模型
    #这里只给出单GPU训练的定义方式
    #多GPU并行训练的模型定义方式略有差异
    model = getattr(models,opt.model)(**opt.model_setting[opt.model])   #用字典将模型参数转入
    print(model)
    #若有存储点则导入模型
    if opt.load_model_path:		
        model.load(opt.load_model_path)
    #是否在GPU上进行训练
    #如果是那么需要调用cuda()方法将内部参数等张量传入gpu
    if opt.use_gpu: model.cuda()

    #使用pysyft定义指定数量客户端
    createVar = locals()
    for id in range(opt.num_clients):
        createVar['Number'+str(id)] = syft.VirtualWorker(hook,id='Number'+str(id))

    #定义数据
    
    #使用pytorch的dataloader将数据集封装进行多进程读取
    #训练集则使用pysyft的FederatedDataLoader封装
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,),(0.3081,))
    ])
    names = locals()
    #将训练数据分配至各个客户端
    train_data = torchvision.datasets.MNIST("./datasets",train=True,transform=transforms,download=True).federate([names.get('Number'+str(k))for k in range(opt.num_clients)])
    test_data = torchvision.datasets.MNIST("./datasets",train=False,transform=transforms,download=True)
    train_dataloader = syft.FederatedDataLoader(train_data,batch_size=opt.batch_size,shuffle=True)
    test_dataloader = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False)
    
    #定义损失函数和优化方法
    criterion = nn.CrossEntropyLoss()	#使用交叉熵损失函数
    lr = opt.lr		#学习率
    #训练方法为随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,weight_decay=opt.weight_decay)	
    
    #指标统计工具
    loss_meter = meter.AverageValueMeter()  #损失统计工具
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)   #统计分类过程的混淆矩阵
    vis_index = 0   #训练过程可视化序列编号

    #初始化最高测试准确率，便于保存
    max_ac = round(100./opt.num_classes,2)
    #迭代训练
    model.train()   #将模型置为训练模式
    for epoch in range(opt.max_epoch):
        #重置统计值
        loss_meter.reset()  
        confusion_matrix.reset()
        
        #开始单次迭代
        for ii,(data,label) in enumerate(train_dataloader):
            model.send(data.location)   #全局模型分配至参与迭代客户端
            data,label = data.to(device),label.to(device)
            optimizer.zero_grad()		#清空上轮训练积累的梯度
            score = model(data)		    #在训练数据上进行inference
            loss = criterion(score,label)		#计算推断结果与目标的损失
            loss.backward()			     #根据损失计算梯度
            optimizer.step()		    #根据梯度进行单次训练
            model.get()                 #从客户端获得更新
            #统计指标
            loss_meter.add(loss.get().detach())
            confusion_matrix.add(score.get().detach(),label.get().detach())
            #如果有需要，进入debug模式
            if os.path.exists(opt.debug_file):
                #根据需要来写
                import ipdb
                ipdb.set_trace
            
            #按照给定绘制频率进行可视化
            if ii%opt.vis_freq == opt.vis_freq-1:
                if vis_index == 0:
                    update = 'insert'
                else:
                    update = 'append'
                vis.line(X = torch.Tensor([vis_index]),Y=loss_meter.value()[0].unsqueeze(0),win = 'trian_loss',update=update,opts=dict(title='train_loss'))
                cf = confusion_matrix.value()
                mAP,min_AP,_ = cf_calc(cf)
                vis.line(X=torch.Tensor([vis_index]),Y=torch.Tensor([[mAP,min_AP]]),win='train_precision',update=update,opts=dict(title='train_precision',legend=['mAP','min_AP']))
                vis_index += 1
        test_mAP,test_min_AP,_ = val(model,test_dataloader)
        if epoch == 0:
            update = 'insert'
        else:
            update = 'append'
        vis.line(X=torch.Tensor([epoch]),Y=torch.Tensor([[test_mAP,test_min_AP]]),win='test_precision',update=update,opts=dict(title='test_precision',legend=['mAP','min_AP']))
        if test_mAP>=max_ac:
            max_ac = test_mAP
            model.save()

        #以文本形式显示训练过程的参数等信息，根据需要填写
        # vis.text(txt')
        
        #在训练时会有一些调整学习率的策略
        # if condition:		#满足某个条件则进行调整
        #     lr = new_lr
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
    vis.close()
def val(model,dataloader):
    '''
    计算模型在验证集上的指标
    '''
    model.eval()	#将模型置为验证模式
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    with torch.no_grad():   #验证/测试时不需要计算梯度，使用torch.no_grad可以减少内存使用
        for data,label in dataloader:
                data,label = data.to(device),label.to(device)
                score = model(data)
                confusion_matrix.add(score,label)

    #统计指标
    return cf_calc(confusion_matrix.value())

def test(**kwargs):
    '''
    测试并保存结果
    '''
    pass

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
        python {0} train --env='env0701' --lr=0.01
        python {0} test --dataset='path/to/dataset/root/'
        python {0} help
    avaiable args:'''.format(__file__))
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__ == "__main__":
    import fire
    fire.Fire()