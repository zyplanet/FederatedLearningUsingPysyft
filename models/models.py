from .BasicModule import BasicModule
import torch.nn as nn

#网络结构中常用的模块要进行封装
class vgg_conv1(nn.Module):
    '''vgg模型中用的两卷积核模块
    
    Attributes:
    	inchannels: int输入图的通道数量
    	outchannels: int输出图的通道数量
    	f: 模块包含的计算结构
    '''
    def __init__(self,inchannels,outchannels):
        '''初始化
		
		Args:
			pass
        '''
        super(vgg_conv1,self).__init__()
        #nn.Conv2d是二维卷积，下面所填的参数分别为
        #输入图通道数，卷积变换后通道数，卷积所使用核大小
        #加边(即给图像周围加零以得到固定大小的输出）的长度，是否添加截距项
        #nn.ReLU()是ReLU非线性变换，inplace选择是否进行原地计算（即对原有数值进行覆盖，可以节约内存）
        #nn.BatchNorm2d是批归一化运算，用于对中间计算结果进行归一化，提高模型的稳定性
        self.f = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size=3, padding=1,bias=False),
			nn.ReLU(inplace = True),
			nn.BatchNorm2d(outchannels),
			nn.Conv2d(outchannels,outchannels,kernel_size=3,padding=1,bias=False),
			nn.ReLU(inplace = True),
            nn.BatchNorm2d(outchannels),
			nn.MaxPool2d(kernel_size=2)
            )
    def forward(self,x):
        '''用定义好的结构对输入进行计算
        
        Args:
        	x: tensor输入张量
        
        Returns:
        	out: tensor输出张量
        '''
        out = self.f(x)
        return out

#下面的模块类似，不进行注释
class vgg_conv2(nn.Module):
    def __init__(self,inchannels,outchannels):
        super(vgg_conv2,self).__init__()
        self.f = nn.Sequential(
        	nn.Conv2d(inchannels,outchannels,kernel_size = 3,padding=1,bias = False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(outchannels,outchannels,kernel_size = 3,padding=1,bias = False),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(outchannels),
            nn.Conv2d(outchannels,outchannels,kernel_size = 1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.MaxPool2d(kernel_size = 2)
        )
    def forward(self,x):
        out = self.f(x)
        return out
#定义vgg16
class vgg16(BasicModule):
    '''继承BasicModule，按论文的结构来写
    
    Attributes:
    	drop_rate: float Dropout运算中将张量分量置零的概率
    	num_classes: int最终模型需要分类的类型数量
    	layer1-layer5: Module类，定义了vgg16卷积层的结构
    	classifier: Sequential类，最终对卷积结果进行分类的神经网络
    '''
    def __init__(self,**kwargs):
        '''初始化
        
        Args:
        	pass.
        '''
        default_drop_rate=0.5
        default_num_classes=2
        self.drop_rate = default_drop_rate
        self.num_classes = default_num_classes
        if kwargs:
            self.parse(kwargs)
        super(vgg16,self).__init__()
        self.layer1 = vgg_conv1(3,64)
        self.layer2 = vgg_conv1(64,128)
        self.layer3 = vgg_conv2(128,256)
        self.layer4 = vgg_conv2(256,512)
        self.layer5 = vgg_conv2(512,512)
        self.classifier = nn.Sequential(
        	nn.Linear(25088,4096),
            nn.Dropout(self.drop_rate),
            nn.Linear(4096,4096),
            nn.Dropout(self.drop_rate),
            nn.Linear(4096,self.num_classes)
        )
    def forward(self,x):
        '''用定义好的结构对输入进行计算
        
        Args:
        	pass
        Returns:
        	pass
        '''
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out.view(x.size(0),-1))
        
        return out

import collections
class MLP(BasicModule):
    '''继承BasicModule,一个灵活的多层感知机

    Attributes:
        layer_config: tuple，表示各层神经元个数
        f: Sequential类，由layer_config形成有序字典，用Sequential封装得到的网络主体结构
    '''
    def __init__(self,**kwargs):
        super(MLP,self).__init__()
        defualt_config = (784,512,256,10)
        self.layer_config = defualt_config
        
        if kwargs:
            self.parse(kwargs)
        layer_dict = collections.OrderedDict({})
        for ids in range(len(self.layer_config)-1):
            if ids+1 != len(self.layer_config)-1:
                layer_dict.update(
                    {
                        'layer_'+str(ids):nn.Sequential(
                             nn.Linear(self.layer_config[ids],self.layer_config[ids+1]),
                             nn.ReLU()
                    )
                    }
                )
            else:
                layer_dict.update(
                    {
                        'layer_'+str(ids):nn.Sequential(
                            nn.Linear(self.layer_config[ids],self.layer_config[ids+1]),
                            nn.Softmax(dim=1)
                        )
                    }
                )

        self.f = nn.Sequential(layer_dict)

    def forward(self,x):
        return self.f(x.view(-1,self.layer_config[0]))

