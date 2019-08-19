import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T		#pytorch自带的数据处理库

class DogCat(Dataset):
    """用Dataset对dog vs. cat进行封装
    
    Attributes:
    	root: str数据绝对路径
    	transform: 供定义数据变换方式
    	train: bool指定当前是否为训练集
    	test: bool指定当前是否为测试集
    	img: list图片的绝对地址组成的列表
    """
    def __init__(self,root,transform=None,train=True,test=False):
        """初始化
        
        根据train,test的真假确定是训练集，验证集还是测试集
        并以此调整self.img和self.transform
        
        """
        self.test = test
        imgs = [os.path.join(root,img)for img in os.listdir(root)]
        imgs_num = len(imgs)	#统计数据总量
        #对kaggle提供的训练数据进行划分，得到训练集和验证集，八二开
        #测试集则保持不变
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.8*imgs_num)]
        else:
            self.imgs = imgs[int(0.8*imgs_num):]
            
        if transform is None:
            normalize = T.Normalize(mean = [0.485,0.456,0.406],
                                   std = [0.229,0.224,0.225])	#归一化
            
            # 若为测试集、验证集，不需要增加数据噪声
            if self.test or not train:
                #compose将一系列数变换复合起来
                self.transform = T.Compose([
                    T.Resize((224,224)),    #统一尺寸
                    T.ToTensor(),	#转换为张量
                    normalize
                ])
            # 为训练集添加噪声
            else:
                
                self.transform = T.Compose([
                    T.Resize((224,224)),    #统一尺寸
                    T.RandomResizedCrop(224),	#随机裁剪并resize到固定大小
                    T.RandomHorizontalFlip(),	#随机水平翻转
                    T.ToTensor(),
                    normalize
                ])
    def __getitem__(self,index):
        """根据序号读取一张图片
        
        返回图片张量，训练集与验证集有图片类别用于计算损失
        测试集没有类别
        调用Dataset对数据集封装时，必须有__getitem__方法
        
        Args:
        	index: int图片编号
        
        Returns:
        	若为self.test为True
        		tensor图片，longtensor图片id
        	若self.test为False
        		tensor图片，longtensor类别
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(img_path.split('\\')[-1])
        else:
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
        data = Image.open(img_path)		#调用PIL的api
        data = self.transform(data)	#使用定义好的transforms对数据进行变换
        
        return data,label
    
    def __len__(self):
        """显示所包含样本数量
        
        使用Dataset对数据封装时，必须有__len__方法
        """
        return len(self.imgs)

