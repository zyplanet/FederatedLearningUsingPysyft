import torch
import torch.nn as nn
import time
import os 
import warnings
class BasicModule(nn.Module):
    """对nn.Module的简单封装
    
    Attributes:
    	model_name: str进行类定义时的默认名称
    """
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = self.__class__.__name__
    
    def load(self,path):
        """将保存的参数赋给当前模型
        
        Args:
        	path: str保存参数的绝对路径
        	
        """
        self.load_state_dict(torch.load(path))
    
    def save(self,name=None):
        """保存当前模型的参数
        
        保存名称默认为模型名+时间
        
        Args:
        	name: str自定义的保存名称
        """
        if name is None:
            prefix = 'checkpoints\\'+self.model_name+'_'
            name = time.strftime(prefix+'%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(),name)
        return name

    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):	
                warnings.warn("Warning: model has not attribute %s"%k)	#若传入字典中出现无效key，警告
            else:
                setattr(self,k,v)