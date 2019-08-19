import warnings
class DefaultConfig(object):

    #可视化工具参数
    visname = 'Federated_MNIST'


    #模型参数
    model = 'MLP'
    model_setting = {
        'vgg16':{
            'drop_rate':0.5,
            'num_classes':2
        },
        'MLP':{
            'layer_config':(784,512,256,10)
        }
    }

    #路径
    train_data_root = r'.\datasets\train'     #训练集地址
    test_data_root = r'.\datasets\test1'      #测试集地址
    load_model_path = None  #参数存储位置，在checkpoints/目录下'
    debug_file = 'None'   #debug文件夹地址
    result_file = r'.\results'
    
    #数据集参数
    batch_size = 256	#批梯度下降的单批次图片数量
    use_gpu = False		#是否使用gpu
    num_workers = 6		#多进程读取数据的进程数
    num_clients = 2    #参与联邦学习的客户端数量

    #训练参数
    random_seed = 1
    Devices_ID = '0'    #选择使用GPU的编号
    max_epoch = 10	#一次训练的迭代次数
    lr = 1e-3		    #学习率
    weight_decay = 1e-4  #正则化参数
    vis_freq = 2    #进行指标绘制的频率
    num_classes = 10
    def parse(self,kwargs):
        '''通过字典对defaultconfig类中的属性更新
        
        Args:
            kwargs: dict包含需要修改的参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):	
                warnings.warn("Warning: opt has not attribute %s"%k)	#若传入字典中出现无效key，警告
            else:
                setattr(self,k,v)
        #打印修改后的参数配置供检查
        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'parse':
                print(k,getattr(self,k))