import torch

def one_hot_encoding(tensor,num_classes):
    """将N维类别向量进行one_hot编码

    Args:
        tensor: LongTensor, 需要进行onehot编码的N维向量
        num_classes: int, 总共的类别数
    Returns: 
        Longtensor, 形状为N*K的onehot张量

    Raises:
        pass
    """
    onehot = torch.zeros(tensor.size(0),num_classes)
    for rows,value in enumerate(tensor):
        onehot[rows,value.numpy()]=1
    return onehot

def cf_calc(confusion_matrix):
    """calculate index of a given confusion_matrix

    Args: 
        confusion_matrix: K*K matrix, rows corresponds to ground truth
    and columns corresponds to prediction.

    Return:
        avg_precision: float, mean average_precision of given confusion_matrix.
        min_precsion: float, the min precision of given confusion_matrix.
        min_class: float, class with min precision.
    
    Raises:
        pass
    """
    num_classes = confusion_matrix.shape[0]
    class_ac = []
    correct = 0
    for index in range(num_classes):
        correct += confusion_matrix[index][index]
        class_ac.append(100.*confusion_matrix[index][index]/(confusion_matrix[:,index].sum())if confusion_matrix[:,index].sum()!=0 else 0)
    ac = 100.*correct/confusion_matrix.sum()
    lowest_ac = min(class_ac)
    lowest_class = class_ac.index(lowest_ac)
    return round(ac,2),round(lowest_ac,2),lowest_class