import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_dataset_num(file):
    '''
    Get the numbers of dataset in file(group), only apply to
    one group in h5.
    得到h5文件中dataset的数量，只适用于文件中只有单个group, dataset为二维矩阵的情况
    
    Parameters
    ----------
    file : TYPE 'file' object of h5py module
        DESCRIPTION. 
        使用h5py打开的h5文件

    Returns
    -------
    count : TYPE int
        DESCRIPTION. numbers of datasets in file
        文件中dataset的数目
    max_axis1 : TYPE int
        DESCRIPTION. the max dim of axis 1 of datasets
        datasets矩阵中轴1的最大数目
    max_axis2 : TYPE int
        DESCRIPTION. the max dim of axis 2 of datasets
        datasets矩阵中轴1的最大数目
    '''
    count = 0
    max_axis1 = 0
    max_axis2 = 0
    for key in file.keys():
        count += 1
        max_axis1 = max(max_axis1, file[key].shape[0])
        max_axis2 = max(max_axis2, file[key].shape[1])
        
    return count, max_axis1, max_axis2


def replace_nan(matrix):
    '''
    replace or delete nan in matrix.
    if the nan in this row is over half values, delete this row
    if the nan is this row is less half values, replace with mean of columns of each nan
    代替或者删除矩阵中的nan
    如果一行中的nan超过一半，则删除这一行
    如果一行中的nan少于一半，则用nan所在的列的平均值代替该nan
    Parameters
    ----------
    matrix : TYPE 2-dims ndarray
        DESCRIPTION. Input matrix needing be dealt with nan
        需要被替换nan的矩阵

    Returns
    -------
    matrix : TYPE 2-dims ndarray
        DESCRIPTION. Output matrix without nan
        没有nan的输出矩阵
    deleted_rows : TYPE List
        DESCRIPTION. the index of deleted rows
        被删除nan的行的索引
    '''
    n_rows, n_cols = matrix.shape
    nan_indices = np.argwhere(np.isnan(matrix))
    deleted_rows = set()
    for i, j in nan_indices:
        row = matrix[i, :]
        if np.isnan(row).sum() >= n_cols / 2:
            deleted_rows.add(i)
        elif np.isnan(row).sum() <= n_cols / 2 and np.isfinite(row).sum() > n_cols / 2:
            col_mean = np.nanmean(matrix[:, j])
            matrix[i, j] = col_mean
    if deleted_rows:
        matrix = np.delete(matrix, list(deleted_rows), axis=0)
        print("Deleted rows:", deleted_rows)
    return matrix, deleted_rows

def features_selection(weights, features, threshold):
    '''
    Using weights of chanels to reduce dims of features
    adding the features with high weights until the weights is cumulative to threshold
    将权重高的特征通道累加，直到权重叠加达到阈值,选取达到阈值的那一个通道的特征

    Parameters
    ----------
    weights : TYPE 1-dim ndarray
        DESCRIPTION.  
    features : TYPE 2-dims ndarray
        DESCRIPTION.
    threshold : TYPE float
        DESCRIPTION.

    Returns
    -------
    features_new : TYPE 1-dims ndarray
        DESCRIPTION. features with dim reduction

    '''
    num_features = len(weights)
    sorted_indices = np.argsort(weights)[::-1]  # 将权重从高到低排序的索引

    cumulative_weight = 0
    for i in sorted_indices:
        cumulative_weight += weights[i]
        features_new = np.inner(weights[i], features[i])
        if cumulative_weight >= threshold:
            break

    return features_new

def features_with_max_attn(weights, features):
    '''
    Using weights of chanels to reduce dims of features
    Using the features with the highest weight
    选取attn最大的的那一个通道的特征

    Parameters
    ----------
    weights : TYPE 1-dim ndarray
        DESCRIPTION.  
    features : TYPE 2-dims ndarray
        DESCRIPTION.

    Returns
    -------
    features_new : TYPE 1-dims ndarray
        DESCRIPTION. features with dim reduction

    '''
    sorted_indices = np.argsort(weights)[::-1]
    features_new = features[sorted_indices[0]]
    return features_new


def features_pooling(weights, features, threshold):
    '''
    Using weights of chanels to reduce dims of features
    adding the features with high weights until the weights is cumulative to threshold
    将权重高的特征通道累加，直到权重重叠加达到阈值

    Parameters
    ----------
    weights : TYPE 1-dim ndarray
        DESCRIPTION.  
    features : TYPE 2-dims ndarray
        DESCRIPTION.
    threshold : TYPE float
        DESCRIPTION.

    Returns
    -------
    features_new : TYPE 1-dims ndarray
        DESCRIPTION. features with dim reduction

    '''
    length_features = features.shape[1]
    sorted_indices = np.argsort(weights)[::-1]  # 将权重从高到低排序的索引
    features_new = np.zeros(length_features)

    cumulative_weight = 0
    idx_under_threshold = []
    #features_new = np.zeros(features.shape[1])
    for i in sorted_indices:
        cumulative_weight += weights[i]
        idx_under_threshold.append(i)
        #features_new += np.inner(weights[i], features[i])
        if cumulative_weight >= threshold:
            features_new = (np.inner(weights[idx_under_threshold], features[idx_under_threshold].T) )
            break
        
    return features_new



def get_features(attn_array, feature_array, threshold):
    '''
    Get the feautres of each patient
    

    Parameters
    ----------
    attn_array : TYPE 2-d array
        DESCRIPTION. Attention of each channels and patients, M*N
        M is the dimension of patients
        N is the dimension of channels
    feature_array : TYPE 3-d array
        DESCRIPTION. features of each channels and patients, M*N*Z
        M is the dimension of patients
        N is the dimension of channels
        Z is the length of features
    threshold : TYPE
        DESCRIPTION. Threshold for features Dimension Reduction

    Returns
    -------
    features : TYPE 2-d array
        DESCRIPTION. features of patients, M*Z
        The ouput features matrix is a 2-D matrix,reprenting the features of patients
        axis1 is the index of patients, axis2 is the length of features.
        
        eg. features[0] is a vector which means the feature of patient 0.

    '''
    num = len(attn_array)
    num_features = feature_array.shape[2]
    features = np.zeros(  (num, num_features)  )
    for i in range(num):
        #features[i] = features_selection(attn_array[i], feature_array[i], 0.5 )
        features[i] = features_pooling(attn_array[i], feature_array[i], 0.5 )
    return features  

def replace_with_median(arr, threshold):
    '''
    Repplace the absolute values over threshold with median of this row.

    Parameters
    ----------
    arr : TYPE 2-d array
        DESCRIPTION.
    threshold : TYPE float
        DESCRIPTION.

    Returns
    -------
    arr_copy : TYPE 2-d array
        DESCRIPTION.
        The output array is after replacement.
    '''
    # 使用NumPy创建数组的副本，以便不会修改原始数组
    arr_copy = np.copy(arr)
    
    # 找到大于阈值的元素的索引
    indices = np.where(abs(arr_copy) > threshold)[0]
    
    # 将大于阈值的元素替换为剩余数值的中位数
    median = np.median(arr_copy)
    arr_copy[indices] = median
    
    return arr_copy

def filt_singlevalue(features, threshold = 100):
    '''
    filt the siglevalue (absolute values over threshold) of features

    Parameters
    ----------
    features : TYPE  2-d array 
        DESCRIPTION.
        features of patient

    Returns
    -------
    features : TYPE 2-d array
        DESCRIPTION.
        features after filtering

    '''
    num = len(features)
    for i in range(num):
        features[i] = replace_with_median(features[i], threshold)
        
    return features

def create_adjacency_matrix(data, threshold = 0.999):
    '''
    Create adjacency matrix by Pearson correlation
    If the correlation is over threshold, which means these two subject is similar
    useing 1 represents edges between them

    Parameters
    ----------
    data : TYPE 2-d array
        DESCRIPTION.
        Behaviors and physiology information  
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.999.
        
    Returns
    -------
    adjacency_matrix : TYPE
        DESCRIPTION.
        adjacency matrix, this matrix is symmetrical by this way
        NO actition is needed to adjust it to symmetrical and add I

    '''
    # 计算皮尔逊相关系数矩阵
    correlation_matrix = np.corrcoef(data, rowvar=False)
    
    # 构建邻接矩阵
    adjacency_matrix = np.zeros_like(correlation_matrix)
    adjacency_matrix[correlation_matrix > threshold] = 1  # 设置相关系数阈值
    
    return adjacency_matrix

def encode_onehot(labels):
    '''
    Encode the label to onehot format

    Parameters
    ----------
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    labels_onehot : TYPE
        DESCRIPTION.

    '''
    classes = set(labels)
    print("4",classes)

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}    # identity() 输入n为行数或列数，返回一个n*n的对角阵，对角线元素为1，其余为0。通过这种方式，就把每个类别标签向量化了
    #print("5",classes_dict)
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # get() 函数返回指定键的值。
    #print("6",labels_onehot)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix,按行标准化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, cmap=plt.cm.Blues):
    """
    画出模型的混淆矩阵

    y_true: 真实标签
    y_pred: 预测标签
    classes: 各类别的标签
    normalize: 是否将混淆矩阵进行归一化处理
    cmap: 矩阵颜色映射
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 如果classes 为None，则从y_true和y_pred中获取类别标签
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred), axis=0))

    # 如果normalize为True，则将混淆矩阵进行归一化处理
    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # 添加数值标注
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # 添加坐标轴标签
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
