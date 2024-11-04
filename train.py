import numpy as np
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_data(preprocessed_file, train_part):
    """
    加载预处理的数据并划分训练集和测试集。

    参数：
    - preprocessed_file: str, 预处理后的数据文件路径
    - train_indices: array, 训练集索引
    - test_indices: array, 测试集索引

    返回：
    - X_train, y_train, X_test, y_test
    """
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    # 打印出来看看features和label对不对
    print("训练集特征形状：", features.shape)
    print("训练集标签形状：", labels.shape)

    # 提取训练集和测试集
    # 确保 X_train 是二维数组
    # 判断实验现在是第几部分
    # 如果是第一部分的话索引就是0-2000
    # 如果是第二部分的话索引就是2001-4000
    if train_part == 1:
        X_train = features[0:2000]
        y_train = labels[0:2000]
    if train_part == 2:
        X_train = features[2001:4000]
        y_train = labels[2001:4000]

    print("训练集特征形状:", X_train.shape)
    print("训练集标签形状:", y_train.shape)

    return X_train, y_train


def train_svm(X_train, y_train, C=1.0):
    """
    使用 SVM 进行训练。

    参数：
    - X_train: array, 训练特征
    - y_train: array, 训练标签
    - C: float, SVM 正则化参数

    返回：
    - classifier: 训练好的 SVM 分类器
    """
    classifier = svm.SVC(C=C, kernel='linear', decision_function_shape='ovr')
    classifier.fit(X_train, y_train)
    return classifier

def save_classifier(classifier, filename):
    """
    保存训练好的分类器到文件中。

    参数：
    - classifier: 训练好的分类器
    - filename: str, 保存的文件名

    返回：
    - None
    """
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"分类器已保存至 {filename}")

# 主程序
if __name__ == "__main__":
    preprocessed_file = 'preprocessed_data.pkl'
    train_part = 1 # 因为实验分为两个部分，所以这个参数用来指定实验是第一部分还是第二部分，这会影响到训练集的索引范围
    # train_indices = np.arange(0, 2000)  # 直接指定训练集的索引范围为 1 到 2000
    # train_indices = np.arange(2001, 4000)  # 直接指定训练集的索引范围为 2001 到 4000

    X_train, y_train = load_data(preprocessed_file, train_part)

    # 训练 SVM 模型
    print("开始训练 SVM 模型...")
    classifier = train_svm(X_train, y_train, C=1.0)

    # 保存训练好的分类器
    save_classifier(classifier, 'trained_classifier.pkl')
