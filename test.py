import numpy as np
import pickle
from sklearn.metrics import accuracy_score


def load_data(preprocessed_file, train_part):
    """
    加载预处理的数据并划分测试集。

    参数：
    - preprocessed_file: str, 预处理后的数据文件路径
    - test_indices: array, 测试集索引

    返回：
    - X_test, y_test
    """
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)

    features = data['features']
    labels = data['labels']

    # 提取测试集
    # 判断实验现在是第几部分
    # 如果是第一部分的话索引就是2001-4000
    # 如果是第二部分的话索引就是0-2000
    if train_part == 1:
        X_test = features[2001:4000]
        y_test = labels[2001:4000]
    if train_part == 2:
        X_test = features[0:2000]
        y_test = labels[0:2000]

    return X_test, y_test


def load_classifier(filename):
    """
    从文件加载分类器。

    参数：
    - filename: str, 文件名

    返回：
    - classifier: 加载的分类器
    """
    with open(filename, 'rb') as f:
        classifier = pickle.load(f)
    print(f"分类器从 {filename} 加载成功")
    return classifier


def evaluate_model(classifier, X_test, y_test):
    """
    在测试集上评估模型的准确率。

    参数：
    - classifier: 训练好的分类器
    - X_test: array, 测试特征
    - y_test: array, 测试标签

    返回：
    - accuracy: float, 模型在测试集上的准确率
    """
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 主程序
if __name__ == "__main__":
    preprocessed_file = 'preprocessed_data.pkl'
    train_part = 1 # 因为实验分为两个部分，所以这个参数用来指定实验是第一部分还是第二部分，这会影响到训练集的索引范围

    # 加载测试集
    X_test, y_test = load_data(preprocessed_file, train_part)

    # 加载训练好的分类器
    classifier = load_classifier('trained_classifier.pkl')

    # 测试 SVM 模型
    accuracy = evaluate_model(classifier, X_test, y_test)
    print(f"SVM 模型的准确率: {accuracy}")
