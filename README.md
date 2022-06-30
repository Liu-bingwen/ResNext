# 垃圾分类系统
·在Pytorch架构上利用ResNext101模型进行垃圾分类

·model.py：ResNext101模型的基本参数配置

·split_data.py：将数据集按比例划分成训练集和测试集

·train.py：通过GPU对ResNext101预训练模型进行训练

·predict.py：对输入图片进行预测，输出概率最高的类型及其概率
