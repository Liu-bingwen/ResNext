import os
from shutil import copy, rmtree
import random
import json


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.2

    # 指向你解压后的flower_photos文件夹
    data_root = os.path.join(os.getcwd(),"../raw_data")
    origin_flower_path = os.path.join(data_root, "data")

    json_path = '../rule/garbage_classify_rule.json'

    with open(json_path, "r", encoding="utf-8") as f:
        raw_class = json.load(f)

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in raw_class.values():
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in raw_class.values():
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 处理标签文件
    key_class = {}  # 角标->图片文件名
    images = os.listdir(origin_flower_path)
    for cla in images:
        if ".txt" in cla:
            str = open(os.path.join(origin_flower_path, cla), "r").readline()
            a = str.split(",")[0]  # 图片文件名
            b = str.split(",")[1]  # 标签角标
            b = b.replace(' ', '')
            if b not in key_class:
                key_class[b] = [a]
            else:
                key_class[b].append(a)

    for key in key_class.keys():
        list_path = key_class[key]
        num = len(list_path)
        eval_index = random.sample(list_path, k=int(num * split_rate))
        for index, image in enumerate(list_path):
            if image in eval_index:  # 训练集
                image_path = os.path.join(origin_flower_path, image)
                new_path = os.path.join(val_root, raw_class[key])
                copy(image_path, new_path)
            else:  # 测试集
                image_path = os.path.join(origin_flower_path, image)
                new_path = os.path.join(train_root, raw_class[key])
                copy(image_path, new_path)
        print("[{}] processing {} images".format(raw_class[key], num))
    print("success!!")


if __name__ == '__main__':
    main()
