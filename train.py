import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import resnext101_32x8d


def show_train(index, myWin, len_bar):
    myWin.index = index + 1
    number = ((myWin.index / len_bar) * 100)
    myWin.progressBar.setValue(number)


def main(epochs, myWin):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnext101_32x8d()
    save_path = './resNext101.pth'  # 模型保存路径
    model_weight_path = "./resnext101_32x8d.pth"  # 预训练模型路径

    class_num = 40  # 分类个数

    batch_size = 3
    nw = 8

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.CenterCrop((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((256, 256)),
                                   transforms.CenterCrop((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_path = os.path.join(os.getcwd(), "../raw_data")  # 数据集路径

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # # 将标签保存到新的json文件中
    raw_list = train_dataset.class_to_idx
    raw_list = dict((val, key) for key, val in raw_list.items())  # 倒转字典
    json_str = json.dumps(raw_list, indent=4)
    with open('../rule/new_classify_rule.json', 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("[{} images for training and {} images for validation]".format(len(train_dataset), len(validate_dataset)))

    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    #  只训练最后一层
    for param in model.parameters():
        param.requires_grad = False

    # change fc layer structure
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, class_num)
    model.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        print("epoch [{}/{}] training".format(epoch + 1, epochs))
        # train
        model.train()
        running_loss = 0.0

        if isinstance(myWin, str):
            train_bar = tqdm(train_loader, file=sys.stdout)
        else:
            train_bar = train_loader

        if not isinstance(myWin, str):
            myWin.progressBar.setValue(0)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            if isinstance(myWin, str):
                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
            else:
                show_train(step, myWin, len(train_bar))

            running_loss += loss.item()

        # validate
        model.eval()
        acc = 0.0
        print("epoch [{}/{}] validating".format(epoch + 1, epochs))
        with torch.no_grad():
            if isinstance(myWin, str):
                val_bar = tqdm(validate_loader, file=sys.stdout)
            else:
                val_bar = validate_loader

            if not isinstance(myWin, str):
                myWin.progressBar.setValue(0)

            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                if isinstance(myWin, str):
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
                else:
                    show_train(step, myWin, len(val_bar))

        val_accurate = acc / len(validate_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
        print()

    print("success!!")


if __name__ == '__main__':
    epochs = 1  # 迭代次数
    main(epochs, "null")
