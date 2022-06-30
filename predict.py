import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnext101_32x8d


def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    json_path = '../rule/new_classify_rule.json'
    model = resnext101_32x8d(num_classes=40).to(device)

    data_transform = transforms.Compose(
        [transforms.Resize((256, 256)),  # 缩放最大边=256
         transforms.CenterCrop((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image

    img = Image.open(img_path)
    # [N, C, H, W]
    img = data_transform(img)
    # 增加维度
    img = torch.unsqueeze(img, dim=0)

    # 获取标签
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    # print("class_indict",class_indict)

    # create model

    # load model weights
    weights_path = "./resNext101.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
    print()
    print("print_res:", predict_cla,print_res)


if __name__ == '__main__':
    main("./test_data/1.jpg")
