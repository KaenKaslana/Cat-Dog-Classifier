import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 加载模型
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)  # 猫狗二分类
model.load_state_dict(torch.load('vgg16_cats_dogs.pth', map_location=torch.device('cpu')))
model.eval()

# 检查是否可用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义类别名称
classes = ['cat', 'dog']  # 根据您的实际类别名称

# 图像预处理
data_transforms = transforms.Compose([
    transforms.Resize(256),             # 调整大小
    transforms.CenterCrop(224),         # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    # 预处理图像
    image = data_transforms(image)
    image = image.unsqueeze(0)  # 添加批次维度
    image = image.to(device)

    # 前向传播
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    predicted_class = classes[preds[0]]
    return predicted_class

if __name__ == '__main__':
    img_path = 'black_cat.png'  # 替换为您要预测的图像路径
    result = predict_image(img_path)
    print(f"模型预测结果：{result}")
