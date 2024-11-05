import torch
import torch.nn as nn
from torchvision import models, transforms
from captum.attr import LayerGradCam
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载模型
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)  # 假设有两个类别：猫和狗
model.load_state_dict(torch.load('model/vgg16_cats_dogs.pth', map_location=torch.device('cpu')))
model.eval()

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 使用 ImageNet 的均值和标准差
                         std=[0.229, 0.224, 0.225])
])

# 加载并预处理图像
img_path = 'white_cat.png'  # 替换为您的图像路径
img = Image.open(img_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# 定义目标层
target_layer = model.features[29]  # VGG16 的最后一个卷积层

# 初始化 LayerGradCam
layer_gc = LayerGradCam(model, target_layer)

# 获取模型输出并预测类别
output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()

# 计算 Grad-CAM
attributions = layer_gc.attribute(input_tensor, target=predicted_class)

# 上采样到输入图像大小
attributions = torch.nn.functional.interpolate(attributions, size=(224, 224), mode='bilinear')

# 将张量转换为 NumPy 数组并应用 ReLU
grad_cam = attributions.squeeze().cpu().detach().numpy()
grad_cam = np.maximum(grad_cam, 0)

# 归一化
grad_cam = grad_cam / grad_cam.max()

# 准备原始图像
input_image = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
input_image = input_image * np.array([0.229, 0.224, 0.225])  # 反归一化
input_image = input_image + np.array([0.485, 0.456, 0.406])
input_image = np.clip(input_image, 0, 1)

# 创建热力图
heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
heatmap = heatmap[..., ::-1]  # BGR 转 RGB

# 叠加热力图到原始图像
superimposed_img = heatmap * 0.4 + input_image

# 显示结果
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(superimposed_img)
plt.show()

# 可选：保存结果
plt.imsave('white_cam_pytorch.jpg', superimposed_img)
