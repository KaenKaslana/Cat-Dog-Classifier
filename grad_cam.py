import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# 加载模型
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('vgg16_cats_dogs.pth'))
model.eval()

# 检查是否可用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 注册 hook 函数
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.cpu().data.numpy())

# 获取最后一个卷积层的名称
finalconv_name = 'features'

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# 定义图像预处理
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   normalize
])

# 定义函数计算 Grad-CAM
def returnCAM(feature_conv, weight_softmax, class_idx):
    # 生成 CAM
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# 获取 softmax 权重
params = list(model.parameters())
weight_softmax = params[-2].cpu().data.numpy()

# 加载图像
img_path = 'image.png'  # 替换为您的图像路径
img_pil = Image.open(img_path)
img_tensor = preprocess(img_pil)
img_variable = img_tensor.unsqueeze(0).to(device)

# 前向传播
logit = model(img_variable)

# 预测结果
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.cpu().numpy()
idx = idx.cpu().numpy()

# 输出预测结果
classes = ['cat', 'dog']  # 请根据您的实际类别修改
for i in range(0, 2):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# 生成 CAM
CAMs = returnCAM(features_blobs[0], weight_softmax, idx[0])

# 显示并保存结果
img = cv2.imread(img_path)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('cam.jpg', result)
