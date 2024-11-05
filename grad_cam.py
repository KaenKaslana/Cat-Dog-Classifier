import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# 定义一个 Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda

        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, index=None):
        if self.cuda:
            features, output = self.extractor(input_img.cuda())
        else:
            features, output = self.extractor(input_img)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # 清除梯度
        self.model.zero_grad()
        # 计算目标类别的得分
        one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot[0][index] = 1
        if self.cuda:
            one_hot = one_hot.cuda()

        # 反向传播
        output.backward(gradient=one_hot, retain_graph=True)

        # 获取梯度和特征图
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()[0, :]
        target = features[-1].cpu().data.numpy()[0, :]

        # 计算权重
        weights = np.mean(grads_val, axis=(1, 2))

        # 计算加权和
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # ReLU
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_img.shape[2], input_img.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

# 定义一个辅助类，用于提取特征和梯度
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermediate layers.
    3. Gradients from intermediate layers.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # 提取特征
        target_activations, x = self.feature_extractor(x)

        # 通过分类器
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x)
        return target_activations, x

# 定义特征提取器
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from target intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # 前向传播
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

# 图像预处理
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   normalize
])

# 加载模型
model = models.vgg16(pretrained=False)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('model/vgg16_cats_dogs.pth', map_location=torch.device('cpu')))
model.eval()

# 检查是否可用 GPU
use_cuda = torch.cuda.is_available()

# 选择目标层
grad_cam = GradCAM(model=model, target_layer_names=["29"], use_cuda=use_cuda)

# 加载图像
img_path = 'white_cat.png'  # 替换为您的图像路径
img = Image.open(img_path).convert('RGB')
img_tensor = preprocess(img)
img_variable = img_tensor.unsqueeze(0)

# 生成 Grad-CAM
mask = grad_cam(img_variable)

# 可视化
heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('white_cam.jpg', result)
