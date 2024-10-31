# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

import time

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 每个 epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('训练完成，共用时 {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('最佳验证集准确率: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # 设置数据目录
    data_dir = 'split-dataset'  # 替换为您的数据集路径

    # 数据预处理和增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),              # 转为张量
            transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),             # 调整大小
            transforms.CenterCrop(224),         # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # 创建训练和验证数据集
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        ) for x in ['train', 'val']
    }

    # 创建数据加载器
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=32,
            shuffle=True,
            num_workers=4  # 可根据需要调整，Windows 上需要注意多进程问题
        ) for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # 检查是否可用 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载预训练的 VGG16 模型
    model = models.vgg16(pretrained=True)

    # 冻结卷积层参数
    for param in model.features.parameters():
        param.requires_grad = False

    # 修改分类器部分
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 2)  # 猫狗二分类

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 开始训练
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=10)

    # 保存模型
    torch.save(model.state_dict(), 'vgg16_cats_dogs.pth')
