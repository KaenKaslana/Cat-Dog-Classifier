import os
import shutil
import random

def split_dataset(original_dataset_dir, base_dir, split_ratio=0.8):
    # 创建训练和验证目录
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # 获取类别列表，例如 ['cats', 'dogs']
    classes = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]

    for cls in classes:
        # 定义原始类别目录和目标类别目录
        cls_dir = os.path.join(original_dataset_dir, cls)
        train_cls_dir = os.path.join(train_dir, cls)
        validation_cls_dir = os.path.join(validation_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(validation_cls_dir, exist_ok=True)

        # 获取所有文件名
        filenames = os.listdir(cls_dir)
        random.shuffle(filenames)
        split_point = int(len(filenames) * split_ratio)
        train_filenames = filenames[:split_point]
        validation_filenames = filenames[split_point:]

        # 复制文件到训练集
        for fname in train_filenames:
            src = os.path.join(cls_dir, fname)
            dst = os.path.join(train_cls_dir, fname)
            shutil.copyfile(src, dst)

        # 复制文件到验证集
        for fname in validation_filenames:
            src = os.path.join(cls_dir, fname)
            dst = os.path.join(validation_cls_dir, fname)
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    original_dataset_dir = 'good-data'  # 您的原始数据集路径，包含'cats'和'dogs'文件夹
    base_dir = 'split-dataset'  # 划分后的数据集存储路径
    split_ratio = 0.8  # 训练集所占比例，0.8表示80%用于训练，20%用于验证

    split_dataset(original_dataset_dir, base_dir, split_ratio)
