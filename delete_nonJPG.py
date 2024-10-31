import os
import shutil

def merge_all_folders_in_directory(parent_directory, target_folder):
    try:
        # 如果目标文件夹不存在，则创建它
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # 遍历父目录中的所有文件夹
        for folder_name in os.listdir(parent_directory):
            folder_path = os.path.join(parent_directory, folder_name)
            
            # 确保只处理文件夹
            if os.path.isdir(folder_path):
                # 遍历源文件夹中的所有文件
                for filename in os.listdir(folder_path):
                    # 获取文件完整路径
                    source_file = os.path.join(folder_path, filename)
                    
                    # 检查文件是否是jpg文件
                    if os.path.isfile(source_file) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # 构建目标文件路径
                        target_file = os.path.join(target_folder, filename)
                        
                        # 如果目标文件已经存在，则重命名以避免冲突
                        counter = 1
                        while os.path.exists(target_file):
                            name, ext = os.path.splitext(filename)
                            target_file = os.path.join(target_folder, f"{name}_{counter}{ext}")
                            counter += 1
                        
                        # 复制文件到目标文件夹
                        shutil.copy2(source_file, target_file)
                        print(f"Copied: {source_file} to {target_file}")
        print("所有文件已成功合并到目标文件夹。")
    except Exception as e:
        print(f"出错了: {e}")

# 设定父目录和目标文件夹
parent_directory = r"C:\Users\feixi\Downloads\low-resolution\low-resolution"  # 替换为你的父目录路径
target_folder = r"C:\Users\feixi\Downloads\low-resolution\merged"  # 替换为你的目标文件夹路径

# 调用函数
merge_all_folders_in_directory(parent_directory, target_folder)
