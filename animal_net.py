import os
import ssl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd

# 禁用SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context

# 数据集根目录
data_folder = '/Users/lxl/code/animal/4-animal-classification'

def main():
    train_data, label_to_index = collectData()
    trainModel(train_data)
    predicateData(label_to_index)

def collectData():
    # 读取训练集图片及其相关类别标签
    train_data = []
    label_mapping = {'cat': 0, 'deer': 1, 'dog': 2, 'horse': 3}  # 类别标签到数字标签的映射

    # 读取训练数据
    for class_label, class_index in label_mapping.items():
        class_folder = os.path.join(data_folder, 'train', class_label)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_folder, img_name)
                    train_data.append({'image_path': img_path, 'numeric_label': class_index})

    return train_data, label_mapping

def trainModel(train_data):
    # 定义自定义数据集
    class CustomDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 获取图像路径和标签
            img_path = self.data[idx]['image_path']
            numeric_label = self.data[idx]['numeric_label']

            # 读取图像
            img = Image.open(img_path).convert('RGB')

            # 数据增强和转换
            if self.transform:
                img = self.transform(img)

            return img, numeric_label

    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 创建数据加载器
    dataset = CustomDataset(data=train_data, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 定义模型
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练权重

    # 修改全连接层，输出类别数为4
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(10):  # 例如，训练10个epoch
        for inputs, labels in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/10, Loss: {loss.item()}')

    # 模型训练完成，可以保存模型并在测试集上进行评估
    torch.save(model.state_dict(), 'trained_model.pth')

def predicateData(label_to_index):
    # 创建与之前训练时相同架构的模型
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练权重

    # 修改全连接层，输出类别数为4
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)

    # 从文件中加载之前训练好的模型参数
    model.load_state_dict(torch.load('trained_model.pth'))

    # 将模型设置为评估模式
    model.eval()

    # 读取测试集图片
    predictions = []
    predictions_folder = os.path.join(data_folder, 'test/test')
    for file_name in os.listdir(predictions_folder):
        if file_name.endswith('.jpg'):
            # 从文件名中提取 id
            image_id = int(file_name.split('.')[0])

            # 读取图像
            img_path = os.path.join(predictions_folder, file_name)
            img = Image.open(img_path).convert('RGB')

            # 数据预处理
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img = transform(img)

            # 添加 batch 维度
            img = img.unsqueeze(0)

            # 模型推理
            with torch.no_grad():
                output = model(img)

            # 获取预测结果
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label.item()

            # 将结果添加到列表中
            predictions.append({'ID': image_id, 'Label': predicted_label})

            # 在终端输出预测结果
            # print(f'Image ID: {image_id}, Predicted Label: {predicted_label}')

    # 保存预测结果到 submission.csv
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('4-animal-classification/Sample_submission.csv', index=False)

if __name__ == "__main__":
    main()
