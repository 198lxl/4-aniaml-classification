# 4-aniaml-classification
### 动物分类项目要求

- **任务**: 使用深度学习模型对4种动物（猫、鹿、狗、马）的图像进行分类。
- **数据集**:
  - 训练数据：2800张图像，分为4个类别（猫、鹿、狗、马）。
  - 测试数据：729张图像。
- **标签**: 猫/鹿/狗/马 -> 0/1/2/3
- **提交文件**: 提交的样例文件 `Sample_submission.csv`。

### 项目结构

```
4-animal-classification/
├── test/
│   └── test/
├── train/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   └── horse/
├── Sample_ubmission.csv
└── animal_net.py
```

### 实现代码详细分析

#### 导入必要的库

```python
import os
import ssl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
```

这些库用于处理文件路径、图像操作和深度学习模型的训练与评估。

#### 禁用SSL证书验证

```python
ssl._create_default_https_context = ssl._create_unverified_context
```

禁用SSL证书验证以避免网络请求中的SSL证书错误。

#### 数据集根目录

```python
data_folder = '/Users/lxl/code/animal/4-animal-classification'
```

定义数据集的根目录，用于后续的数据读取和处理。

#### 主函数

```python
def main():
    train_data, label_to_index = collectData()
    trainModel(train_data)
    predicateData(label_to_index)

if __name__ == "__main__":
    main()
```

主函数 `main()` 依次调用数据收集、模型训练和数据预测的函数。

#### 数据收集函数

```python
def collectData():
    train_data = []
    label_mapping = {'cat': 0, 'deer': 1, 'dog': 2, 'horse': 3}  # 类别标签到数字标签的映射

    for class_label, class_index in label_mapping.items():
        class_folder = os.path.join(data_folder, 'train', class_label)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.endswith('.jpg'):
                    img_path = os.path.join(class_folder, img_name)
                    train_data.append({'image_path': img_path, 'numeric_label': class_index})

    return train_data, label_mapping
```

1. 定义 `label_mapping`：将类别标签（猫、鹿、狗、马）映射到数字标签（0, 1, 2, 3）。
2. 遍历每个类别文件夹，读取其中所有 `.jpg` 文件，存储图像路径和对应的数字标签。

#### 模型训练函数

```python
def trainModel(train_data):
    class CustomDataset(Dataset):
        def __init__(self, data, transform=None):
            self.data = data
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_path = self.data[idx]['image_path']
            numeric_label = self.data[idx]['numeric_label']
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, numeric_label
```

1. 定义 `CustomDataset` 类，用于自定义数据集的读取和预处理。
   - `__init__`：初始化数据和转换操作。
   - `__len__`：返回数据集的大小。
   - `__getitem__`：读取指定索引的图像及其标签，并进行预处理。

```python
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
```

2. 定义图像预处理操作，将图像调整为 224x224 大小并转换为 Tensor。

```python
    dataset = CustomDataset(data=train_data, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

3. 使用 `CustomDataset` 类创建数据集实例，并使用 `DataLoader` 创建数据加载器，以批次方式加载数据并打乱顺序。

```python
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练权重

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
```

4. 加载预训练的 ResNet-18 模型，并将最后的全连接层修改为输出 4 个类别。

```python
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

5. 定义交叉熵损失函数和 Adam 优化器。

```python
    for epoch in range(10):  # 例如，训练10个epoch
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/10, Loss: {loss.item()}')
```

6. 训练模型：
   - 前向传播：将输入数据传入模型，计算输出。
   - 计算损失：使用交叉熵损失函数计算预测结果与真实标签之间的差异。
   - 反向传播和优化：计算梯度，并使用优化器更新模型参数。

```python
    torch.save(model.state_dict(), 'trained_model.pth')
```

7. 模型训练完成后，保存训练好的模型参数到文件 `trained_model.pth`。

#### 数据预测函数

```python
def predicateData(label_to_index):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 加载预训练权重

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()
```

1. 创建与训练时相同的模型架构，并加载训练好的模型参数。
2. 将模型设置为评估模式，以禁用 Dropout 等训练时特定操作。

```python
    predictions = []
    predictions_folder = os.path.join(data_folder, 'test/test')
    for file_name in os.listdir(predictions_folder):
        if file_name.endswith('.jpg'):
            image_id = int(file_name.split('.')[0])
            img_path = os.path.join(predictions_folder, file_name)
            img = Image.open(img_path).convert('RGB')
```
3. 遍历测试集文件夹中的每张图片文件，提取图片 ID 并读取图像。

```python
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            img = transform(img)
            img = img.unsqueeze(0)
```

4. 对图像进行预处理，将图像调整为 224x224 大小并转换为 Tensor，添加批次维度。

```python
            with torch.no_grad():
                output = model(img)
            _, predicted_label = torch.max(output, 1)
            predicted_label = predicted_label.item()
            predictions.append({'ID': image_id, 'Label': predicted_label})
            print(f'Image ID: {image_id}, Predicted Label: {predicted_label}')
```

5. 模型推理：
   - 使用模型进行预测，获取输出结果。
   - 获取预测的标签（取最大概率对应的类别）。
   - 将预测结果（图片 ID 和标签）添加到列表 `predictions` 中，并在终端输出预测结果。

```python
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('Sample_submission.csv', index=False)
```

6. 将预测结果保存到 `Sample_submission.csv` 文件中。
