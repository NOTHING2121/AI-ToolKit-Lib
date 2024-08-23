import torchvision
from torchvision import transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import pandas as pd


def predict_loader(image_path):
    # 定义图像预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 加载并预处理图像
    img = Image.open(image_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加批次维度
    return img,img_tensor
def upload_dataset(path,filetype):
    if filetype=='fashionMNIST':
        dataset=datasets.FashionMNIST(path,train=True,download=True)
    elif filetype=='cifar10':
        dataset=datasets.CIFAR10(path,train=True,download=True)
    elif filetype=='MNIST':
        dataset=datasets.MNIST(path,train=True,download=True)
    return dataset

def preprocessing(dataset,filetype):
    filetype=filetype.lower()
    if filetype=='fashionmnist':
        # 针对FashionMNIST的transform
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道
            transforms.Resize(224),
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对每个通道进行归一化
        ])
        dataset.transform = transform
    elif filetype=='cifar10':
        # 针对CIFAR-10的transform
        transform = transforms.Compose([
            transforms.Resize(224),  # CIFAR-10 图像原始大小是 32x32
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对每个通道进行归一化
        ])
        dataset.transform = transform

    elif filetype=='mnist':
        # 针对MNIST的transform
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3通道
            transforms.Resize(224),  # 确保图像大小一致
            transforms.ToTensor(),  # 将图片转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对每个通道进行归一化
        ])
        dataset.transform = transform
    else:
        print('File type not supported')
    return dataset

def train_loader_image(path, filetype,batch_size=64,):


    dataset=preprocessing(upload_dataset(path,filetype),filetype)

    # 将数据集划分为训练集和测试集（80%训练，20%测试）
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def upload_local_csv_file(filePath,features,target_column):
    # 确保所有指定的列都存在于数据中
    X= pd.DataFrame()
    df = pd.read_csv(filePath)
    missing_columns = set(features + [target_column]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in CSV data: {missing_columns}")
    for col in features:
        if col in df.columns:
            X[col]=df[col]
    y=df[target_column]
    return X,y
