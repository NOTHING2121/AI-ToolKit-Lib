from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim, nn
from xgboost import XGBClassifier

from AIToolKit.utils.dataset_loader import predict_loader
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.sentiments import PatternAnalyzer
import torch
from AIToolKit.utils.dataset_loader import upload_local_csv_file,train_loader_image
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import djangoProject.settings as settings
import matplotlib.pyplot as plt

def model_predict(model, image_path):
    model_name = model.__class__.__name__
    if model_name == 'ResNet50' or model_name == 'MobileNet':
        return analyze_1(model.getModel(), image_path)
    elif model_name.lower() == 'yolov8':
        return analyze_2(model.get_model(), image_path)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def analyze_1(model, image_path):
    model.eval()
    img, img_tensor = predict_loader(image_path)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    with open(r"C:\Users\Yjw\PycharmProjects\pythonProject\MainApplication\models\imagenet_classes.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # 获取预测的类别标签
    predicted_label = labels[predicted.item()]

    # 在图像上绘制预测结果
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"Predicted: {predicted_label}"
    draw.text((10, 10), text, font=font, fill=(255, 0, 0))

    # 将处理后的图像存储到 results 文件夹中
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    processed_image_path = os.path.join(results_dir, "processed_image_with_label.jpg")
    img.save(processed_image_path)


    # 返回预测结果和处理后的图像路径
    return  processed_image_path

def analyze_2(model, image_path):
    results = model(image_path, task='segment')

    # 提取并可视化分割结果
    segmented_image = results[0].plot()  # 获取分割后的图像

    # 将处理后的图像存储到 results 文件夹中
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    segmented_image_path = os.path.join(results_dir, "segmented_image.jpg")
    cv2.imwrite(segmented_image_path, segmented_image)

    return segmented_image_path

def analyze_3(model, text):
    if model == 'NaiveBayesAnalyzer':
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        return (f"Classification: {blob.sentiment.classification}, "
                f"Positive Probability: {blob.sentiment.p_pos}, "
                f"Negative Probability: {blob.sentiment.p_neg}")
    elif model == 'PatternAnalyzer':
        blob = TextBlob(text, analyzer=PatternAnalyzer())
        return f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}"
    else:
        return "Invalid analyzer type. Please choose 'NaiveBayesAnalyzer' or 'PatternAnalyzer'."

def train_model(model,filetype,epochs=8, learning_rates=0.01):
    test_loader,train_loader=train_loader_image('./dataset',filetype)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer=optim.SGD(model.parameters(),lr=learning_rates,momentum=0.9)
    criterion=nn.CrossEntropyLoss()
    accuracies=[]
    results=[]
    for epoch in range(epochs):
        train(model,device,train_loader,optimizer,criterion)
        accuracy = test(model, test_loader,device,criterion)
        accuracies.append(accuracy)
        print(f'Learning Rate: {learning_rates} | Epoch: {epoch} | Accuracy: {accuracy}%')
    results.append(accuracies)
    return results


def train(model,device,train_loader,optimizer,criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model,test_loader,device,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

#为本地上传文件训练对应模型
def train_for_localCSV(model, filepath, features,target,test_size=0.2, random_state=42):
    X,y=upload_local_csv_file(filepath,features,target)
    """
    训练模型，并返回训练和测试的准确度。
    """
    if isinstance(model.get_model(), XGBClassifier):
        label_encoder=LabelEncoder()
        y=label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    return train_accuracy, test_accuracy

def save_model(model, directory='./save_model', filename='saved_model'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    return torch.save(model.state_dict(), filepath)

def plot_accuracy(models,train_accuracies,test_accuracies):
   n = len(models)  # 模型的数量
   fig, ax = plt.subplots()
   index = range(n)
   bar_width = 0.35
   opacity = 0.8

   # 绘制训练准确率条形
   train_bars = ax.bar(index, train_accuracies, bar_width, alpha=opacity, label='Train Accuracy')

   # 绘制测试准确率条形
   test_bars = ax.bar([p + bar_width for p in index], test_accuracies, bar_width, alpha=opacity, label='Test Accuracy')

   ax.set_xlabel('Model')
   ax.set_ylabel('Accuracy')
   ax.set_title('Train and Test Accuracy Comparison')
   ax.set_xticks([p + bar_width / 2 for p in index])
   ax.set_xticklabels(models)
   ax.legend()

   plt.show()
