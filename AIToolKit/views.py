import base64
import io
import logging

from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from matplotlib import pyplot as plt

from djangoProject import settings
from .forms import PredictionForm,TrainForm
from .models.MyNet import MyNet
from .models.YoloV8 import YoloV8
from .models.ResNet50 import ResNet50
from .models.mobileNet import MobileNet
from .models.AlexNet import AlexNet
from .models.XGBoostModel import XGBoostModel
from .models.Lightgbm import LightGBMModel
from .models.RandomForestModel import RandomForestModel
from .models.utils import model_predict, analyze_3,train_for_localCSV,train_model,save_model
from .utils.dataset_loader import predict_loader
import os
import sys
from io import StringIO

def index(request):
    return render(request, 'AIToolKit/index.html')

def predict_mode(request):
    return render(request, 'AIToolKit/predict_mode.html')
# 设置日志收集器
logger = logging.getLogger(__name__)
def predict(request):
    logs = []  # 初始化日志列表
    if request.method == 'POST':
        form = PredictionForm(request.POST, request.FILES)
        if form.is_valid():
            task = form.cleaned_data['task']
            model = form.cleaned_data['model']
            text_input = form.cleaned_data['text_input']
            image_file = request.FILES.get('image_file')
            if image_file:
                # 保存上传的文件
                image_path = os.path.join('AIToolKit', 'Media', image_file.name)
                with open(image_path, 'wb+') as destination:
                    for chunk in image_file.chunks():
                        destination.write(chunk)
                logs.append(f"Image saved to {image_path}")  # 记录日志

            result = None
            # 按照任务类型和模型创建分析器并进行预测
            if task == "文本分析":
                if model in ["PatternAnalyzer", "NaiveBayesAnalyzer"]:
                    # 直接调用模型的预测方法
                    result = analyze_3(model, text_input)
                    logs.append(f"Text analysis completed using {model}")
            elif task == "图像分割":
                if model == "YoloV8":
                    analyzer = YoloV8()
                    result = model_predict(analyzer, image_path)
                    logs.append(f"Image segmentation completed using {model}")
            elif task == "图像识别":
                if model == "ResNet50":
                    analyzer = ResNet50()
                    result = model_predict(analyzer, image_path)
                    logs.append(f"Image recognition completed using ResNet50")
                elif model == "MobileNet":
                    analyzer = MobileNet()
                    result = model_predict(analyzer, image_path)
                    logs.append(f"Image recognition completed using MobileNet")

            if result:
                if task in ['图像分割', '图像识别'] and os.path.exists(result):
                    response_url = request.build_absolute_uri('/AIToolKit/results/') + os.path.basename(result)
                    logs.append(f"Result available at {response_url}")
                    return JsonResponse({'image_url': response_url, 'logs': logs})
                else:
                    return JsonResponse({'result': result, 'logs': logs})
            else:
                logs.append("No result available or image not processed")
                return JsonResponse({'error': "No result or image not processed", 'logs': logs}, status=400)
        else:
            logs.append("Form validation failed")
            return JsonResponse({'error': 'Invalid form data', 'logs': logs}, status=400)

    logs.append("Invalid request method")
    return JsonResponse({'error': 'Invalid request', 'logs': logs}, status=400)


@csrf_exempt
def train_model_view(request):
    logs = []  # 初始化日志列表

    if request.method == 'POST':
        dataset_type = request.POST.get('dataset_type')
        model_type = request.POST.get('model_type')
        try:
            epochs = int(request.POST.get('epochs', 8))
            learning_rate = float(request.POST.get('learning_rate', 0.01))
        except (ValueError, TypeError):
            logs.append("Invalid epochs or learning rate")
            return JsonResponse({'error': 'Invalid epochs or learning rate', 'logs': logs}, status=400)

        # 根据请求中的 model_type 动态选择模型
        if model_type == 'MyNet':
            model = MyNet()
        elif model_type == 'ResNet':
            model = ResNet50().getModel()
        elif model_type == 'MobileNet':
            model = MobileNet().get_model()
        elif model_type == 'AlexNet':
            model = AlexNet()
        else:
            logs.append("Invalid model type selected")
            return JsonResponse({'error': 'Invalid model type selected', 'logs': logs}, status=400)

        logs.append(f"Training started with {model_type} on {dataset_type}")
        Accuracy = train_model(model, dataset_type, epochs, learning_rate)

        # 展开嵌套的列表（如果是嵌套的）
        if any(isinstance(i, list) for i in Accuracy):
            Accuracy = [item for sublist in Accuracy for item in sublist]

        # 根据准确率计算损失
        loss = [100 - acc for acc in Accuracy]
        logs.append(f"Training completed. Final accuracy: {Accuracy[-1]}")

        # 返回损失数据和日志信息
        return JsonResponse({
            'loss_plot': loss,
            'logs': logs
        })

    # 如果是GET请求，渲染一个HTML页面或返回默认响应
    return render(request, 'AIToolKit/train_mode.html')



def local_CSVdataset_train(request):
    if request.method == 'POST':
        # 获取前端表单中的数据
        model1_type = request.POST.get('model1_type')
        model2_type = request.POST.get('model2_type')
        filepath = request.POST.get('filepath')
        features = request.POST.get('features').split(',')
        target = request.POST.get('target')
        test_size = float(request.POST.get('test_size'))
        random_state = int(request.POST.get('random_state'))

        # 初始化模型
        model1 = get_model(model1_type)
        model2 = get_model(model2_type)

        # 训练模型并获取训练与测试准确率
        train_acc1, test_acc1 = train_for_localCSV(model1, filepath, features, target, test_size, random_state)
        train_acc2, test_acc2 = train_for_localCSV(model2, filepath, features, target, test_size, random_state)

        # 将模型类型和准确率包装在JSON中返回
        response_data = {
            'models': [model1_type, model2_type],
            'train_accuracies': [train_acc1, train_acc2],
            'test_accuracies': [test_acc1, test_acc2]
        }

        return JsonResponse(response_data)
    else:
        # 如果不是POST请求，返回主页面
        return render(request, 'AIToolKit/train_mode.html')


def get_model(model_type):
    """根据模型类型初始化并返回模型实例"""
    if model_type == 'XGBoostModel':
        return XGBoostModel()
    elif model_type == 'Lightgbm':
        return LightGBMModel()
    elif model_type == 'RandomForestModel':
        return RandomForestModel()
    else:
        raise ValueError("Unsupported model type")

