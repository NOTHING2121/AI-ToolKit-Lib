from django import forms

class PredictionForm(forms.Form):
    TASK_CHOICES = [
        ('图像分割', '图像分割'),
        ('图像识别', '图像识别'),
        ('文本分析', '文本分析'),
    ]

    MODEL_CHOICES = [
        ('YoloV8', 'YoloV8'),
        ('ResNet50', 'ResNet50'),
        ('MobileNet', 'MobileNet'),
        ('PatternAnalyzer', 'PatternAnalyzer'),
        ('NaiveBayesAnalyzer', 'NaiveBayesAnalyzer'),
    ]

    task = forms.ChoiceField(choices=TASK_CHOICES, required=True)
    model = forms.ChoiceField(choices=MODEL_CHOICES, required=True)
    text_input = forms.CharField(widget=forms.Textarea, required=False)
    image_file = forms.ImageField(required=False)



class TrainForm(forms.Form):
    MODEL_CHOICES = [
        ('XGBoostModel', 'XGBoostModel'),
        ('Lightgbm', 'Lightgbm'),
        ('RandomForestModel', 'RandomForestModel'),
        ]
    model_type = forms.ChoiceField(choices=MODEL_CHOICES)
    filepath = forms.CharField(max_length=1000)
    features = forms.CharField(max_length=1000)  # 用户输入特征名称，以逗号分隔
    target = forms.CharField(max_length=100)
