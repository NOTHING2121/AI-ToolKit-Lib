import os

from django.urls import path
from django.views.static import serve

from . import views

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

app_name = 'AIToolKit'
urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('predict_mode/', views.predict_mode, name='predict_mode'),
    path('train_model_view/', views.train_model_view, name='train_mode'),
    # 添加这行
    path('local_CSVdataset_train/', views.local_CSVdataset_train, name='local_CSVdataset_train'),
    # path('load_image_dataset_train/',views.load_image_dataset_train,name='load_image_dataset_train'),

]

