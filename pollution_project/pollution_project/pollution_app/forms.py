from django import forms
from .models import DataUpload
from .models import PollutionData
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


class DataUploadForm(forms.ModelForm):
    class Meta:
        model = DataUpload
        fields = ['file']
        widgets = {
            'file': forms.FileInput(attrs={'accept': '.csv'})
        }

class PollutionDataForm(forms.ModelForm):
    class Meta:
        model = PollutionData
        fields = ['date', 'pm25', 'pm10', 'so2', 'co', 'no2',  'o3',  'temperature', 'pressure','precipitation',  'humidity', 'wind_speed', 'wind_direction']
        widgets = {
            'date': forms.DateTimeInput(
                attrs={'type': 'datetime-local', 'class': 'form-control'}  # 直接添加class
            ),
            'pm25': forms.NumberInput(attrs={'class': 'form-control'}),
            'pm10': forms.NumberInput(attrs={'class': 'form-control'}),
            'so2': forms.NumberInput(attrs={'class': 'form-control'}),
            'co': forms.NumberInput(attrs={'class': 'form-control'}),
            'no2': forms.NumberInput(attrs={'class': 'form-control'}),
            'o3': forms.NumberInput(attrs={'class': 'form-control'}),
            'temperature': forms.NumberInput(attrs={'class': 'form-control'}),
            'pressure': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'precipitation': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'humidity': forms.NumberInput(attrs={'class': 'form-control'}),
            'wind_speed': forms.NumberInput(attrs={'class': 'form-control'}),
            'wind_direction': forms.NumberInput(attrs={'class': 'form-control', 'step': '1'}),
        }

# pollution_app/views.py
from django.shortcuts import render, redirect
from .forms import DataUploadForm
import pandas as pd
from django.db import transaction
from .models import PollutionData, DataUpload
import csv
from django.utils import timezone


def upload_data(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save()
            try:
                # 处理CSV文件
                with open(upload.file.path, 'r', encoding='utf-8-sig') as file:
                    reader = csv.DictReader(file)
                    records = []
                    count = 0

                    for row in reader:
                        # CSV列名：date, pm25, pm10, temperature, humidity, wind_speed
                        records.append(PollutionData(
                            date=pd.to_datetime(row['date']),
                            pm25=float(row['pm25']),
                            pm10=float(row['pm10']),
                            temperature=float(row['temperature']),
                            humidity=float(row['humidity']),
                            wind_speed=float(row.get('wind_speed', 0))  # 可选字段
                        ))
                        count += 1

                        # 每1000条记录提交一次，避免内存溢出
                        if count % 1000 == 0:
                            with transaction.atomic():
                                PollutionData.objects.bulk_create(records)
                            records = []

                    # 提交剩余记录
                    if records:
                        with transaction.atomic():
                            PollutionData.objects.bulk_create(records)

                    # 更新上传状态
                    upload.status = '成功'
                    upload.records_processed = count
                    upload.save()

                    return redirect('upload_success')

            except Exception as e:
                upload.status = '失败'
                upload.error_message = str(e)
                upload.save()
                return render(request, 'upload.html', {
                    'form': form,
                    'error': f'上传失败: {str(e)}'
                })
    else:
        form = DataUploadForm()
    return render(request, 'upload.html', {'form': form})


def upload_success(request):
    return render(request, 'upload_success.html')


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user

class LoginForm(AuthenticationForm):
    username = forms.CharField(label='用户名')
    password = forms.CharField(label='密码', widget=forms.PasswordInput)