from django.shortcuts import render, redirect, HttpResponse,get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.contrib.auth.forms import UserChangeForm
from django.db import models
from .models import (
    PollutionData, PredictionResult, DataUpload,
    UserPollutant, SearchHistory
)
from .models import PollutionData
from .forms import DataUploadForm, RegisterForm, LoginForm, PollutionDataForm
from django.utils import timezone
from datetime import timedelta, date, datetime
from .prediction_model import predict_pollution, train_and_save_model, batch_predict_pollution
from .prediction_model import load_model
import pandas as pd
import csv
from django.db import transaction
from django.contrib.auth.models import User
import chardet
import os
import json
from io import StringIO, BytesIO

# 自定义个人信息表单（只允许修改部分字段）
class ProfileForm(UserChangeForm):
    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name']  # 可修改的字段
        exclude = ['password']  # 隐藏密码字段（后续修改密码可单独做功能）

# 监测分析
def monitoring_analysis(request):
    """
        监测分析页面视图函数
        可根据需要传递污染物数据等上下文
        """
    # 如需在监测分析页面展示污染物数据，可在此查询并传递
    # 示例：获取所有污染物类型供页面使用
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

    return render(request, 'pollution_app/monitoring.html', {
        'pollutants': pollutants,
        # 可添加其他需要传递到模板的数据
    })

def pollution_map(request):
    # 实际项目中可在此获取并传递真实的污染数据
    return render(request, 'pollution_app/pollution_map.html')

def sounding_view(request):
    """处理探空图页面的视图函数"""
    # 实际应用中可以在这里获取探空数据并传递到模板
    context = {
        'page_title': '贺州市气象探空图',
        # 可以添加实际的探空数据
    }
    return render(request, 'pollution_app/sounding.html', context)


def trend_chart(request):
    """
    走势图页面视图函数
    提供污染物走势图所需的数据
    """
    # 可以在这里查询数据库获取实际数据
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']

    return render(request, 'pollution_app/trend_chart.html', {
        'pollutants': pollutants,
    })


def windrose(request):
    """
    风玫瑰图页面视图函数
    提供风玫瑰图所需的污染物和气象数据
    """
    # 可根据实际需求从数据库获取数据
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
    wind_directions = ['北', '东北', '东', '东南', '南', '西南', '西', '西北']

    return render(request, 'pollution_app/windrose.html', {
        'pollutants': pollutants,
        'wind_directions': wind_directions
    })



# 数据管理
def data_management(request):
    """数据管理页面视图函数"""
    upload_success = False
    upload_error = None
    records_processed = 0

    if request.method == 'POST':
        # 处理数据上传
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
                        records.append(PollutionData(
                            date=pd.to_datetime(row['date']),
                            pm25=float(row['pm25']),
                            pm10=float(row['pm10']),
                            no2=float(row.get('no2', 0)),
                            so2=float(row.get('so2', 0)),
                            o3=float(row.get('o3', 0)),
                            co=float(row.get('co', 0)),
                            temperature=float(row['temperature']),
                            humidity=float(row['humidity']),
                            wind_speed=float(row.get('wind_speed', 0))
                        ))
                        count += 1

                        if count % 1000 == 0:
                            PollutionData.objects.bulk_create(records)
                            records = []

                    if records:
                        PollutionData.objects.bulk_create(records)

                    upload.status = '成功'
                    upload.records_processed = count
                    upload.save()

                    upload_success = True
                    records_processed = count

            except Exception as e:
                upload.status = '失败'
                upload.error_message = str(e)
                upload.save()
                upload_error = str(e)
    else:
        form = DataUploadForm()

    return render(request, 'pollution_app/data_management.html', {
        'upload_success': upload_success,
        'upload_error': upload_error,
        'records_processed': records_processed
    })


def export_data(request):
    """数据导出功能"""
    # 获取请求参数
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    pollutants = request.GET.getlist('pollutants', ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co'])
    export_format = request.GET.get('format', 'csv')

    # 验证日期参数
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return HttpResponse("日期格式错误，请使用YYYY-MM-DD", status=400)

    # 查询数据
    queryset = PollutionData.objects.filter(
        date__date__gte=start_date,
        date__date__lte=end_date
    ).order_by('date')

    if not queryset.exists():
        return HttpResponse("指定日期范围内没有数据", status=404)

    # 准备导出数据
    columns = ['date'] + pollutants + ['temperature', 'humidity', 'wind_speed']

    # 处理CSV格式
    if export_format == 'csv':
        output = StringIO()
        writer = csv.writer(output)
        # 写入表头
        header = ['日期时间'] + [
            {'pm25': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO₂',
             'so2': 'SO₂', 'o3': 'O₃', 'co': 'CO'}[p] for p in pollutants
        ] + ['温度(℃)', '湿度(%)', '风速']
        writer.writerow(header)

        # 写入数据
        for item in queryset:
            row = [item.date.strftime('%Y-%m-%d %H:%M')]
            for p in pollutants:
                row.append(getattr(item, p, ''))
            row.extend([item.temperature, item.humidity, item.wind_speed])
            writer.writerow(row)

        response = HttpResponse(output.getvalue(), content_type='text/csv')
        response[
            'Content-Disposition'] = f'attachment; filename="pollution_data_{start_date_str}_to_{end_date_str}.csv"'
        return response

    # 处理Excel格式（需要安装pandas和openpyxl）
    elif export_format == 'excel':
        data = []
        for item in queryset:
            row = {'日期时间': item.date.strftime('%Y-%m-%d %H:%M')}
            for p in pollutants:
                row[{'pm25': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO₂',
                     'so2': 'SO₂', 'o3': 'O₃', 'co': 'CO'}[p]] = getattr(item, p, '')
            row['温度(℃)'] = item.temperature
            row['湿度(%)'] = item.humidity
            row['风速'] = item.wind_speed
            data.append(row)

        df = pd.DataFrame(data)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='污染数据')

        response = HttpResponse(output.getvalue(),
                                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response[
            'Content-Disposition'] = f'attachment; filename="pollution_data_{start_date_str}_to_{end_date_str}.xlsx"'
        return response

    return HttpResponse("不支持的导出格式", status=400)

# 用户注册
def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # 默认关注一些污染物
            default_pollutants = ['pm25', 'pm10']
            for pollutant in default_pollutants:
                UserPollutant.objects.create(user=user, pollutant_code=pollutant)
            login(request, user)
            messages.success(request, f'注册成功！欢迎 {user.username}')
            return redirect('index')
    else:
        form = RegisterForm()
    return render(request, 'pollution_app/register.html', {'form': form})


# 用户登录
def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f'您已登录为 {username}')
                
                # 生成JWT令牌
                from .utils.jwt_auth import generate_jwt_token
                token = generate_jwt_token(user)
                
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'code': 200,
                        'message': '登录成功',
                        'token': token,
                        'user': {
                            'id': user.id,
                            'username': user.username,
                            'email': user.email
                        }
                    })
                
                return redirect('index')
            else:
                messages.error(request, '用户名或密码不正确')
                # 如果是AJAX请求，返回JSON响应
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({'code': 401, 'message': '用户名或密码不正确'}, status=401)
        else:
            messages.error(request, '用户名或密码不正确')
            # 如果是AJAX请求，返回JSON响应
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'code': 401, 'message': '用户名或密码不正确'}, status=401)
    else:
        form = LoginForm()
    return render(request, 'pollution_app/login.html', {'form': form})


# 用户退出
def user_logout(request):
    # 移除请求方法判断，无论GET/POST都执行退出
    logout(request)  # 清除登录状态
    return redirect('index')  # 退出后重定向到首页


from django.views.decorators.cache import never_cache

@never_cache
def index(request):
    # 初始化context
    context = {}

    # 获取前端筛选参数
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    pollutants = request.GET.getlist('pollutants', ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co'])

    # 处理日期变量
    start_date = None
    end_date = None
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        except ValueError:
            pass

    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
            pass

    # 用户登录状态处理
    if request.user.is_authenticated:
        # 保存查询历史
        SearchHistory.objects.create(
            user=request.user,
            start_date=start_date,
            end_date=end_date,
            pollutants=','.join(pollutants)
        )
        # 获取用户关注的污染物
        user_pollutants = UserPollutant.objects.filter(
            user=request.user
        ).values_list('pollutant_code', flat=True)
        context['user_pollutants'] = list(user_pollutants)
    else:
        context['user_pollutants'] = []

    # 初始化查询集
    queryset = PollutionData.objects.all().order_by('-date')

    # 处理日期筛选
    if start_date_str:
        try:
            queryset = queryset.filter(date__date__gte=start_date)
        except ValueError:
            messages.warning(request, "开始日期格式错误，请使用YYYY-MM-DD")

    if end_date_str:
        try:
            queryset = queryset.filter(date__date__lte=end_date)
        except ValueError:
            messages.warning(request, "结束日期格式错误，请使用YYYY-MM-DD")

    # 默认取最近7天数据
    if not start_date_str and not end_date_str:
        queryset = queryset.filter(date__gte=timezone.now() - timedelta(days=7))

    # 限制表格显示数量
    recent_data = queryset[:20]

    # 格式化图表数据
    chart_data = {
        'dates': [item.date.strftime('%Y-%m-%d %H:%M') for item in queryset],
        'pollutants': {}
    }
    for pollutant in pollutants:
        if hasattr(PollutionData, pollutant):
            chart_data['pollutants'][pollutant] = [getattr(item, pollutant) for item in queryset]
        else:
            chart_data['pollutants'][pollutant] = []

    # 生成预测结果
    if not PredictionResult.objects.filter(target_date__gte=date.today()).exists():
        generate_future_predictions()

    # 获取未来3天的预测结果
    future_predictions = PredictionResult.objects.filter(
        target_date__gte=date.today(),
        target_date__lte=date.today() + timedelta(days=3)
    ).order_by('target_date')

    # 获取城市污染数据（AQI排名）
    from .models import CityPollutionData
    city_pollution_data = CityPollutionData.objects.order_by('aqi')
    
    # 获取AQI前五（最好）和后五（最差）的城市
    top5_cities = city_pollution_data[:5]  # AQI最小的5个城市（最好）
    bottom5_cities = city_pollution_data.order_by('-aqi')[:5]  # AQI最大的5个城市（最差）
    
    # 合并数据，先显示最好的5个，再显示最差的5个
    ranked_cities = list(top5_cities) + list(bottom5_cities)
    
    # 获取最新的空气质量数据用于概览
    latest_data = PollutionData.objects.order_by('-date').first()
    if latest_data:
        # 计算AQI指数（简化版）
        def calculate_aqi(pm25, pm10, o3):
            # 简化的AQI计算，实际应使用标准公式
            aqi = 0.5 * pm25 + 0.3 * pm10 + 0.2 * o3
            return min(500, max(0, int(aqi)))
        
        aqi = calculate_aqi(latest_data.pm25, latest_data.pm10, latest_data.o3)
        
        # 确定空气质量等级
        def get_aqi_level(aqi):
            if aqi <= 50:
                return "优", "text-success"
            elif aqi <= 100:
                return "良", "text-primary"
            elif aqi <= 150:
                return "轻度污染", "text-warning"
            elif aqi <= 200:
                return "中度污染", "text-orange"
            elif aqi <= 300:
                return "重度污染", "text-danger"
            else:
                return "严重污染", "text-dark"
        
        aqi_level, aqi_class = get_aqi_level(aqi)
        
        # 确定各污染物等级
        def get_pollutant_level(name, value):
            levels = {
                'pm25': [(35, "优", "text-success"), (75, "良", "text-primary"), (115, "轻度污染", "text-warning"), (150, "中度污染", "text-orange"), (250, "重度污染", "text-danger")],
                'pm10': [(50, "优", "text-success"), (150, "良", "text-primary"), (250, "轻度污染", "text-warning"), (350, "中度污染", "text-orange"), (420, "重度污染", "text-danger")],
                'o3': [(100, "优", "text-success"), (160, "良", "text-primary"), (215, "轻度污染", "text-warning"), (265, "中度污染", "text-orange"), (800, "重度污染", "text-danger")]
            }
            
            for limit, level, css_class in levels.get(name, []):
                if value <= limit:
                    return level, css_class
            return "重度污染", "text-danger"
        
        pm25_level, pm25_class = get_pollutant_level('pm25', latest_data.pm25)
        pm10_level, pm10_class = get_pollutant_level('pm10', latest_data.pm10)
        o3_level, o3_class = get_pollutant_level('o3', latest_data.o3)
        
        # 添加到context
        context.update({
            'latest_data': latest_data,
            'aqi': aqi,
            'aqi_level': aqi_level,
            'aqi_class': aqi_class,
            'pm25_level': pm25_level,
            'pm25_class': pm25_class,
            'pm10_level': pm10_level,
            'pm10_class': pm10_class,
            'o3_level': o3_level,
            'o3_class': o3_class
        })
    else:
        # 如果没有数据，使用默认值
        context.update({
            'latest_data': None,
            'aqi': 68,
            'aqi_level': "良",
            'aqi_class': "text-primary",
            'pm25_level': "良好",
            'pm25_class': "text-success",
            'pm10_level': "轻度污染",
            'pm10_class': "text-warning",
            'o3_level': "良好",
            'o3_class': "text-success"
        })

    # 更新context
    context.update({
        'recent_data': recent_data,
        'predictions': future_predictions,
        'chart_data': json.dumps(chart_data),
        'selected_pollutants': pollutants,
        'start_date': start_date_str or '',
        'end_date': end_date_str or '',
        'ranked_cities': ranked_cities
    })
    return render(request, 'pollution_app/index.html', context)

# 个人信息修改视图
@login_required
def view_profile(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, '个人信息修改成功！')
            return redirect('view_profile')
    else:
        form = ProfileForm(instance=request.user)
    return render(request, 'pollution_app/profile.html', {'form': form})

# 切换关注的污染物
@login_required
def toggle_pollutant(request, pollutant_code):
    if pollutant_code in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
        user_pollutant, created = UserPollutant.objects.get_or_create(
            user=request.user,
            pollutant_code=pollutant_code
        )
        if not created:
            user_pollutant.delete()
    return redirect('index')


# 查看用户查询历史
@login_required
def view_history(request):
    histories = SearchHistory.objects.filter(user=request.user).order_by('-search_time')[:10]
    return render(request, 'pollution_app/history.html', {'histories': histories})


def generate_future_predictions():
    """生成未来3天的污染预测结果并保存到数据库"""
    latest_data = PollutionData.objects.order_by('-date').first()

    if latest_data:
        # 删除旧的预测结果（避免重复）
        PredictionResult.objects.filter(target_date__gte=date.today()).delete()

        # 生成未来3天的预测
        for days_ahead in range(1, 4):
            target_date = date.today() + timedelta(days=days_ahead)

            # 1. 线性回归模型预测（假设准确率85%）
            linear_pred = predict_pollution(
                date=target_date,
                temperature=latest_data.temperature,
                humidity=latest_data.humidity,
                wind_speed=latest_data.wind_speed,
                pm25=latest_data.pm25,
                pm10=latest_data.pm10,
                no2=latest_data.no2,
                so2=latest_data.so2,
                o3=latest_data.o3,
                co=latest_data.co,
                model_type='linear'  # 指定线性模型
            )

            # 2. 神经网络模型预测（假设准确率90%）
            nn_pred = predict_pollution(
                date=target_date,
                temperature=latest_data.temperature,
                humidity=latest_data.humidity,
                wind_speed=latest_data.wind_speed,
                pm25=latest_data.pm25,
                pm10=latest_data.pm10,
                no2=latest_data.no2,
                so2=latest_data.so2,
                o3=latest_data.o3,
                co=latest_data.co,
                model_type='nn'  # 指定神经网络模型
            )

            # 3. 后端随机森林模型预测
            backend_pred = predict_pollution(
                date=target_date,
                temperature=latest_data.temperature,
                humidity=latest_data.humidity,
                wind_speed=latest_data.wind_speed,
                pm25=latest_data.pm25,
                pm10=latest_data.pm10,
                no2=latest_data.no2,
                so2=latest_data.so2,
                o3=latest_data.o3,
                co=latest_data.co,
                model_type='backend_rf'  # 指定后端随机森林模型
            )

            # 4. 保存三模型预测结果到数据库
            PredictionResult.objects.create(
                target_date=target_date,
                # 线性回归结果
                pm25_pred=linear_pred['pm25'],
                pm10_pred=linear_pred['pm10'],
                no2_pred=linear_pred['no2'],
                so2_pred=linear_pred['so2'],
                o3_pred=linear_pred['o3'],
                co_pred=linear_pred['co'],
                linear_accuracy=85.0,  # 线性模型准确率

                # 神经网络结果
                pm25_nn_pred=nn_pred['pm25'],
                pm10_nn_pred=nn_pred['pm10'],
                no2_nn_pred=nn_pred['no2'],
                so2_nn_pred=nn_pred['so2'],
                o3_nn_pred=nn_pred['o3'],
                co_nn_pred=nn_pred['co'],
                nn_accuracy=90.0,  # 神经网络模型准确率
                
                # 后端随机森林模型结果
                pm25_rf_pred=backend_pred['pm25'],
                pm10_rf_pred=backend_pred['pm10'],
                no2_rf_pred=backend_pred['no2'],
                so2_rf_pred=backend_pred['so2'],
                o3_rf_pred=backend_pred['o3'],
                co_rf_pred=backend_pred['co'],
                rf_accuracy=80.0  # 后端模型准确率（假设值）
            )




def upload_data(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.save(commit=False)
            upload.status = '处理中'
            upload.save()

            try:
                # 检测文件编码
                with open(upload.file.path, 'rb') as f:
                    result = chardet.detect(f.read(10000))
                    file_encoding = result['encoding'] or 'utf-8'

                # 读取CSV文件
                with open(upload.file.path, 'r', encoding=file_encoding) as file:
                    reader = csv.DictReader(file)
                    records = []
                    count = 0

                    for row in reader:
                        # 验证并解析数据
                        try:
                            date_val = pd.to_datetime(row['date'])
                            pm25 = float(row['pm25'])
                            pm10 = float(row['pm10'])
                            no2 = float(row.get('no2', 0))
                            so2 = float(row.get('so2', 0))
                            o3 = float(row.get('o3', 0))
                            co = float(row.get('co', 0))
                            temperature = float(row['temperature'])
                            humidity = float(row['humidity'])
                            wind_speed = float(row.get('wind_speed', 0))
                            wind_direction = float(row.get('wind_direction', 0))
                            pressure = float(row.get('pressure', 0))
                            precipitation = float(row.get('precipitation', 0))
                        except (KeyError, ValueError) as e:
                            upload.status = '失败'
                            upload.error_message = f"第 {count + 1} 行数据格式错误: {str(e)}"
                            upload.save()
                            messages.error(request, f"上传失败: {upload.error_message}")
                            return redirect('upload_data')

                        # 创建记录
                        records.append(PollutionData(
                            date=date_val,
                            pm25=pm25,
                            pm10=pm10,
                            no2=no2,
                            so2=so2,
                            o3=o3,
                            co=co,
                            temperature=temperature,
                            humidity=humidity,
                            wind_speed=wind_speed
                        ))
                        count += 1

                        # 批量提交（每1000条）
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

                    messages.success(request, f"成功导入 {count} 条记录")
                    return redirect('upload_data')

            except Exception as e:
                upload.status = '失败'
                upload.error_message = str(e)
                upload.save()
                messages.error(request, f"上传失败: {str(e)}")
                return redirect('upload_data')
    else:
        form = DataUploadForm()

    # 获取最近的上传记录
    recent_uploads = DataUpload.objects.order_by('-upload_date')[:5]

    return render(request, 'pollution_app/upload_data.html', {
        'form': form,
        'recent_uploads': recent_uploads
    })

# 数据上传页面（原数据管理的上传功能）
def data_upload(request):
    upload_success = False
    upload_error = None
    records_processed = 0

    if request.method == 'POST':
        # 复用原数据上传逻辑
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
                        records.append(PollutionData(
                            date=pd.to_datetime(row['date']),
                            pm25=float(row['pm25']),
                            pm10=float(row['pm10']),
                            no2=float(row.get('no2', 0)),
                            so2=float(row.get('so2', 0)),
                            o3=float(row.get('o3', 0)),
                            co=float(row.get('co', 0)),
                            temperature=float(row['temperature']),
                            humidity=float(row['humidity']),
                            wind_speed=float(row.get('wind_speed', 0)),
                            wind_direction=float(row.get('wind_direction', 0)),  # 新增
                            pressure=float(row.get('pressure', 0)),  # 新增
                            precipitation=float(row.get('precipitation', 0))  # 新增
                        ))
                        count += 1

                        if count % 1000 == 0:
                            PollutionData.objects.bulk_create(records)
                            records = []

                    if records:
                        PollutionData.objects.bulk_create(records)

                    upload.status = '成功'
                    upload.records_processed = count
                    upload.save()

                    upload_success = True
                    records_processed = count

            except Exception as e:
                upload.status = '失败'
                upload.error_message = str(e)
                upload.save()
                upload_error = str(e)
    else:
        form = DataUploadForm()

    return render(request, 'pollution_app/data_upload.html', {
        'upload_success': upload_success,
        'upload_error': upload_error,
        'records_processed': records_processed,
        'form': form
    })


# 历史数据查询页面
def history_query(request):
    # 仅返回查询页面，逻辑保留在前端JS
    return render(request, 'pollution_app/history_query.html')


# 数据导出页面
def data_export(request):
    # 仅返回导出页面，处理逻辑仍用原export_data函数
    return render(request, 'pollution_app/data_export.html')


def upload_success(request):
    return render(request, 'pollution_app/upload_success.html')


def test_view(request):
    return HttpResponse("测试视图正常工作")


# API登录视图
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def api_login(request):
    """
    API登录视图，返回JWT令牌
    """
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
        except json.JSONDecodeError:
            return JsonResponse({'code': 400, 'message': '无效的请求数据'}, status=400)
        
        if not username or not password:
            return JsonResponse({'code': 400, 'message': '用户名和密码不能为空'}, status=400)
        
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            from .utils.jwt_auth import generate_jwt_token
            token = generate_jwt_token(user)
            return JsonResponse({
                'code': 200,
                'message': '登录成功',
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email
                }
            })
        else:
            return JsonResponse({'code': 401, 'message': '用户名或密码不正确'}, status=401)
    else:
        return JsonResponse({'code': 405, 'message': '只支持POST请求'}, status=405)


# 测试JWT认证的视图
from .decorators import jwt_required

@jwt_required
def test_jwt_auth(request):
    """
    测试JWT认证的视图
    """
    return JsonResponse({
        'code': 200,
        'message': '认证成功',
        'user': {
            'id': request.user.id,
            'username': request.user.username,
            'email': request.user.email
        }
    })


@csrf_exempt
def api_batch_predict(request):
    """
    批量预测API端点
    支持一次性预测多个时间点的污染数据
    """
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body)
            
            # 获取请求参数
            prediction_data = data.get('prediction_data', [])
            model_type = data.get('model_type', 'linear')
            
            if not prediction_data:
                return JsonResponse({'code': 400, 'message': '预测数据不能为空'}, status=400)
            
            # 处理预测数据
            processed_data = []
            for item in prediction_data:
                try:
                    processed_item = {
                        'date': datetime.strptime(item['date'], '%Y-%m-%d'),
                        'temperature': float(item['temperature']),
                        'humidity': float(item['humidity']),
                        'wind_speed': float(item['wind_speed']),
                        'pm25': float(item['pm25']),
                        'pm10': float(item['pm10']),
                        'no2': float(item['no2']),
                        'so2': float(item['so2']),
                        'o3': float(item['o3']),
                        'co': float(item['co'])
                    }
                    processed_data.append(processed_item)
                except Exception as e:
                    return JsonResponse({'code': 400, 'message': f'数据格式错误: {str(e)}'}, status=400)
            
            # 执行批量预测
            results = batch_predict_pollution(processed_data, model_type=model_type)
            
            # 返回结果
            return JsonResponse({
                'code': 200,
                'message': '批量预测成功',
                'data': results,
                'model_type': model_type,
                'count': len(results)
            })
        except json.JSONDecodeError:
            return JsonResponse({'code': 400, 'message': '无效的请求数据'}, status=400)
        except Exception as e:
            return JsonResponse({'code': 500, 'message': f'预测失败: {str(e)}'}, status=500)
    else:
        return JsonResponse({'code': 405, 'message': '只支持POST请求'}, status=405)



def air_pollution_forecast(request):
    """大气污染预测页面视图函数"""
    # 生成预测结果（如果不存在）
    if not PredictionResult.objects.filter(target_date__gte=date.today()).exists():
        generate_future_predictions()
    
    # 获取预测数据
    prediction_objects = PredictionResult.objects.all().order_by('target_date')
    
    # 转换为模板期望的格式
    predictions = {
        'model1': [],  # 线性回归
        'model2': [],  # 神经网络
        'model3': []   # 后端随机森林
    }
    
    for pred in prediction_objects:
        # 模型一（线性回归）
        predictions['model1'].append({
            'target_date': pred.target_date,
            'pm25_pred': pred.pm25_pred,
            'pm10_pred': pred.pm10_pred,
            'no2_pred': pred.no2_pred,
            'so2_pred': pred.so2_pred,
            'o3_pred': pred.o3_pred,
            'co_pred': pred.co_pred,
            'overall_aqi_pred': getattr(pred, 'overall_aqi_pred', 0),
            'main_pollutant_pred': getattr(pred, 'main_pollutant_pred', '无'),
            'aqi_level_pred': getattr(pred, 'aqi_level_pred', '未知')
        })
        
        # 模型二（神经网络）
        predictions['model2'].append({
            'target_date': pred.target_date,
            'pm25_pred': pred.pm25_nn_pred,
            'pm10_pred': pred.pm10_nn_pred,
            'no2_pred': pred.no2_nn_pred,
            'so2_pred': pred.so2_nn_pred,
            'o3_pred': pred.o3_nn_pred,
            'co_pred': pred.co_nn_pred,
            'overall_aqi_pred': getattr(pred, 'overall_aqi_nn_pred', 0),
            'main_pollutant_pred': getattr(pred, 'main_pollutant_nn_pred', '无'),
            'aqi_level_pred': getattr(pred, 'aqi_level_nn_pred', '未知')
        })
        
        # 模型三（后端随机森林）
        predictions['model3'].append({
            'target_date': pred.target_date,
            'pm25_pred': getattr(pred, 'pm25_rf_pred', pred.pm25_pred),
            'pm10_pred': getattr(pred, 'pm10_rf_pred', pred.pm10_pred),
            'no2_pred': getattr(pred, 'no2_rf_pred', pred.no2_pred),
            'so2_pred': getattr(pred, 'so2_rf_pred', pred.so2_pred),
            'o3_pred': getattr(pred, 'o3_rf_pred', pred.o3_pred),
            'co_pred': getattr(pred, 'co_rf_pred', pred.co_pred),
            'overall_aqi_pred': getattr(pred, 'overall_aqi_rf_pred', 0),
            'main_pollutant_pred': getattr(pred, 'main_pollutant_rf_pred', '无'),
            'aqi_level_pred': getattr(pred, 'aqi_level_rf_pred', '未知')
        })

    return render(request, 'pollution_app/air_pollution_forecast.html', {
        'predictions': predictions,
    })


from django.shortcuts import render
import pandas as pd


def prediction_evaluation(request):
    # 模拟评估数据（实际应从数据库或模型训练结果获取）
    evaluation_data = [
        {
            'model': '线性回归',
            'city': '贺州市',
            'train_mae': 8.23,
            'train_rmse': 10.56,
            'train_r2': 0.85,
            'test_mae': 9.12,
            'test_rmse': 11.34,
            'test_r2': 0.82,
            'training_time': 2.45
        },
        {
            'model': '神经网络',
            'city': '贺州市',
            'train_mae': 5.12,
            'train_rmse': 7.34,
            'train_r2': 0.92,
            'test_mae': 6.89,
            'test_rmse': 8.76,
            'test_r2': 0.89,
            'training_time': 45.67
        },
        {
            'model': '后端随机森林',
            'city': '贺州市',
            'train_mae': 7.94,
            'train_rmse': 10.09,
            'train_r2': -8.07,
            'test_mae': 7.94,
            'test_rmse': 10.09,
            'test_r2': -8.07,
            'training_time': 1.5
        },
        # 其他城市和模型数据...
    ]

    # 计算平均指标用于图表
    linear_data = [item for item in evaluation_data if item['model'] == '线性回归']
    nn_data = [item for item in evaluation_data if item['model'] == '神经网络']
    rf_data = [item for item in evaluation_data if item['model'] == '后端随机森林']

    avg_rmse = [
        sum(item['test_rmse'] for item in linear_data) / len(linear_data),
        sum(item['test_rmse'] for item in nn_data) / len(nn_data),
        sum(item['test_rmse'] for item in rf_data) / len(rf_data)
    ]

    avg_r2 = [
        sum(item['test_r2'] for item in linear_data) / len(linear_data),
        sum(item['test_r2'] for item in nn_data) / len(nn_data),
        sum(item['test_r2'] for item in rf_data) / len(rf_data)
    ]

    return render(request, 'pollution_app/prediction_evaluation.html', {
        'evaluation_data': evaluation_data,
        'avg_rmse': avg_rmse,
        'avg_r2': avg_r2
    })


# 数据操作主页（查询/列表）
def data_operation(request):
    # 获取所有已上传数据
    data_list = PollutionData.objects.all().order_by('-date')

    # 处理搜索请求
    search_query = request.GET.get('search', '')
    if search_query:
        # 尝试匹配日期（支持YYYY-MM-DD或YYYY-MM-DD HH:MM格式）
        try:
            # 精确匹配日期时间
            search_date = datetime.strptime(search_query, '%Y-%m-%d')
            data_list = data_list.filter(date__date=search_date)
        except ValueError:
            try:
                search_datetime = datetime.strptime(search_query, '%Y-%m-%d %H:%M')
                data_list = data_list.filter(date=search_datetime)
            except ValueError:
                # 匹配污染物数值（模糊匹配）
                data_list = data_list.filter(
                    models.Q(pm25__icontains=search_query) |
                    models.Q(pm10__icontains=search_query) |
                    models.Q(no2__icontains=search_query) |
                    models.Q(so2__icontains=search_query) |
                    models.Q(o3__icontains=search_query) |
                    models.Q(co__icontains=search_query) |
                    models.Q(temperature__icontains=search_query) |
                    models.Q(humidity__icontains=search_query)
                )

    return render(request, 'pollution_app/data_operation.html', {
        'data_list': data_list,
        'form': PollutionDataForm()  # 用于新增/编辑的表单
    })

# 新增数据
def add_data(request):
    if request.method == 'POST':
        form = PollutionDataForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "数据添加成功！")
            return redirect('data_operation')
    return redirect('data_operation')

# 编辑数据
def edit_data(request, data_id):
    data = get_object_or_404(PollutionData, id=data_id)
    if request.method == 'POST':
        form = PollutionDataForm(request.POST, instance=data)
        if form.is_valid():
            form.save()
            messages.success(request, "数据更新成功！")
            return redirect('data_operation')
    return render(request, 'pollution_app/data_operation.html', {
        'data_list': PollutionData.objects.all().order_by('-date'),
        'form': PollutionDataForm(instance=data),
        'edit_id': data_id  # 标记当前编辑的ID
    })

# 删除数据
def delete_data(request, data_id):
    data = get_object_or_404(PollutionData, id=data_id)
    data.delete()
    messages.success(request, "数据已删除！")
    return redirect('data_operation')


# 切换数据隐藏/显示状态
def toggle_data_visibility(request, data_id):
    data = get_object_or_404(PollutionData, id=data_id)
    data.is_hidden = not data.is_hidden  # 反转状态
    data.save()

    # 提示信息
    status = "隐藏" if data.is_hidden else "显示"
    messages.success(request, f"数据已成功{status}")
    return redirect('data_operation')  # 重定向回数据操作页面


# 系统管理视图
from .decorators import permission_required, role_required


@permission_required('system_settings')
def system_settings(request):
    """
    系统设置页面
    """
    return render(request, 'pollution_app/system_settings.html', {
        'page_title': '系统设置'
    })


@permission_required('user_management')
def user_management(request):
    """
    用户管理页面
    """
    users = User.objects.all()
    return render(request, 'pollution_app/user_management.html', {
        'page_title': '用户管理',
        'users': users
    })


@permission_required('role_management')
def role_management(request):
    """
    角色管理页面
    """
    from .models import Role, Permission
    roles = Role.objects.all()
    permissions = Permission.objects.all()
    return render(request, 'pollution_app/role_management.html', {
        'page_title': '角色管理',
        'roles': roles,
        'permissions': permissions
    })


@permission_required('model_management')
def model_management_view(request):
    """
    模型管理页面
    """
    from .models import ModelEvaluation
    models = ModelEvaluation.objects.all().order_by('-created_at')
    return render(request, 'pollution_app/model_management.html', {
        'page_title': '模型管理',
        'models': models
    })


@permission_required('batch_prediction')
def batch_prediction_view(request):
    """
    批量预测页面
    """
    return render(request, 'pollution_app/batch_prediction.html', {
        'page_title': '批量预测'
    })


@permission_required('prediction_history')
def prediction_history_view(request):
    """
    预测历史页面
    """
    # 获取查询参数
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    model_type = request.GET.get('model_type')
    
    # 构建查询
    query = PredictionResult.objects.all()
    
    # 应用过滤条件
    if start_date:
        try:
            query = query.filter(predict_time__date__gte=start_date)
        except:
            pass
    if end_date:
        try:
            query = query.filter(predict_time__date__lte=end_date)
        except:
            pass
    
    # 按预测时间倒序排序
    predictions = query.order_by('-predict_time')
    
    return render(request, 'pollution_app/prediction_history.html', {
        'page_title': '预测历史',
        'predictions': predictions,
        'start_date': start_date,
        'end_date': end_date,
        'model_type': model_type
    })


@permission_required('prediction_history')
def export_prediction_history(request):
    """
    导出预测历史数据
    """
    # 获取查询参数
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # 构建查询
    query = PredictionResult.objects.all()
    
    # 应用过滤条件
    if start_date:
        try:
            query = query.filter(predict_time__date__gte=start_date)
        except:
            pass
    if end_date:
        try:
            query = query.filter(predict_time__date__lte=end_date)
        except:
            pass
    
    # 按预测时间排序
    predictions = query.order_by('predict_time')
    
    # 准备CSV数据
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # 写入表头
    writer.writerow([
        '预测时间', '目标日期', 
        '线性回归PM2.5', '线性回归PM10', '线性回归NO2', '线性回归SO2', '线性回归O3', '线性回归CO',
        '神经网络PM2.5', '神经网络PM10', '神经网络NO2', '神经网络SO2', '神经网络O3', '神经网络CO',
        '随机森林PM2.5', '随机森林PM10', '随机森林NO2', '随机森林SO2', '随机森林O3', '随机森林CO',
        '线性准确率', '神经网络准确率', '随机森林准确率',
        '综合AQI', '主要污染物', '污染等级'
    ])
    
    # 写入数据
    for pred in predictions:
        writer.writerow([
            pred.predict_time.strftime('%Y-%m-%d %H:%M:%S'),
            pred.target_date.strftime('%Y-%m-%d'),
            pred.pm25_pred,
            pred.pm10_pred,
            pred.no2_pred,
            pred.so2_pred,
            pred.o3_pred,
            pred.co_pred,
            pred.pm25_nn_pred,
            pred.pm10_nn_pred,
            pred.no2_nn_pred,
            pred.so2_nn_pred,
            pred.o3_nn_pred,
            pred.co_nn_pred,
            getattr(pred, 'pm25_rf_pred', 0),
            getattr(pred, 'pm10_rf_pred', 0),
            getattr(pred, 'no2_rf_pred', 0),
            getattr(pred, 'so2_rf_pred', 0),
            getattr(pred, 'o3_rf_pred', 0),
            getattr(pred, 'co_rf_pred', 0),
            pred.linear_accuracy or 0,
            pred.nn_accuracy or 0,
            getattr(pred, 'rf_accuracy', 0),
            pred.overall_aqi_pred or 0,
            pred.main_pollutant_pred or '',
            pred.aqi_level_pred or ''
        ])
    
    # 创建响应
    response = HttpResponse(output.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="prediction_history_{start_date or "all"}_{end_date or "all"}.csv"'
    return response


# API: 获取预测历史数据
@jwt_required
def api_prediction_history(request):
    """
    获取预测历史数据
    """
    # 获取查询参数
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    limit = request.GET.get('limit', 100)
    
    # 构建查询
    query = PredictionResult.objects.all()
    
    # 应用过滤条件
    if start_date:
        try:
            query = query.filter(predict_time__date__gte=start_date)
        except:
            pass
    if end_date:
        try:
            query = query.filter(predict_time__date__lte=end_date)
        except:
            pass
    
    # 限制数量并排序
    try:
        limit = int(limit)
    except:
        limit = 100
    
    predictions = query.order_by('-predict_time')[:limit]
    
    # 构建响应数据
    data = []
    for pred in predictions:
        data.append({
            'id': pred.id,
            'predict_time': pred.predict_time.isoformat(),
            'target_date': pred.target_date.isoformat(),
            'linear': {
                'pm25': pred.pm25_pred,
                'pm10': pred.pm10_pred,
                'no2': pred.no2_pred,
                'so2': pred.so2_pred,
                'o3': pred.o3_pred,
                'co': pred.co_pred,
                'accuracy': pred.linear_accuracy or 0
            },
            'neural_network': {
                'pm25': pred.pm25_nn_pred,
                'pm10': pred.pm10_nn_pred,
                'no2': pred.no2_nn_pred,
                'so2': pred.so2_nn_pred,
                'o3': pred.o3_nn_pred,
                'co': pred.co_nn_pred,
                'accuracy': pred.nn_accuracy or 0
            },
            'random_forest': {
                'pm25': getattr(pred, 'pm25_rf_pred', 0),
                'pm10': getattr(pred, 'pm10_rf_pred', 0),
                'no2': getattr(pred, 'no2_rf_pred', 0),
                'so2': getattr(pred, 'so2_rf_pred', 0),
                'o3': getattr(pred, 'o3_rf_pred', 0),
                'co': getattr(pred, 'co_rf_pred', 0),
                'accuracy': getattr(pred, 'rf_accuracy', 0)
            },
            'overall_aqi': pred.overall_aqi_pred or 0,
            'main_pollutant': pred.main_pollutant_pred or '',
            'aqi_level': pred.aqi_level_pred or ''
        })
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'data': data,
        'count': len(data)
    })


# 系统日志和监控视图
@permission_required('system_settings')
def system_logs_view(request):
    """
    系统日志页面
    """
    # 获取查询参数
    level = request.GET.get('level')
    action = request.GET.get('action')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # 构建查询
    query = SystemLog.objects.all()
    
    # 应用过滤条件
    if level:
        query = query.filter(level=level)
    if action:
        query = query.filter(action=action)
    if start_date:
        try:
            query = query.filter(created_at__date__gte=start_date)
        except:
            pass
    if end_date:
        try:
            query = query.filter(created_at__date__lte=end_date)
        except:
            pass
    
    # 按创建时间倒序排序
    logs = query.order_by('-created_at')[:1000]  # 限制数量
    
    return render(request, 'pollution_app/system_logs.html', {
        'page_title': '系统日志',
        'logs': logs,
        'level': level,
        'action': action,
        'start_date': start_date,
        'end_date': end_date
    })


@permission_required('system_settings')
def system_monitoring_view(request):
    """
    系统监控页面
    """
    # 记录当前系统状态
    from .models import record_system_status
    record_system_status()
    
    # 获取最近的系统状态
    statuses = SystemStatus.objects.order_by('-created_at')[:24]  # 最近24条记录
    
    # 构建图表数据
    chart_data = {
        'labels': [s.created_at.strftime('%H:%M') for s in reversed(statuses)],
        'cpu': [s.cpu_usage for s in reversed(statuses)],
        'memory': [s.memory_usage for s in reversed(statuses)],
        'disk': [s.disk_usage for s in reversed(statuses)],
        'response_time': [s.response_time for s in reversed(statuses)]
    }
    
    return render(request, 'pollution_app/system_monitoring.html', {
        'page_title': '系统监控',
        'statuses': statuses,
        'chart_data': json.dumps(chart_data)
    })


# API: 获取系统日志
@jwt_required
def api_system_logs(request):
    """
    获取系统日志
    """
    # 获取查询参数
    level = request.GET.get('level')
    action = request.GET.get('action')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    limit = request.GET.get('limit', 100)
    
    # 构建查询
    query = SystemLog.objects.all()
    
    # 应用过滤条件
    if level:
        query = query.filter(level=level)
    if action:
        query = query.filter(action=action)
    if start_date:
        try:
            query = query.filter(created_at__date__gte=start_date)
        except:
            pass
    if end_date:
        try:
            query = query.filter(created_at__date__lte=end_date)
        except:
            pass
    
    # 限制数量并排序
    try:
        limit = int(limit)
    except:
        limit = 100
    
    logs = query.order_by('-created_at')[:limit]
    
    # 构建响应数据
    data = []
    for log in logs:
        data.append({
            'id': log.id,
            'level': log.level,
            'level_display': log.get_level_display(),
            'action': log.action,
            'action_display': log.get_action_display(),
            'message': log.message,
            'user': log.user.username if log.user else None,
            'ip_address': log.ip_address,
            'created_at': log.created_at.isoformat()
        })
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'data': data,
        'count': len(data)
    })


# API: 获取系统状态
@jwt_required
def api_system_status(request):
    """
    获取系统状态
    """
    # 记录当前系统状态
    from .models import record_system_status
    record_system_status()
    
    # 获取最近的系统状态
    limit = request.GET.get('limit', 24)
    try:
        limit = int(limit)
    except:
        limit = 24
    
    statuses = SystemStatus.objects.order_by('-created_at')[:limit]
    
    # 构建响应数据
    data = []
    for status in statuses:
        data.append({
            'id': status.id,
            'cpu_usage': status.cpu_usage,
            'memory_usage': status.memory_usage,
            'disk_usage': status.disk_usage,
            'network_in': status.network_in,
            'network_out': status.network_out,
            'active_connections': status.active_connections,
            'response_time': status.response_time,
            'created_at': status.created_at.isoformat()
        })
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'data': data,
        'count': len(data)
    })


# API: 获取最新污染数据
def api_latest_data(request):
    """
    获取最新的污染数据
    支持通过AJAX获取实时数据，用于前端实时更新
    """
    # 获取查询参数
    limit = request.GET.get('limit', 10)
    try:
        limit = int(limit)
    except:
        limit = 10
    
    # 获取最近的污染数据
    latest_data = PollutionData.objects.order_by('-date')[:limit]
    
    # 构建响应数据
    data = []
    for item in latest_data:
        data.append({
            'id': item.id,
            'date': item.date.strftime('%Y-%m-%d %H:%M'),
            'pm25': item.pm25,
            'pm10': item.pm10,
            'no2': item.no2,
            'so2': item.so2,
            'o3': item.o3,
            'co': item.co,
            'temperature': item.temperature,
            'humidity': item.humidity,
            'wind_speed': item.wind_speed,
            'wind_direction': item.wind_direction,
            'pressure': item.pressure,
            'precipitation': item.precipitation
        })
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'data': data,
        'count': len(data),
        'timestamp': datetime.now().isoformat()
    })


# API: 获取用户角色和权限
@jwt_required
def api_user_permissions(request):
    """
    获取当前用户的角色和权限
    """
    from .models import get_user_roles, get_user_permissions
    
    roles = get_user_roles(request.user)
    permissions = get_user_permissions(request.user)
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'user': {
            'id': request.user.id,
            'username': request.user.username,
            'email': request.user.email
        },
        'roles': [{'id': role.id, 'name': role.name} for role in roles],
        'permissions': [{'id': perm.id, 'name': perm.name, 'codename': perm.codename} for perm in permissions]
    })


# 数据版本控制视图
@permission_required('data_edit')
def data_version_management(request):
    """
    数据版本管理页面
    """
    from .models import DataVersion
    versions = DataVersion.objects.all().order_by('-created_at')
    return render(request, 'pollution_app/data_version_management.html', {
        'page_title': '数据版本管理',
        'versions': versions
    })


@permission_required('data_edit')
def create_data_version_view(request):
    """
    创建数据版本页面
    """
    if request.method == 'POST':
        version_name = request.POST.get('version_name')
        description = request.POST.get('description')
        data_ids = request.POST.getlist('data_ids')
        
        if not version_name:
            messages.error(request, '版本名称不能为空')
            return redirect('create_data_version_view')
        
        if not data_ids:
            messages.error(request, '请选择要包含的数据')
            return redirect('create_data_version_view')
        
        # 创建数据版本
        from .models import create_data_version
        version = create_data_version(version_name, description, data_ids, request.user)
        
        messages.success(request, f'数据版本 "{version.version_name}" 创建成功')
        return redirect('data_version_management')
    else:
        # 获取可选数据
        data = PollutionData.objects.filter(is_hidden=False).order_by('-date')[:100]
        return render(request, 'pollution_app/create_data_version.html', {
            'page_title': '创建数据版本',
            'data': data
        })


@permission_required('data_edit')
def view_version_data(request, version_id):
    """
    查看指定版本的数据
    """
    from .models import DataVersion, get_version_data
    try:
        version = DataVersion.objects.get(id=version_id)
        data = get_version_data(version_id)
        return render(request, 'pollution_app/view_version_data.html', {
            'page_title': f'版本数据 - {version.version_name}',
            'version': version,
            'data': data
        })
    except DataVersion.DoesNotExist:
        messages.error(request, '数据版本不存在')
        return redirect('data_version_management')


@permission_required('data_edit')
def activate_data_version(request, version_id):
    """
    激活指定的数据版本
    """
    from .models import DataVersion
    try:
        version = DataVersion.objects.get(id=version_id)
        version.activate()
        messages.success(request, f'数据版本 "{version.version_name}" 已激活')
    except DataVersion.DoesNotExist:
        messages.error(request, '数据版本不存在')
    return redirect('data_version_management')


@permission_required('data_edit')
def delete_data_version(request, version_id):
    """
    删除数据版本
    """
    from .models import DataVersion
    try:
        version = DataVersion.objects.get(id=version_id)
        version_name = version.version_name
        version.delete()
        messages.success(request, f'数据版本 "{version_name}" 已删除')
    except DataVersion.DoesNotExist:
        messages.error(request, '数据版本不存在')
    return redirect('data_version_management')


# API: 获取数据版本列表
@jwt_required
def api_data_versions(request):
    """
    获取数据版本列表
    """
    from .models import DataVersion
    versions = DataVersion.objects.all().order_by('-created_at')
    
    version_list = []
    for version in versions:
        version_list.append({
            'id': version.id,
            'version_name': version.version_name,
            'description': version.description,
            'data_count': version.data_count,
            'upload_user': version.upload_user.username if version.upload_user else None,
            'created_at': version.created_at.isoformat(),
            'is_active': version.is_active
        })
    
    return JsonResponse({
        'code': 200,
        'message': '获取成功',
        'data': version_list
    })


# API: 获取指定版本的数据
@jwt_required
def api_version_data(request, version_id):
    """
    获取指定版本的数据
    """
    from .models import DataVersion, get_version_data
    try:
        version = DataVersion.objects.get(id=version_id)
        data = get_version_data(version_id)
        
        data_list = []
        for item in data:
            data_list.append({
                'id': item.id,
                'date': item.date.isoformat(),
                'pm25': item.pm25,
                'pm10': item.pm10,
                'no2': item.no2,
                'so2': item.so2,
                'o3': item.o3,
                'co': item.co,
                'temperature': item.temperature,
                'humidity': item.humidity,
                'wind_speed': item.wind_speed
            })
        
        return JsonResponse({
            'code': 200,
            'message': '获取成功',
            'version': {
                'id': version.id,
                'version_name': version.version_name,
                'description': version.description,
                'data_count': version.data_count
            },
            'data': data_list
        })
    except DataVersion.DoesNotExist:
        return JsonResponse({'code': 404, 'message': '数据版本不存在'}, status=404)