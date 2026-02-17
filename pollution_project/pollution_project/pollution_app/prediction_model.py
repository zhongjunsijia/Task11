import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
import os
from django.conf import settings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
from .utils.model_manager import model_manager


def train_and_save_model(model_types=None):
    """训练污染预测模型并保存"""
    # 模拟数据
    data = {
        'date': pd.date_range(start='2025-01-01', periods=100),
        'pm25': np.random.randint(20, 150, 100),
        'pm10': np.random.randint(30, 200, 100),
        # 污染物模拟数据
        'no2': np.random.randint(10, 100, 100),  # 二氧化氮
        'so2': np.random.randint(5, 80, 100),  # 二氧化硫
        'o3': np.random.randint(20, 180, 100),  # 臭氧
        'co': np.random.uniform(0.5, 7, 100),  # 一氧化碳
        # 原有气象数据
        'temperature': np.random.uniform(10, 35, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'wind_speed': np.random.uniform(1, 8, 100)
    }
    df = pd.DataFrame(data)

    # 特征工程：提取日期特征
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # 定义特征（所有模型共享相同特征）
    features = ['day_of_week', 'month', 'temperature', 'humidity', 'wind_speed']
    
    # 使用模型管理器训练所有污染物的模型
    if model_types is None:
        model_types = ['linear', 'rf', 'mlp']
    
    # 训练所有模型
    results = model_manager.train_all_pollutants(df, features, model_types=model_types)
    
    # 保存训练结果
    model_manager.save_model_info(results, 'model_training_results.json')
    
    return results  # 返回完整评估指标

def load_model(model_name):
    """加载已保存的模型（支持新污染物模型）"""
    model_path = os.path.join(settings.BASE_DIR, 'pollution_app', 'models', f'{model_name}_model.pkl')
    return joblib.load(model_path)


def load_backend_model():
    """加载后端训练的随机森林模型"""
    model_path = os.path.join(settings.BASE_DIR, 'pollution_app', 'models', 'rf_model_horizon1.pkl')
    return joblib.load(model_path)


def load_scaler():
    """加载后端训练的标准化器"""
    scaler_path = os.path.join(settings.BASE_DIR, 'pollution_app', 'models', 'scaler.pkl')
    return joblib.load(scaler_path)


def predict_pollution(date, temperature, humidity, wind_speed,
                      pm25, pm10, no2, so2, o3, co, model_type='linear'):  # model_type参数
    """支持多种模型类型的预测函数"""
    # 如果使用后端模型
    if model_type == 'backend_rf':
        try:
            # 加载后端模型
            model = load_backend_model()
            # 加载标准化器
            scaler = load_scaler()
            
            # 准备时间窗口数据（模拟最近3个时间点的数据）
            # 注意：这里使用当前数据重复3次作为模拟窗口，实际应用中应使用真实的历史数据
            window_data = np.array([
                [pm10, temperature, humidity],
                [pm10, temperature, humidity],
                [pm10, temperature, humidity]
            ])
            
            # 数据标准化
            window_data_scaled = scaler.transform(window_data)
            
            # 展平输入（RF不支持时序输入）
            input_data = window_data_scaled.reshape(1, -1)
            
            # 预测
            pm25_pred = model.predict(input_data)[0]
            
            # 返回结果（只预测pm2.5，其他污染物使用默认值）
            return {
                'pm25': round(pm25_pred, 2),
                'pm10': pm10,  # 使用输入值作为默认值
                'no2': no2,    # 使用输入值作为默认值
                'so2': so2,    # 使用输入值作为默认值
                'o3': o3,      # 使用输入值作为默认值
                'co': co,      # 使用输入值作为默认值
                'date': date.strftime('%Y-%m-%d')
            }
        except Exception as e:
            print(f"后端模型预测失败: {e}")
            # 失败时回退到线性模型
            model_type = 'linear'
    
    # 使用模型管理器加载模型（支持多种模型类型）
    try:
        # 特征准备（与训练时一致）
        day_of_week = date.weekday()
        month = date.month
        features = np.array([[day_of_week, month, temperature, humidity, wind_speed]])
        
        # 预测结果字典
        predictions = {'date': date.strftime('%Y-%m-%d')}
        
        # 对每个污染物进行预测
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        for pollutant in pollutants:
            try:
                # 加载模型和标准化器
                model_filename = f"{pollutant}_{model_type}_model.pkl"
                scaler_filename = f"{pollutant}_{model_type}_scaler.pkl"
                
                model = model_manager.load_model(model_filename)
                scaler = model_manager.load_scaler(scaler_filename)
                
                # 数据标准化
                features_scaled = scaler.transform(features)
                
                # 预测
                pred = model.predict(features_scaled)[0]
                predictions[pollutant] = round(pred, 2)
            except Exception as e:
                print(f"预测 {pollutant} 时出错: {e}")
                # 如果模型加载失败，使用输入值作为默认值
                predictions[pollutant] = locals()[pollutant]
        
        return predictions
    except Exception as e:
        print(f"模型预测失败: {e}")
        # 失败时使用原始方法
        # 使用前端模型（线性回归或神经网络）
        suffix = '' if model_type == 'linear' else '_nn'
        model_pm25 = load_model(f'pm25{suffix}')
        model_pm10 = load_model(f'pm10{suffix}')
        model_no2 = load_model(f'no2{suffix}')
        model_so2 = load_model(f'so2{suffix}')
        model_o3 = load_model(f'o3{suffix}')
        model_co = load_model(f'co{suffix}')

        # 特征准备（与训练时一致）
        day_of_week = date.weekday()
        month = date.month
        features = np.array([[day_of_week, month, temperature, humidity, wind_speed]])

        # 预测并返回结果
        return {
                'pm25': round(model_pm25.predict(features)[0], 2),
                'pm10': round(model_pm10.predict(features)[0], 2),
                'no2': round(model_no2.predict(features)[0], 2),
                'so2': round(model_so2.predict(features)[0], 2),
                'o3': round(model_o3.predict(features)[0], 2),
                'co': round(model_co.predict(features)[0], 2),
                'date': date.strftime('%Y-%m-%d')
            }


def batch_predict_pollution(prediction_data, model_type='linear'):
    """
    批量预测污染数据
    :param prediction_data: 预测数据列表，每个元素包含：
        {
            'date': 日期对象,
            'temperature': 温度,
            'humidity': 湿度,
            'wind_speed': 风速,
            'pm25': 当前PM2.5值,
            'pm10': 当前PM10值,
            'no2': 当前NO2值,
            'so2': 当前SO2值,
            'o3': 当前O3值,
            'co': 当前CO值
        }
    :param model_type: 模型类型
    :return: 批量预测结果列表
    """
    import time
    start_time = time.time()
    
    # 后端随机森林模型只支持PM2.5预测，且处理方式不同
    if model_type == 'backend_rf':
        try:
            # 加载后端模型和标准化器
            model = load_backend_model()
            scaler = load_scaler()
            
            results = []
            for data in prediction_data:
                date = data['date']
                pm10 = data['pm10']
                temperature = data['temperature']
                humidity = data['humidity']
                
                # 准备时间窗口数据
                window_data = np.array([
                    [pm10, temperature, humidity],
                    [pm10, temperature, humidity],
                    [pm10, temperature, humidity]
                ])
                
                # 数据标准化
                window_data_scaled = scaler.transform(window_data)
                
                # 展平输入
                input_data = window_data_scaled.reshape(1, -1)
                
                # 预测
                pm25_pred = model.predict(input_data)[0]
                
                # 构建结果
                result = {
                    'pm25': round(pm25_pred, 2),
                    'pm10': pm10,
                    'no2': data['no2'],
                    'so2': data['so2'],
                    'o3': data['o3'],
                    'co': data['co'],
                    'date': date.strftime('%Y-%m-%d')
                }
                results.append(result)
            
            print(f"批量预测完成，耗时: {time.time() - start_time:.2f}秒")
            return results
        except Exception as e:
            print(f"后端模型批量预测失败: {e}")
            # 失败时回退到线性模型
            model_type = 'linear'
    
    # 使用模型管理器进行批量预测
    try:
        # 准备特征数据
        features_list = []
        dates = []
        current_values = []
        
        for data in prediction_data:
            date = data['date']
            day_of_week = date.weekday()
            month = date.month
            features = [day_of_week, month, data['temperature'], data['humidity'], data['wind_speed']]
            features_list.append(features)
            dates.append(date)
            current_values.append({
                'pm25': data['pm25'],
                'pm10': data['pm10'],
                'no2': data['no2'],
                'so2': data['so2'],
                'o3': data['o3'],
                'co': data['co']
            })
        
        # 转换为numpy数组
        features_array = np.array(features_list)
        
        # 对每个污染物进行批量预测
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        predictions = {}
        
        for pollutant in pollutants:
            try:
                # 加载模型和标准化器
                model_filename = f"{pollutant}_{model_type}_model.pkl"
                scaler_filename = f"{pollutant}_{model_type}_scaler.pkl"
                
                model = model_manager.load_model(model_filename)
                scaler = model_manager.load_scaler(scaler_filename)
                
                # 数据标准化
                features_scaled = scaler.transform(features_array)
                
                # 批量预测
                preds = model.predict(features_scaled)
                predictions[pollutant] = [round(pred, 2) for pred in preds]
            except Exception as e:
                print(f"批量预测 {pollutant} 时出错: {e}")
                # 如果模型加载失败，使用输入值作为默认值
                predictions[pollutant] = [cv[pollutant] for cv in current_values]
        
        # 构建结果列表
        results = []
        for i, date in enumerate(dates):
            result = {
                'pm25': predictions['pm25'][i],
                'pm10': predictions['pm10'][i],
                'no2': predictions['no2'][i],
                'so2': predictions['so2'][i],
                'o3': predictions['o3'][i],
                'co': predictions['co'][i],
                'date': date.strftime('%Y-%m-%d')
            }
            results.append(result)
        
        print(f"批量预测完成，耗时: {time.time() - start_time:.2f}秒")
        return results
    except Exception as e:
        print(f"批量预测失败: {e}")
        # 失败时使用单个预测方法
        results = []
        for data in prediction_data:
            result = predict_pollution(
                data['date'],
                data['temperature'],
                data['humidity'],
                data['wind_speed'],
                data['pm25'],
                data['pm10'],
                data['no2'],
                data['so2'],
                data['o3'],
                data['co'],
                model_type=model_type
            )
            results.append(result)
        
        print(f"单个预测回退完成，耗时: {time.time() - start_time:.2f}秒")
        return results
