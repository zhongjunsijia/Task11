import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import joblib
from django.conf import settings
from django.utils import timezone
from ..models import ModelEvaluation


class ModelManager:
    """
    模型管理类
    支持多种模型类型、超参数自动调优、模型评估和管理
    """
    
    def __init__(self):
        """
        初始化模型管理器
        """
        self.model_dir = os.path.join(settings.BASE_DIR, 'pollution_app', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler_dir = os.path.join(self.model_dir, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)
        self.model_configs = {
            'linear': {
                'class': LinearRegression,
                'params': {}
            },
            'ridge': {
                'class': Ridge,
                'params': {
                    'alpha': uniform(0.1, 10.0)
                }
            },
            'lasso': {
                'class': Lasso,
                'params': {
                    'alpha': uniform(0.001, 1.0)
                }
            },
            'elasticnet': {
                'class': ElasticNet,
                'params': {
                    'alpha': uniform(0.001, 1.0),
                    'l1_ratio': uniform(0.0, 1.0)
                }
            },
            'rf': {
                'class': RandomForestRegressor,
                'params': {
                    'n_estimators': randint(50, 200),
                    'max_depth': randint(3, 15),
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5)
                }
            },
            'gb': {
                'class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': randint(50, 200),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 10)
                }
            },
            'svr': {
                'class': SVR,
                'params': {
                    'C': uniform(0.1, 10.0),
                    'gamma': uniform(0.001, 0.1),
                    'kernel': ['rbf', 'linear']
                }
            },
            'mlp': {
                'class': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                    'activation': ['relu', 'tanh'],
                    'learning_rate_init': uniform(0.001, 0.01),
                    'max_iter': [500, 1000]
                }
            }
        }
    
    def train_model(self, X, y, model_type='rf', optimize=True, n_iter=10, cv=5, model_name='default', pollutant='pm25', city='unknown', version=None, activate=False):
        """
        训练模型
        :param X: 特征数据
        :param y: 目标数据
        :param model_type: 模型类型
        :param optimize: 是否进行超参数优化
        :param n_iter: 随机搜索迭代次数
        :param cv: 交叉验证折数
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :param city: 城市名称
        :param version: 模型版本
        :param activate: 是否激活为当前版本
        :return: 训练好的模型和评估指标
        """
        start_time = time.time()
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 选择模型
        if model_type not in self.model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_configs[model_type]['class']
        params = self.model_configs[model_type]['params']
        
        if optimize and params:
            # 使用随机搜索进行超参数优化
            model = RandomizedSearchCV(
                model_class(),
                param_distributions=params,
                n_iter=n_iter,
                cv=KFold(n_splits=cv, shuffle=True, random_state=42),
                scoring='r2',
                random_state=42,
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_
            print(f"Best parameters for {model_type}: {best_params}")
        else:
            # 使用默认参数
            best_model = model_class()
            best_model.fit(X_train_scaled, y_train)
            best_params = {}
        
        # 评估模型
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        training_time = time.time() - start_time
        
        # 计算指标
        train_metrics = {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        }
        
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'training_time': training_time,
            'best_params': best_params
        }
        
        # 生成版本号
        if version is None:
            # 获取当前最大版本号并递增
            latest_version = ModelEvaluation.objects.filter(
                model_name=model_name,
                pollutant=pollutant
            ).order_by('-version').first()
            
            if latest_version:
                # 解析版本号并递增
                try:
                    version_num = float(latest_version.version.replace('v', ''))
                    new_version = f"v{version_num + 0.1:.1f}"
                except:
                    new_version = "v1.0"
            else:
                new_version = "v1.0"
        else:
            new_version = version
        
        # 保存模型和标准化器
        model_filename = f"{pollutant}_{model_type}_{model_name}_{new_version}.pkl"
        scaler_filename = f"{pollutant}_{model_type}_{model_name}_{new_version}_scaler.pkl"
        
        joblib.dump(best_model, os.path.join(self.model_dir, model_filename))
        joblib.dump(scaler, os.path.join(self.scaler_dir, scaler_filename))
        
        # 保存到数据库
        model_evaluation = ModelEvaluation.objects.create(
            model_name=model_name,
            version=new_version,
            model_type=model_type,
            city=city,
            pollutant=pollutant,
            hyperparameters=best_params,
            train_mae=train_metrics['mae'],
            train_rmse=train_metrics['rmse'],
            train_r2=train_metrics['r2'],
            test_mae=test_metrics['mae'],
            test_rmse=test_metrics['rmse'],
            test_r2=test_metrics['r2'],
            training_time=training_time,
            is_active=activate
        )
        
        # 如果需要激活，调用activate方法
        if activate:
            model_evaluation.activate()
        
        print(f"Model trained and saved: {model_filename}")
        print(f"Version: {new_version}")
        print(f"Train metrics: {train_metrics}")
        print(f"Test metrics: {test_metrics}")
        
        return best_model, scaler, metrics, new_version
    
    def train_all_pollutants(self, data, features, pollutants=None, model_types=None, model_name='default', city='unknown', activate=True):
        """
        训练所有污染物的模型
        :param data: 数据
        :param features: 特征列名
        :param pollutants: 污染物列名
        :param model_types: 模型类型列表
        :param model_name: 模型名称
        :param city: 城市名称
        :param activate: 是否激活为当前版本
        :return: 训练结果
        """
        if pollutants is None:
            pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
        
        if model_types is None:
            model_types = ['linear', 'rf', 'mlp']
        
        X = data[features]
        results = {}
        
        for model_type in model_types:
            results[model_type] = {}
            for pollutant in pollutants:
                print(f"Training {model_type} model for {pollutant}...")
                y = data[pollutant]
                # 移除缺失值
                mask = ~np.isnan(y)
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(y_clean) == 0:
                    print(f"No valid data for {pollutant}, skipping...")
                    continue
                
                model, scaler, metrics, version = self.train_model(
                    X_clean, y_clean, 
                    model_type=model_type, 
                    model_name=model_name, 
                    pollutant=pollutant, 
                    city=city, 
                    activate=activate
                )
                
                # 保存模型和标准化器（已在train_model中处理）
                model_filename = f"{pollutant}_{model_type}_{model_name}_{version}.pkl"
                scaler_filename = f"{pollutant}_{model_type}_{model_name}_{version}_scaler.pkl"
                
                results[model_type][pollutant] = {
                    'model': model_filename,
                    'scaler': scaler_filename,
                    'metrics': metrics,
                    'version': version
                }
                
                print(f"{model_type} model for {pollutant} trained successfully!")
                print(f"Version: {version}")
                print()
        
        return results
    
    def load_model(self, model_name):
        """
        加载模型
        :param model_name: 模型名称
        :return: 模型
        """
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        return joblib.load(model_path)
    
    def load_scaler(self, scaler_name):
        """
        加载标准化器
        :param scaler_name: 标准化器名称
        :return: 标准化器
        """
        scaler_path = os.path.join(self.scaler_dir, scaler_name)
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        return joblib.load(scaler_path)
    
    def evaluate_model(self, model, scaler, X, y):
        """
        评估模型
        :param model: 模型
        :param scaler: 标准化器
        :param X: 特征数据
        :param y: 目标数据
        :return: 评估指标
        """
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def compare_models(self, X, y, model_types=None):
        """
        比较不同模型的性能
        :param X: 特征数据
        :param y: 目标数据
        :param model_types: 模型类型列表
        :return: 模型比较结果
        """
        if model_types is None:
            model_types = ['linear', 'ridge', 'lasso', 'rf', 'gb', 'svr', 'mlp']
        
        results = {}
        
        for model_type in model_types:
            try:
                print(f"Evaluating {model_type}...")
                model, scaler, metrics = self.train_model(X, y, model_type=model_type)
                results[model_type] = metrics
                print(f"{model_type}: {metrics}")
            except Exception as e:
                print(f"Error evaluating {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def get_best_model(self, comparison_results, metric='r2'):
        """
        获取最佳模型
        :param comparison_results: 模型比较结果
        :param metric: 评估指标
        :return: 最佳模型类型和指标
        """
        best_model = None
        best_score = -float('inf')
        
        for model_type, metrics in comparison_results.items():
            if 'error' in metrics:
                continue
            
            score = metrics.get(metric, -float('inf'))
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model, best_score
    
    def save_model_info(self, model_info, filename):
        """
        保存模型信息
        :param model_info: 模型信息
        :param filename: 文件名
        """
        info_path = os.path.join(self.model_dir, filename)
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    def load_model_info(self, filename):
        """
        加载模型信息
        :param filename: 文件名
        :return: 模型信息
        """
        info_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(info_path):
            return None
        
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_model_by_version(self, model_name, pollutant, version=None):
        """
        加载特定版本的模型
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :param version: 模型版本，None表示加载当前激活的版本
        :return: 模型和标准化器
        """
        if version is None:
            # 加载当前激活的版本
            active_model = ModelEvaluation.objects.filter(
                model_name=model_name,
                pollutant=pollutant,
                is_active=True
            ).first()
            
            if not active_model:
                # 如果没有激活的版本，加载最新版本
                active_model = ModelEvaluation.objects.filter(
                    model_name=model_name,
                    pollutant=pollutant
                ).order_by('-created_at').first()
            
            if not active_model:
                raise FileNotFoundError(f"No model found for {model_name}-{pollutant}")
            
            version = active_model.version
            model_type = active_model.model_type
        else:
            # 加载指定版本
            model_info = ModelEvaluation.objects.filter(
                model_name=model_name,
                pollutant=pollutant,
                version=version
            ).first()
            
            if not model_info:
                raise FileNotFoundError(f"Model version {version} not found for {model_name}-{pollutant}")
            
            model_type = model_info.model_type
        
        # 构建文件名
        model_filename = f"{pollutant}_{model_type}_{model_name}_{version}.pkl"
        scaler_filename = f"{pollutant}_{model_type}_{model_name}_{version}_scaler.pkl"
        
        # 加载模型和标准化器
        model = self.load_model(model_filename)
        scaler = self.load_scaler(scaler_filename)
        
        return model, scaler
    
    def get_model_versions(self, model_name, pollutant):
        """
        获取模型的所有版本
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :return: 模型版本列表
        """
        versions = ModelEvaluation.objects.filter(
            model_name=model_name,
            pollutant=pollutant
        ).order_by('-created_at').values(
            'version', 'model_type', 'is_active', 'test_r2', 'created_at'
        )
        
        return list(versions)
    
    def get_active_model(self, model_name, pollutant):
        """
        获取当前激活的模型
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :return: 激活的模型信息
        """
        active_model = ModelEvaluation.objects.filter(
            model_name=model_name,
            pollutant=pollutant,
            is_active=True
        ).first()
        
        if not active_model:
            # 如果没有激活的版本，返回最新版本
            active_model = ModelEvaluation.objects.filter(
                model_name=model_name,
                pollutant=pollutant
            ).order_by('-created_at').first()
        
        return active_model
    
    def activate_model_version(self, model_name, pollutant, version):
        """
        激活特定版本的模型
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :param version: 模型版本
        :return: 激活的模型实例
        """
        model_evaluation = ModelEvaluation.objects.filter(
            model_name=model_name,
            pollutant=pollutant,
            version=version
        ).first()
        
        if not model_evaluation:
            raise FileNotFoundError(f"Model version {version} not found for {model_name}-{pollutant}")
        
        # 激活模型
        model_evaluation.activate()
        
        return model_evaluation
    
    def compare_model_versions(self, model_name, pollutant):
        """
        比较模型的不同版本
        :param model_name: 模型名称
        :param pollutant: 污染物类型
        :return: 版本比较结果
        """
        versions = ModelEvaluation.objects.filter(
            model_name=model_name,
            pollutant=pollutant
        ).order_by('-created_at')
        
        comparison = []
        for version in versions:
            comparison.append({
                'version': version.version,
                'model_type': version.model_type,
                'test_r2': version.test_r2,
                'test_rmse': version.test_rmse,
                'test_mae': version.test_mae,
                'training_time': version.training_time,
                'is_active': version.is_active,
                'created_at': version.created_at
            })
        
        return comparison


# 创建全局模型管理器实例
model_manager = ModelManager()