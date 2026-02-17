#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理流水线

将原始数据转化为模型可输入的 "特征矩阵 X + 目标变量 y"，确保数据质量符合训练要求。

步骤：
1. 数据接入：收集ERA5气象数据和空气质量监测数据
2. 数据清洗：处理缺失值、异常值和时空一致性检验
3. 时空对齐：空间对齐和时间对齐
4. 特征工程：衍生特征构建、特征选择和数据标准化
5. 数据集构建：时间窗口构建和数据集划分
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import scipy.interpolate as interpolate
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 确保数据目录存在
DATA_DIR = os.path.join('data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class DataProcessingPipeline:
    """数据处理流水线"""
    
    def __init__(self):
        """初始化数据处理流水线"""
        self.era5_data = None
        self.air_quality_data = None
        self.processed_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
    
    def load_era5_data(self, data_dir=None):
        """
        加载ERA5气象数据
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            data_dir = os.path.join(RAW_DIR, 'era5')
        
        print(f"正在加载ERA5气象数据 from {data_dir}...")
        
        # 模拟数据加载，实际项目中需要从ERA5下载数据
        # 这里生成模拟数据用于测试
        # 缩小时间范围和网格分辨率以减少内存使用
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        hours = pd.date_range(start_date, end_date, freq='H')
        
        # 生成网格数据
        # 扩展经度范围以包含广西站点（南宁：108.3, 柳州：109.4）
        lons = np.arange(105, 125, 0.6)  # 105 to 125 degrees east
        lats = np.arange(20, 40, 0.6)  # 20 to 40 degrees north
        
        # 生成变量数据
        data_vars = {
            't2m': (['time', 'lat', 'lon'], np.random.normal(20, 5, (len(hours), len(lats), len(lons)))),
            'u10': (['time', 'lat', 'lon'], np.random.normal(0, 3, (len(hours), len(lats), len(lons)))),
            'v10': (['time', 'lat', 'lon'], np.random.normal(0, 3, (len(hours), len(lats), len(lons)))),
            'sp': (['time', 'lat', 'lon'], np.random.normal(1013, 10, (len(hours), len(lats), len(lons)))),
            'd2m': (['time', 'lat', 'lon'], np.random.normal(15, 5, (len(hours), len(lats), len(lons))))
        }
        
        # 添加分层气压层数据
        levels = [1000, 850, 700, 500]
        for level in levels:
            data_vars[f't_{level}'] = (['time', 'lat', 'lon'], 
                                      np.random.normal(15 - (1000 - level)/100, 8, 
                                                      (len(hours), len(lats), len(lons))))
            data_vars[f'r_{level}'] = (['time', 'lat', 'lon'], 
                                      np.random.normal(50, 30, 
                                                      (len(hours), len(lats), len(lons))))
        
        # 创建xarray数据集
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': hours,
                'lat': lats,
                'lon': lons
            }
        )
        
        self.era5_data = ds
        print(f"ERA5数据加载完成，形状: {ds.dims}")
        return ds
    
    def load_air_quality_data(self, data_dir=None):
        """
        加载空气质量监测数据
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            data_dir = os.path.join(RAW_DIR, 'air_quality')
        
        print(f"正在加载空气质量监测数据 from {data_dir}...")
        
        # 模拟数据加载，实际项目中需要从中国环境监测总站获取数据
        # 这里生成模拟数据用于测试
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 12, 31)
        hours = pd.date_range(start_date, end_date, freq='H')
        
        # 站点信息（京津冀+广西）
        stations = [
            {'station_id': 'BJ001', 'name': '北京城区', 'lon': 116.4, 'lat': 39.9},
            {'station_id': 'TJ001', 'name': '天津城区', 'lon': 117.2, 'lat': 39.1},
            {'station_id': 'HB001', 'name': '石家庄', 'lon': 114.5, 'lat': 38.0},
            {'station_id': 'GX001', 'name': '南宁', 'lon': 108.3, 'lat': 22.8},
            {'station_id': 'GX002', 'name': '柳州', 'lon': 109.4, 'lat': 24.3}
        ]
        
        # 生成数据
        data = []
        for station in stations:
            for hour in hours:
                # 生成污染物数据
                pm25 = max(0, np.random.normal(50, 30))
                pm10 = max(0, np.random.normal(80, 40))
                no2 = max(0, np.random.normal(40, 20))
                so2 = max(0, np.random.normal(15, 10))
                o3 = max(0, np.random.normal(60, 30))
                co = max(0, np.random.normal(1, 0.5))
                
                # 随机添加一些缺失值
                if np.random.random() < 0.05:  # 5%的缺失率
                    pm25 = np.nan
                if np.random.random() < 0.05:
                    pm10 = np.nan
                
                data.append({
                    'station_id': station['station_id'],
                    'station_name': station['name'],
                    'lon': station['lon'],
                    'lat': station['lat'],
                    'time': hour,
                    'pm25': pm25,
                    'pm10': pm10,
                    'no2': no2,
                    'so2': so2,
                    'o3': o3,
                    'co': co
                })
        
        df = pd.DataFrame(data)
        self.air_quality_data = df
        print(f"空气质量数据加载完成，形状: {df.shape}")
        print(f"站点数量: {df['station_id'].nunique()}")
        return df
    
    def clean_data(self):
        """
        数据清洗
        - 缺失值处理
        - 异常值处理
        - 时空一致性检验
        """
        print("\n正在进行数据清洗...")
        
        # 复制数据
        df = self.air_quality_data.copy()
        
        # 1. 缺失值处理
        print("处理缺失值...")
        
        # 按站点和污染物分组处理
        cleaned_data = []
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id].copy()
            station_data = station_data.sort_values('time')
            
            for pollutant in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
                # 单小时缺失用线性插值
                station_data[pollutant] = station_data[pollutant].interpolate(method='linear')
                
                # 检查连续缺失
                # 这里简化处理，实际项目中需要更复杂的逻辑
                
        cleaned_data = pd.concat(cleaned_data) if cleaned_data else df
        
        # 2. 异常值处理
        print("处理异常值...")
        
        # 剔除物理不合理值
        cleaned_data['pm25'] = cleaned_data['pm25'].apply(lambda x: np.nan if x > 1000 else x)
        cleaned_data['pm10'] = cleaned_data['pm10'].apply(lambda x: np.nan if x > 1500 else x)
        cleaned_data['no2'] = cleaned_data['no2'].apply(lambda x: np.nan if x > 1000 else x)
        cleaned_data['so2'] = cleaned_data['so2'].apply(lambda x: np.nan if x > 500 else x)
        cleaned_data['o3'] = cleaned_data['o3'].apply(lambda x: np.nan if x > 800 else x)
        cleaned_data['co'] = cleaned_data['co'].apply(lambda x: np.nan if x > 50 else x)
        
        # 再次插值处理异常值导致的缺失
        for station_id in cleaned_data['station_id'].unique():
            mask = cleaned_data['station_id'] == station_id
            for pollutant in ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']:
                cleaned_data.loc[mask, pollutant] = cleaned_data.loc[mask, pollutant].interpolate(method='linear')
        
        # 3. 时空一致性检验
        print("时空一致性检验...")
        
        # 简化处理，实际项目中需要更复杂的检验
        # 剔除长时间断档的站点
        station_counts = cleaned_data.groupby('station_id').size()
        valid_stations = station_counts[station_counts > len(pd.date_range('2022-01-01', '2022-12-31', freq='H')) * 0.8].index
        cleaned_data = cleaned_data[cleaned_data['station_id'].isin(valid_stations)]
        
        self.air_quality_data = cleaned_data
        print(f"数据清洗完成，剩余数据形状: {cleaned_data.shape}")
        return cleaned_data
    
    def spatial_temporal_alignment(self):
        """
        时空对齐
        - 空间对齐：双线性插值将ERA5网格数据匹配到监测站点
        - 时间对齐：确保时间戳一致
        """
        print("\n正在进行时空对齐...")
        
        # 1. 空间对齐
        print("空间对齐：双线性插值...")
        
        aligned_data = []
        for station_id in self.air_quality_data['station_id'].unique():
            station_data = self.air_quality_data[self.air_quality_data['station_id'] == station_id].copy()
            station_lon = station_data['lon'].iloc[0]
            station_lat = station_data['lat'].iloc[0]
            
            # 双线性插值获取ERA5数据
            for _, row in station_data.iterrows():
                time = row['time']
                
                # 提取对应时间的ERA5数据
                era5_time_data = self.era5_data.sel(time=time, method='nearest')
                
                # 双线性插值
                era5_values = {}
                
                # 获取ERA5网格范围
                lon = era5_time_data['lon'].values
                lat = era5_time_data['lat'].values
                
                # 确保站点坐标在网格范围内
                station_lat_clamped = max(lat.min(), min(lat.max(), station_lat))
                station_lon_clamped = max(lon.min(), min(lon.max(), station_lon))
                
                # 尝试获取气象数据
                try:
                    for var in ['t2m', 'u10', 'v10', 'sp', 'd2m']:
                        # 创建插值函数
                        data = era5_time_data[var].values
                        
                        # 双线性插值
                        interp_func = interpolate.RegularGridInterpolator(
                            (lat, lon), data, method='linear'
                        )
                        value = interp_func((station_lat_clamped, station_lon_clamped))
                        # 确保值是标量
                        if isinstance(value, np.ndarray):
                            value = value.item()
                        era5_values[var] = value
                    
                    # 添加分层气压层数据
                    levels = [1000, 850, 700, 500]
                    for level in levels:
                        for var in [f't_{level}', f'r_{level}']:
                            data = era5_time_data[var].values
                            
                            interp_func = interpolate.RegularGridInterpolator(
                                (lat, lon), data, method='linear'
                            )
                            value = interp_func((station_lat_clamped, station_lon_clamped))
                            # 确保值是标量
                            if isinstance(value, np.ndarray):
                                value = value.item()
                            era5_values[var] = value
                    
                    # 合并数据
                    row_data = row.to_dict()
                    row_data.update(era5_values)
                    aligned_data.append(row_data)
                except Exception as e:
                    # 如果插值失败，跳过该数据点
                    print(f"插值失败，跳过数据点: {e}")
                    continue
        
        aligned_df = pd.DataFrame(aligned_data)
        
        # 2. 时间对齐
        print("时间对齐：确保时间戳一致...")
        # 确保时间戳为整点时刻
        aligned_df['time'] = aligned_df['time'].dt.floor('H')
        
        self.processed_data = aligned_df
        print(f"时空对齐完成，数据形状: {aligned_df.shape}")
        return aligned_df
    
    def feature_engineering(self):
        """
        特征工程
        - 衍生特征构建
        - 特征选择
        - 数据标准化
        """
        print("\n正在进行特征工程...")
        
        # 复制数据
        df = self.processed_data.copy()
        
        # 1. 衍生特征构建
        print("构建衍生特征...")
        
        # 确保数据类型正确
        df['u10'] = pd.to_numeric(df['u10'], errors='coerce')
        df['v10'] = pd.to_numeric(df['v10'], errors='coerce')
        df['d2m'] = pd.to_numeric(df['d2m'], errors='coerce')
        df['t2m'] = pd.to_numeric(df['t2m'], errors='coerce')
        df['t_1000'] = pd.to_numeric(df['t_1000'], errors='coerce')
        df['t_850'] = pd.to_numeric(df['t_850'], errors='coerce')
        
        # 处理缺失值
        df = df.fillna(df.mean())
        
        # 计算风速和风向
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
        df['wind_direction'] = np.arctan2(df['v10'], df['u10']) * 180 / np.pi
        
        # 计算相对湿度
        # 简化计算，实际项目中需要更精确的公式
        df['relative_humidity'] = 100 * np.exp((17.625 * (df['d2m'] - 273.15)) / (243.04 + (df['d2m'] - 273.15))) / \
                                  np.exp((17.625 * (df['t2m'] - 273.15)) / (243.04 + (df['t2m'] - 273.15)))
        
        # 计算大气稳定度（简化版）
        df['stability'] = df['t_1000'] - df['t_850']
        
        # 2. 特征选择
        print("特征选择...")
        
        # 选择特征和目标变量
        features = ['t2m', 'u10', 'v10', 'sp', 'd2m', 't_1000', 'r_1000', 't_850', 'r_850', 
                   't_700', 'r_700', 't_500', 'r_500', 'wind_speed', 'wind_direction', 
                   'relative_humidity', 'stability']
        target = 'pm25'
        
        # 移除目标变量为NaN的行
        df = df.dropna(subset=[target])
        
        # Pearson相关性分析
        corr_matrix = df[features + [target]].corr()
        corr_with_target = corr_matrix[target].abs()
        selected_features = [f for f in features if corr_with_target[f] > 0.3]
        
        print(f"相关性分析后选择的特征: {selected_features}")
        
        # 随机森林特征重要性
        if len(selected_features) > 0:
            X = df[selected_features]
            y = df[target]
            
            # 处理特征中的NaN
            X = X.fillna(X.mean())
            
            # 训练随机森林
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # 获取特征重要性
            importances = rf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # 选取累计贡献率达80%的Top-K特征
            cumulative_importance = np.cumsum(feature_importance['importance'])
            top_k = np.argmax(cumulative_importance >= 0.8) + 1
            final_features = feature_importance['feature'].iloc[:top_k].tolist()
            
            print(f"随机森林特征重要性分析后选择的特征: {final_features}")
        else:
            final_features = selected_features
        
        # 3. 数据标准化
        print("数据标准化...")
        
        if len(final_features) > 0:
            self.scaler = StandardScaler()
            df[final_features] = self.scaler.fit_transform(df[final_features])
        
        self.processed_data = df
        self.features = final_features
        self.target = target
        print(f"特征工程完成，最终特征数量: {len(final_features)}")
        return df
    
    def build_dataset(self, window_length=24, forecast_horizon=1):
        """
        数据集构建
        - 时间窗口构建
        - 数据集划分
        
        Args:
            window_length: 输入窗口长度（小时）
            forecast_horizon: 预测步长（小时）
        """
        print("\n正在构建数据集...")
        
        # 复制数据
        df = self.processed_data.copy()
        
        # 1. 时间窗口构建
        print(f"构建时间窗口，窗口长度: {window_length}小时，预测步长: {forecast_horizon}小时...")
        
        X = []
        y = []
        times = []
        
        # 按站点分组处理
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id].sort_values('time')
            
            for i in range(len(station_data) - window_length - forecast_horizon + 1):
                # 输入窗口
                window_data = station_data.iloc[i:i+window_length]
                # 目标值
                target_data = station_data.iloc[i+window_length+forecast_horizon-1]
                
                # 提取特征和目标
                if not window_data[self.features].isnull().any().any() and not pd.isnull(target_data[self.target]):
                    X.append(window_data[self.features].values.flatten())
                    y.append(target_data[self.target])
                    times.append(target_data['time'])
        
        X = np.array(X)
        y = np.array(y)
        times = np.array(times)
        
        print(f"时间窗口构建完成，样本数量: {len(X)}")
        
        # 2. 数据集划分
        print("划分数据集...")
        
        # 按时间顺序划分
        # 由于使用2022年数据，调整划分比例
        # 训练集：70%，2022.01-2022.08
        # 验证集：15%，2022.09-2022.10
        # 测试集：15%，2022.11-2022.12
        
        # 创建时间掩码
        train_mask = (times >= np.datetime64('2022-01-01')) & (times <= np.datetime64('2022-08-31'))
        val_mask = (times >= np.datetime64('2022-09-01')) & (times <= np.datetime64('2022-10-31'))
        test_mask = (times >= np.datetime64('2022-11-01')) & (times <= np.datetime64('2022-12-31'))
        
        # 划分数据
        self.X_train = X[train_mask]
        self.y_train = y[train_mask]
        self.X_val = X[val_mask]
        self.y_val = y[val_mask]
        self.X_test = X[test_mask]
        self.y_test = y[test_mask]
        
        print(f"数据集划分完成:")
        print(f"训练集: {len(self.X_train)} samples")
        print(f"验证集: {len(self.X_val)} samples")
        print(f"测试集: {len(self.X_test)} samples")
        
        # 保存数据集
        np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), self.X_train)
        np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), self.y_train)
        np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), self.X_val)
        np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), self.y_val)
        np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), self.X_test)
        np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), self.y_test)
        
        print(f"数据集已保存到: {PROCESSED_DIR}")
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test
        }
    
    def run_pipeline(self, window_length=24, forecast_horizon=1):
        """
        运行完整的数据处理流水线
        
        Args:
            window_length: 输入窗口长度（小时）
            forecast_horizon: 预测步长（小时）
        """
        print("=" * 80)
        print("数据处理流水线")
        print("=" * 80)
        
        # 1. 数据接入
        self.load_era5_data()
        self.load_air_quality_data()
        
        # 2. 数据清洗
        self.clean_data()
        
        # 3. 时空对齐
        self.spatial_temporal_alignment()
        
        # 4. 特征工程
        self.feature_engineering()
        
        # 5. 数据集构建
        dataset = self.build_dataset(window_length, forecast_horizon)
        
        print("\n" + "=" * 80)
        print("数据处理流水线完成！")
        print("=" * 80)
        
        return dataset

if __name__ == "__main__":
    # 创建并运行数据处理流水线
    pipeline = DataProcessingPipeline()
    
    # 测试不同的预测步长
    forecast_horizons = [1, 6, 12, 24]
    
    for horizon in forecast_horizons:
        print(f"\n" + "=" * 80)
        print(f"处理预测步长: {horizon}小时")
        print("=" * 80)
        
        dataset = pipeline.run_pipeline(window_length=24, forecast_horizon=horizon)
        
        # 保存不同预测步长的数据集
        horizon_dir = os.path.join(PROCESSED_DIR, f'horizon_{horizon}')
        os.makedirs(horizon_dir, exist_ok=True)
        
        np.save(os.path.join(horizon_dir, 'X_train.npy'), pipeline.X_train)
        np.save(os.path.join(horizon_dir, 'y_train.npy'), pipeline.y_train)
        np.save(os.path.join(horizon_dir, 'X_val.npy'), pipeline.X_val)
        np.save(os.path.join(horizon_dir, 'y_val.npy'), pipeline.y_val)
        np.save(os.path.join(horizon_dir, 'X_test.npy'), pipeline.X_test)
        np.save(os.path.join(horizon_dir, 'y_test.npy'), pipeline.y_test)
        
        print(f"预测步长 {horizon}小时 的数据集已保存到: {horizon_dir}")
    
    print("\n所有数据处理任务已完成！")
