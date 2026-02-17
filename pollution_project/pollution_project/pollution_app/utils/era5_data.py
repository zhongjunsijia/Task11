import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from django.conf import settings


class ERA5DataHandler:
    """
    ERA5气象数据处理类
    支持从CDS API获取数据或读取本地数据文件
    """
    
    def __init__(self):
        """
        初始化ERA5数据处理器
        """
        # 数据存储目录
        self.data_dir = os.path.join(settings.BASE_DIR, 'data', 'era5')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # CDS API配置
        self.cds_config = {
            'url': 'https://cds.climate.copernicus.eu/api/v2',
            'key': getattr(settings, 'CDS_API_KEY', '')
        }
    
    def get_era5_data(self, variables, start_date, end_date, area=None, resolution=0.25):
        """
        获取ERA5气象数据
        :param variables: 气象变量列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param area: 区域范围 [north, west, south, east]
        :param resolution: 空间分辨率
        :return: 处理后的气象数据
        """
        # 尝试从本地缓存获取
        cache_key = self._generate_cache_key(variables, start_date, end_date, area, resolution)
        cache_file = os.path.join(self.data_dir, f"{cache_key}.csv")
        
        if os.path.exists(cache_file):
            print(f"Loading ERA5 data from cache: {cache_file}")
            return self._load_local_data(cache_file)
        
        # 尝试从CDS API获取
        try:
            data = self._fetch_from_cds(variables, start_date, end_date, area, resolution)
            # 保存到本地缓存
            self._save_to_cache(data, cache_file)
            return data
        except Exception as e:
            print(f"Error fetching ERA5 data: {e}")
            # 如果API获取失败，尝试加载本地示例数据
            return self._load_local_sample_data(variables, start_date, end_date)
    
    def _generate_cache_key(self, variables, start_date, end_date, area, resolution):
        """
        生成缓存键
        :param variables: 气象变量列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param area: 区域范围
        :param resolution: 空间分辨率
        :return: 缓存键字符串
        """
        var_str = '_'.join(sorted(variables))
        date_str = f"{start_date}_{end_date}"
        area_str = "global" if not area else '_'.join(map(str, area))
        return f"era5_{var_str}_{date_str}_{area_str}_{resolution}"
    
    def _fetch_from_cds(self, variables, start_date, end_date, area, resolution):
        """
        从CDS API获取数据
        :param variables: 气象变量列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param area: 区域范围
        :param resolution: 空间分辨率
        :return: 处理后的气象数据
        """
        # 检查CDS API密钥
        if not self.cds_config['key']:
            raise Exception("CDS API key not configured")
        
        # 配置CDS API
        import cdsapi
        c = cdsapi.Client(url=self.cds_config['url'], key=self.cds_config['key'])
        
        # 准备请求参数
        request_params = {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': [str(start_date.year), str(end_date.year)],
            'month': [str(m) for m in range(start_date.month, end_date.month + 1)],
            'day': [str(d) for d in range(1, 32)],
            'time': [f"{h:02d}:00" for h in range(24)],
            'format': 'netcdf'
        }
        
        # 添加区域参数
        if area:
            request_params['area'] = area
        
        # 添加分辨率参数
        request_params['grid'] = [resolution, resolution]
        
        # 下载数据
        temp_file = os.path.join(self.data_dir, f"temp_{datetime.now().timestamp()}.nc")
        c.retrieve('reanalysis-era5-single-levels', request_params, temp_file)
        
        # 处理NetCDF文件
        data = self._process_netcdf(temp_file)
        
        # 清理临时文件
        os.remove(temp_file)
        
        return data
    
    def _process_netcdf(self, file_path):
        """
        处理NetCDF文件
        :param file_path: NetCDF文件路径
        :return: 处理后的数据
        """
        import xarray as xr
        
        # 打开NetCDF文件
        ds = xr.open_dataset(file_path)
        
        # 转换为DataFrame
        df = ds.to_dataframe().reset_index()
        
        # 处理时间格式
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['date'] = df['time']
        
        # 重命名列以匹配系统要求
        column_mapping = {
            't2m': 'temperature',  # 2米温度
            'd2m': 'dewpoint',     # 2米露点温度
            'sp': 'pressure',      # 表面气压
            'tp': 'precipitation', # 总降水量
            'u10': 'u_wind',       # 10米u风分量
            'v10': 'v_wind',       # 10米v风分量
            'blh': 'boundary_layer_height',  # 边界层高度
            'ssr': 'surface_solar_radiation',  # 表面太阳辐射
            'str': 'surface_thermal_radiation'  # 表面热辐射
        }
        
        df = df.rename(columns=column_mapping)
        
        # 计算派生变量
        if 'u_wind' in df.columns and 'v_wind' in df.columns:
            df['wind_speed'] = np.sqrt(df['u_wind']**2 + df['v_wind']**2)
            df['wind_direction'] = np.degrees(np.arctan2(df['v_wind'], df['u_wind']))
            df['wind_direction'] = (df['wind_direction'] + 360) % 360
        
        # 计算湿度
        if 'temperature' in df.columns and 'dewpoint' in df.columns:
            df['humidity'] = self._calculate_humidity(df['temperature'], df['dewpoint'])
        
        return df
    
    def _calculate_humidity(self, temperature, dewpoint):
        """
        计算相对湿度
        :param temperature: 温度（K）
        :param dewpoint: 露点温度（K）
        :return: 相对湿度（%）
        """
        # 转换为摄氏度
        T_c = temperature - 273.15
        Td_c = dewpoint - 273.15
        
        # 计算饱和水汽压
        es = 6.112 * np.exp((17.67 * T_c) / (T_c + 243.5))
        e = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))
        
        # 计算相对湿度
        rh = (e / es) * 100
        rh = np.clip(rh, 0, 100)
        
        return rh
    
    def _load_local_data(self, file_path):
        """
        加载本地数据
        :param file_path: 文件路径
        :return: 加载的数据
        """
        df = pd.read_csv(file_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    
    def _load_local_sample_data(self, variables, start_date, end_date):
        """
        加载本地示例数据
        :param variables: 气象变量列表
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: 示例数据
        """
        # 生成模拟数据
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        n = len(date_range)
        
        data = {
            'date': date_range,
            'temperature': np.random.uniform(280, 300, n),  # 温度（K）
            'humidity': np.random.uniform(40, 80, n),       # 湿度（%）
            'wind_speed': np.random.uniform(1, 10, n),      # 风速（m/s）
            'wind_direction': np.random.uniform(0, 360, n),  # 风向（度）
            'pressure': np.random.uniform(990, 1020, n),     # 气压（hPa）
            'precipitation': np.random.uniform(0, 5, n)      # 降水量（mm）
        }
        
        # 添加请求的变量
        for var in variables:
            if var not in data:
                data[var] = np.random.uniform(0, 100, n)
        
        return pd.DataFrame(data)
    
    def _save_to_cache(self, data, file_path):
        """
        保存数据到缓存
        :param data: 数据
        :param file_path: 文件路径
        """
        data.to_csv(file_path, index=False)
    
    def preprocess_data(self, data):
        """
        预处理数据
        :param data: 原始数据
        :return: 预处理后的数据
        """
        # 复制数据以避免修改原始数据
        df = data.copy()
        
        # 处理缺失值
        df = df.fillna(df.mean())
        
        # 处理异常值
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        # 标准化数据
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'date':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[f'{col}_norm'] = (df[col] - mean) / std
        
        return df
    
    def get_weather_features(self, data):
        """
        提取气象特征
        :param data: 原始数据
        :return: 特征数据
        """
        # 预处理数据
        df = self.preprocess_data(data)
        
        # 提取时间特征
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['season'] = (df['month'] % 12 + 3) // 3
        
        # 提取滞后特征
        for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
            if col in df.columns:
                for lag in [1, 3, 6, 12, 24]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 提取滚动统计特征
        for col in ['temperature', 'humidity', 'wind_speed', 'pressure']:
            if col in df.columns:
                df[f'{col}_rolling_mean_3h'] = df[col].rolling(window=3).mean()
                df[f'{col}_rolling_std_3h'] = df[col].rolling(window=3).std()
                df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24).mean()
                df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24).std()
        
        # 填充滞后特征的缺失值
        df = df.fillna(df.mean())
        
        return df


# 创建全局ERA5数据处理器实例
era5_handler = ERA5DataHandler()