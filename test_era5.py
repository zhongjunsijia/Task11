from pollution_app.utils.era5_data import era5_handler
from datetime import datetime, timedelta

# 测试ERA5数据获取
print("Testing ERA5 data handler...")

# 定义测试参数
variables = ['t2m', 'd2m', 'sp', 'tp', 'u10', 'v10']  # 温度、露点温度、气压、降水量、风分量
start_date = datetime.now() - timedelta(days=7)
end_date = datetime.now()
area = [45, -10, 35, 5]  # 欧洲部分区域
resolution = 0.5

# 测试获取数据
print(f"\nFetching ERA5 data from {start_date} to {end_date}")
print(f"Variables: {variables}")
print(f"Area: {area}")
print(f"Resolution: {resolution}")

try:
    # 获取数据
    data = era5_handler.get_era5_data(variables, start_date, end_date, area, resolution)
    print(f"\nData fetched successfully!")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    # 测试预处理
    print("\nTesting data preprocessing...")
    preprocessed_data = era5_handler.preprocess_data(data)
    print(f"Preprocessed data shape: {preprocessed_data.shape}")
    print(f"Preprocessed columns: {list(preprocessed_data.columns)}")
    
    # 测试特征提取
    print("\nTesting feature extraction...")
    features = era5_handler.get_weather_features(data)
    print(f"Features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    print(f"\nFirst 5 rows of features:")
    print(features.head())
    
    print("\nAll tests passed successfully!")
    
except Exception as e:
    print(f"\nError during testing: {e}")
    import traceback
    traceback.print_exc()
