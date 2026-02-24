from pollution_app.utils.model_manager import model_manager
from pollution_app.prediction_model import train_and_save_model
import numpy as np
import pandas as pd
from datetime import datetime

# 测试模型管理器
print("Testing model manager...")

# 生成测试数据
data = {
    'date': pd.date_range(start='2025-01-01', periods=200),
    'pm25': np.random.randint(20, 150, 200),
    'pm10': np.random.randint(30, 200, 200),
    'no2': np.random.randint(10, 100, 200),
    'so2': np.random.randint(5, 80, 200),
    'o3': np.random.randint(20, 180, 200),
    'co': np.random.uniform(0.5, 7, 200),
    'temperature': np.random.uniform(10, 35, 200),
    'humidity': np.random.uniform(30, 90, 200),
    'wind_speed': np.random.uniform(1, 8, 200)
}
df = pd.DataFrame(data)

# 特征工程
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
features = ['day_of_week', 'month', 'temperature', 'humidity', 'wind_speed']
X = df[features]
y = df['pm25']

# 测试1: 单个模型训练
print("\nTest 1: Single model training")
try:
    model, scaler, metrics = model_manager.train_model(X, y, model_type='rf', optimize=True)
    print(f"Model training completed successfully!")
    print(f"Metrics: {metrics}")
except Exception as e:
    print(f"Error in model training: {e}")

# 测试2: 模型比较
print("\nTest 2: Model comparison")
try:
    comparison_results = model_manager.compare_models(X, y, model_types=['linear', 'rf', 'mlp'])
    print(f"Model comparison completed!")
    for model_type, metrics in comparison_results.items():
        print(f"{model_type}: {metrics}")
    
    # 获取最佳模型
    best_model, best_score = model_manager.get_best_model(comparison_results)
    print(f"\nBest model: {best_model} with R² score: {best_score}")
except Exception as e:
    print(f"Error in model comparison: {e}")

# 测试3: 训练所有污染物的模型
print("\nTest 3: Train all pollutants models")
try:
    results = train_and_save_model(model_types=['linear', 'rf', 'mlp'])
    print(f"All models trained successfully!")
    print(f"Training results saved to model_training_results.json")
except Exception as e:
    print(f"Error in training all models: {e}")

print("\nAll tests completed!")
