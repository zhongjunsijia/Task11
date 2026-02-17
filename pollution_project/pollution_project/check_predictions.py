import os
import django

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pollution_project.settings')
django.setup()

from pollution_app.models import PredictionResult

# 获取所有预测数据
predictions = PredictionResult.objects.all()
print(f"Total predictions: {predictions.count()}")

# 打印前5条预测数据的详细信息
print("\nFirst 5 predictions:")
for i, pred in enumerate(predictions[:5]):
    print(f"\nPrediction {i+1}:")
    print(f"Target date: {pred.target_date}")
    print(f"PM2.5 (linear): {pred.pm25_pred}")
    print(f"PM10 (linear): {pred.pm10_pred}")
    print(f"PM2.5 (NN): {pred.pm25_nn_pred}")
    print(f"PM10 (NN): {pred.pm10_nn_pred}")
    print(f"PM2.5 (RF): {pred.pm25_rf_pred}")
    print(f"PM10 (RF): {pred.pm10_rf_pred}")

# 检查是否有预测值为0的情况
zero_predictions = predictions.filter(pm25_pred=0)
print(f"\nPredictions with PM2.5=0: {zero_predictions.count()}")

# 检查最近的预测数据
latest_pred = predictions.order_by('-target_date').first()
if latest_pred:
    print(f"\nLatest prediction:")
    print(f"Target date: {latest_pred.target_date}")
    print(f"PM2.5 (linear): {latest_pred.pm25_pred}")
    print(f"PM10 (linear): {latest_pred.pm10_pred}")
