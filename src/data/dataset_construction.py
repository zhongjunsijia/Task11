from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

def create_dataset(input_path, output_dir, window_size=24, forecast_horizon=1):
    # 1. 读取数据
    data = pd.read_csv(input_path)

    # 2. 准备特征和目标变量
    # 选择用于预测的特征
    features = ['pm10', 'temperature', 'humidity']
    X = data[features]
    y = data['pm25']

    # 3. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 保存标准化器
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    # 5. 构建时间窗口
    def create_time_windows(features, targets, window_size, forecast_horizon):
        X_windows = []
        y_windows = []

        for i in range(len(features) - window_size - forecast_horizon + 1):
            X_window = features[i:i+window_size]
            y_window = targets[i+window_size:i+window_size+forecast_horizon]
            X_windows.append(X_window)
            y_windows.append(y_window)

        return np.array(X_windows), np.array(y_windows)

    # 6. 构建数据集
    X, y = create_time_windows(X_scaled, y.values, window_size, forecast_horizon)

    # 7. 数据集划分（按时间顺序）
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    test_size = len(X) - train_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 8. 保存数据集
    np.save(f"{output_dir}/X_train_horizon{forecast_horizon}.npy", X_train)
    np.save(f"{output_dir}/y_train_horizon{forecast_horizon}.npy", y_train)
    np.save(f"{output_dir}/X_val_horizon{forecast_horizon}.npy", X_val)
    np.save(f"{output_dir}/y_val_horizon{forecast_horizon}.npy", y_val)
    np.save(f"{output_dir}/X_test_horizon{forecast_horizon}.npy", X_test)
    np.save(f"{output_dir}/y_test_horizon{forecast_horizon}.npy", y_test)

    print(f"数据集构建完成，预测步长={forecast_horizon}")
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

if __name__ == "__main__":
    # 示例调用
    input_path = "../../data/features/selected_features.csv"
    output_dir = "../../data/features"
    create_dataset(input_path, output_dir, window_size=3, forecast_horizon=1)