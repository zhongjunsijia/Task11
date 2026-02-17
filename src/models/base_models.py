from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_rf_model(train_data_dir, model_save_path):
    # 1. 加载数据
    X_train = np.load(f"{train_data_dir}/X_train_horizon1.npy")
    y_train = np.load(f"{train_data_dir}/y_train_horizon1.npy").ravel()
    X_val = np.load(f"{train_data_dir}/X_val_horizon1.npy")
    y_val = np.load(f"{train_data_dir}/y_val_horizon1.npy").ravel()
    X_test = np.load(f"{train_data_dir}/X_test_horizon1.npy")
    y_test = np.load(f"{train_data_dir}/y_test_horizon1.npy").ravel()

    # 2. 展平输入（RF不支持时序输入）
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # 3. 直接使用默认参数训练模型（适应小数据集）
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train_flat, y_train)

    print("使用默认参数训练模型")

    # 5. 保存模型
    joblib.dump(rf, model_save_path)

    # 6. 评估
    y_pred = rf.predict(X_test_flat)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"RF模型评估结果:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

if __name__ == "__main__":
    # 示例调用
    train_data_dir = "../../data/features"
    model_save_path = "../../models/rf_model_horizon1.pkl"
    train_rf_model(train_data_dir, model_save_path)