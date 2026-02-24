from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def select_features(input_path, output_path, importance_output_path):
    # 1. 读取数据
    data = pd.read_csv(input_path)

    # 2. 准备特征和目标变量
    features = ['temperature', 'humidity', 'pm10', 'no2', 'so2', 'co', 'wind_speed', 'hour', 'day', 'month', 'year', 'season']
    X = data[features]
    y = data['pm25']

    # 3. 使用随机森林进行特征重要性评估
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 4. 提取特征重要性
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    # 5. 选取累计贡献率达80%的特征
    cumulative_importance = feature_importance_df['importance'].cumsum()
    selected_features = feature_importance_df[cumulative_importance <= 0.8]['feature'].tolist()

    # 6. 保存选择特征后的数据
    X_selected = X[selected_features]
    selected_data = pd.concat([X_selected, y, data[['date', 'station_id']]], axis=1)
    selected_data.to_csv(output_path, index=False)

    # 7. 保存特征重要性结果
    feature_importance_df.to_csv(importance_output_path, index=False)

    print(f"特征选择完成，数据已保存至 {output_path}")
    print(f"特征重要性结果已保存至 {importance_output_path}")
    print(f"选择的特征数量: {len(selected_features)}")
    print(f"选择的特征: {selected_features}")

if __name__ == "__main__":
    # 示例调用
    input_path = "../../data/features/derived_features.csv"
    output_path = "../../data/features/selected_features.csv"
    importance_output_path = "../../data/features/feature_importance.csv"
    select_features(input_path, output_path, importance_output_path)