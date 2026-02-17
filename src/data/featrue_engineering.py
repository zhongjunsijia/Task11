import pandas as pd
import numpy as np

def create_features(input_path, output_path):
    # 1. 读取数据
    data = pd.read_csv(input_path)

    # 2. 计算风速和风向（如果数据中没有）
    # 由于我们的数据中没有风速和风向，这里跳过

    # 3. 时间特征已经在 data_preprocessing.py 中添加

    # 4. 保存结果
    data.to_csv(output_path, index=False)
    print(f"特征工程完成，数据已保存至 {output_path}")

if __name__ == "__main__":
    # 示例调用
    input_path = "../../data/air_quality/processed/aligned_data.csv"
    output_path = "../../data/features/derived_features.csv"
    create_features(input_path, output_path)