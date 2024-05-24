import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


def remove_outliers(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    return df_cleaned


def optimal_val_predict(file_path, contrast, WB_red, WB_green, WB_blue, avg_brightness, avg_perceived_brightness
                        , avg_hue, avg_saturation, avg_sharpness, avg_highlights, avg_shadow, avg_temperature
                        , avg_noisy, avg_exposure):
    # 用于存储 JSON 对象的列表
    json_objects = []

    # 用于构建单个 JSON 对象的字符串
    current_json_str = ""

    # 分割json文件中不同图片数据
    with open(file_path, 'r') as file:
        for line in file:
            # 移除行首尾的空白字符
            line = line.strip()
            # 如果这一行以 { 开头，表示一个新的 JSON 对象开始
            if line.startswith("{"):
                current_json_str = line
            # 如果这一行以 } 结尾，表示一个 JSON 对象结束
            elif line.endswith("}"):
                current_json_str += line
                try:
                    # 尝试解析这个 JSON 字符串
                    json_obj = json.loads(current_json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                current_json_str = ""
            # 如果是 JSON 对象的一部分，则添加到当前的 JSON 字符串中
            else:
                current_json_str += line

    # 将优秀图片特征数据转换为 DataFrame
    excellent_df = pd.DataFrame(json_objects).select_dtypes(include='number')

    # Remove outliers from your DataFrame
    cleaned_df = remove_outliers(excellent_df)

    # 使用优秀图片特征数据作为训练数据
    df = cleaned_df.copy()
    target_df = cleaned_df

    # 标准化特征
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(df)
    y_scaled = scaler_y.fit_transform(target_df)

    # 初始化并训练神经网络模型
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
    model.fit(X_scaled, y_scaled)

    # 新图片的特征数据
    new_data = {
        "contrast": [contrast],
        "WB_red": [WB_red],
        "WB_green": [WB_green],
        "WB_blue": [WB_blue],
        "avg_brightness": [avg_brightness],
        "avg_perceived_brightness": [avg_perceived_brightness],
        "avg_hue": [avg_hue],
        "avg_saturation": [avg_saturation],
        "avg_sharpness": [avg_sharpness],
        "avg_highlights": [avg_highlights],
        "avg_shadow": [avg_shadow],
        "avg_temperature": [avg_temperature],
        "avg_noisy": [avg_noisy],
        "avg_exposure": [avg_exposure]
    }

    # 转换为 DataFrame
    new_df = pd.DataFrame(new_data)

    # 标准化新图片的特征数据
    new_X_scaled = scaler_X.transform(new_df)

    # 使用模型预测优化后的特征
    optimized_features_scaled = model.predict(new_X_scaled)

    # 反标准化优化后的特征
    optimized_features = scaler_y.inverse_transform(optimized_features_scaled)

    # # 计算预测值与真实值之间的均方误差 (Mean Squared Error, MSE)
    # import numpy as np
    # mse = np.mean((y_scaled - model.predict(X_scaled)) ** 2)
    # print("Mean Squared Error (MSE):", mse)

    # 显示优化后的特征
    optimized_df = pd.DataFrame(optimized_features, columns=target_df.columns)
    return optimized_df
