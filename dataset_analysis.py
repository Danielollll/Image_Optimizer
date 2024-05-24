import pandas as pd
import json


def dataset_desc(file_path):
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

    # 将 JSON 对象的列表转换为 pandas DataFrame
    df = pd.DataFrame(json_objects)

    # 过滤出数值型列
    numeric_df = df.select_dtypes(include='number')

    # Create a list to collect data for all numeric columns
    boxplot_data = []

    # Create an empty DataFrame to store statistics
    stats_df = pd.DataFrame(columns=[
        'Parameter', 'Min', 'Max', 'Mean', 'Median', 'Mode', 'Range',
        'Q1', 'Q3', 'IQR', 'Variance', 'Standard Deviation', 'Skewness'
    ])

    # 生成统计信息并画图
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        boxplot_data.append(numeric_df[column].dropna())

        # 计算统计信息
        stats = {
            'Parameter': column,
            'Min': data.min(),
            'Max': data.max(),
            'Mean': data.mean(),
            'Median': data.median(),
            'Mode': data.mode().values[0] if not data.mode().empty else float('nan'),
            'Range': data.max() - data.min(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25),
            'Variance': data.var(),
            'Standard Deviation': data.std(),
            'Skewness': data.skew()
        }

        # Append the statistics to the stats DataFrame using concat
        stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)

    return stats_df, df
