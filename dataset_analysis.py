import pandas as pd
import json


def dataset_desc(file_path):
    # Stores a list of JSON objects
    json_objects = []

    # A string that builds a single JSON object
    current_json_str = ""

    # Split different image data in json file
    with open(file_path, 'r') as file:
        for line in file:
            # Removes whitespace at the beginning and end of a line
            line = line.strip()
            # If the line begins with '{' , it means that a new JSON object starts
            if line.startswith("{"):
                current_json_str = line
            # If this line ends with '}' , it means that a JSON object ends
            elif line.endswith("}"):
                current_json_str += line
                try:
                    # Parsing JSON strings
                    json_obj = json.loads(current_json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                current_json_str = ""
            # If it is part of a JSON object, it is added to the current JSON string
            else:
                current_json_str += line

    # Convert a list of JSON objects to pandas DataFrame
    df = pd.DataFrame(json_objects)

    # Filter out numeric columns
    numeric_df = df.select_dtypes(include='number')

    # Create a list to collect data for all numeric columns
    boxplot_data = []

    # Create an empty DataFrame to store statistics
    stats_df = pd.DataFrame(columns=[
        'Parameter', 'Min', 'Max', 'Mean', 'Median', 'Mode', 'Range',
        'Q1', 'Q3', 'IQR', 'Variance', 'Standard Deviation', 'Skewness'
    ])

    # Generate statistics and draw graphs
    for column in numeric_df.columns:
        data = numeric_df[column].dropna()
        boxplot_data.append(numeric_df[column].dropna())

        # Computational statistics
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
