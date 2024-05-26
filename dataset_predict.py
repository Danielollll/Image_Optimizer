import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


def remove_outliers(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    df_cleaned = df[(z_scores < threshold).all(axis=1)]
    return df_cleaned


def optimal_val_predict(file_path, contrast, WB_red, WB_green, WB_blue, avg_brightness, avg_perceived_brightness,
                        avg_hue, avg_saturation, avg_sharpness, avg_highlights, avg_shadow, avg_temperature,
                        avg_noisy, avg_exposure):
    # Stores a list of JSON objects
    json_objects = []

    # A string that builds a single JSON object
    current_json_str = ""

    # Split different image data in json file
    with open(file_path, 'r') as file:
        for line in file:
            # Removes whitespace at the beginning and end of a line
            line = line.strip()
            # If the line begins with { , it means that a new JSON object starts
            if line.startswith("{"):
                current_json_str = line
            # If this line ends with } , it means that a JSON object ends
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

    # Convert good image feature data to DataFrame
    excellent_df = pd.DataFrame(json_objects).select_dtypes(include='number')

    # Remove outliers from your DataFrame
    cleaned_df = remove_outliers(excellent_df)

    # Use excellent image feature data as training data
    df = cleaned_df.copy()
    target_df = cleaned_df

    # Standardized feature
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(df)
    y_scaled = scaler_y.fit_transform(target_df)

    # Initialize and train the neural network model
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
    model.fit(X_scaled, y_scaled)

    # Feature data for new images
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

    # convert to DataFrame
    new_df = pd.DataFrame(new_data)

    # Standardize feature data for new images
    new_X_scaled = scaler_X.transform(new_df)

    # Use the model to predict optimized features
    optimized_features_scaled = model.predict(new_X_scaled)

    # The features after anti-standardization optimization
    optimized_features = scaler_y.inverse_transform(optimized_features_scaled)

    # # Calculate the mean square error between the predicted value and the true value (Mean Squared Error, MSE)
    # import numpy as np
    # mse = np.mean((y_scaled - model.predict(X_scaled)) ** 2)
    # print("Mean Squared Error (MSE):", mse)

    # Displays optimized features
    optimized_df = pd.DataFrame(optimized_features, columns=target_df.columns)
    return optimized_df
