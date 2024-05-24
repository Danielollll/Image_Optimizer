import os
import json
import cv2 as cv
import numpy as np
import pandas as pd


# Scale 0-100
def normalize_value(value, min_value, max_value):
    # Ensure the values are within the correct range
    if value < min_value:
        value = min_value
    elif value > max_value:
        value = max_value
    # Normalize the value to a 0-100 scale
    normalized_value = 100.0 * (value - min_value) / (max_value - min_value)
    return normalized_value


def denormalize_value(normalized_value, min_value, max_value):
    denormalized_value = (normalized_value / 100.0) * (max_value - min_value) + min_value
    return denormalized_value


# Functions
def get_noise(img):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    noise_level = np.std(grayscale_image)
    normalized_noise = normalize_value(noise_level, 0, 255)
    return normalized_noise


def modify_noise(img, noise_factor):
    noise_factor_denormalized = denormalize_value(noise_factor, 0, 15)
    gaussian_noise = np.random.normal(0, noise_factor_denormalized, img.shape)
    noisy_img = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_img


def get_hue(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hue_values = hsv_image[:, :, 0]
    average_hue = np.mean(hue_values)
    normalized_hue = normalize_value(average_hue, 0, 179)
    return normalized_hue


def modify_hue(img, hue_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hue_values = hsv_image[:, :, 0]
    target_hue = denormalize_value(hue_factor, 0, 179)
    hue_adjustment = target_hue - np.mean(hue_values)
    # Ensure hue_values and hue_adjustment are numpy arrays
    if isinstance(hue_values, np.ndarray) and isinstance(hue_adjustment, (int, float, np.ndarray)):
        hsv_image[:, :, 0] = (hue_values + hue_adjustment) % 180
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def get_brightness(img):
    average_brightness = np.mean(img)
    normalized_brightness = normalize_value(average_brightness, 0, 255)
    return normalized_brightness


def modify_brightness(img, brightness_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    current_avg_brightness = np.mean(hsv_image[:, :, 2])
    target_brightness = denormalize_value(brightness_factor, 0, 255)
    if current_avg_brightness == 0:
        brightness_adjustment = 1
    else:
        brightness_adjustment = target_brightness / current_avg_brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_adjustment, 0, 255).astype(np.uint8)
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def get_perceived_avg_brightness(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_values = hsv_image[:, :, 2]
    average_value = np.mean(value_values)
    normalized_value = normalize_value(average_value, 0, 255)
    return normalized_value


def modify_perceived_avg_brightness(img, target_avg_value):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_values = hsv_image[:, :, 2]
    current_avg_value = np.mean(value_values)
    target_brightness = denormalize_value(target_avg_value, 0, 255)
    if current_avg_value == 0:
        brightness_adjustment = 1
    else:
        brightness_adjustment = target_brightness / current_avg_value
    adjusted_value = np.clip(value_values * brightness_adjustment, 0, 255)
    hsv_image[:, :, 2] = adjusted_value.astype(np.uint8)
    adjusted_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return adjusted_img


def get_white_balance(img):
    # Calculate mean values for each color channel
    mean_red = np.mean(img[:, :, 2])
    mean_green = np.mean(img[:, :, 1])
    mean_blue = np.mean(img[:, :, 0])
    # Calculate mean intensity of the grayscale image
    mean_gray = np.mean(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
    # Avoid division by zero
    if mean_red == 0 or mean_green == 0 or mean_blue == 0:
        return 0, 0, 0
    # Calculate balance ratios for each channel
    balance_ratio_red = mean_gray / mean_red
    balance_ratio_green = mean_gray / mean_green
    balance_ratio_blue = mean_gray / mean_blue
    # Normalize each ratio separately
    normalized_red = normalize_value(balance_ratio_red, 0, 2)
    normalized_green = normalize_value(balance_ratio_green, 0, 2)
    normalized_blue = normalize_value(balance_ratio_blue, 0, 2)
    return normalized_red, normalized_green, normalized_blue


def modify_white_balance(image, balance_ratio_red=-1, balance_ratio_green=-1, balance_ratio_blue=-1):
    image = image.astype(np.float32)
    b, g, r = cv.split(image)
    # If balance ratios are given as normalized values (0-100), convert them to appropriate ratios
    if balance_ratio_red != -1:
        balance_ratio_red = denormalize_value(balance_ratio_red, 0, 2)
        r = r * balance_ratio_red
    if balance_ratio_green != -1:
        balance_ratio_green = denormalize_value(balance_ratio_green, 0, 2)
        g = g * balance_ratio_green
    if balance_ratio_blue != -1:
        balance_ratio_blue = denormalize_value(balance_ratio_blue, 0, 2)
        b = b * balance_ratio_blue
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    balanced_image = cv.merge((b, g, r))
    balanced_image = balanced_image.astype(np.uint8)
    return balanced_image


def get_color_temperature(img):
    # Calculate the mean values for each color channel
    mean_red = np.mean(img[:, :, 0])
    mean_green = np.mean(img[:, :, 1])
    mean_blue = np.mean(img[:, :, 2])
    # Calculate color temperature based on the ratio of blue to red
    color_temperature = (mean_blue - mean_red) / (mean_blue + mean_green + mean_red)
    normalized_color_temperature = normalize_value(color_temperature, -100, 100)
    return normalized_color_temperature


def modify_color_temperature(img, temperature_factor):
    modified_img = img.copy().astype(np.float32)
    temperature_factor = denormalize_value(temperature_factor, -100, 100)
    # Adjust color temperature by scaling blue and red channels
    modified_img[:, :, 0] += temperature_factor
    modified_img[:, :, 2] -= temperature_factor
    # Clip values to ensure they remain in valid range
    modified_img = np.clip(modified_img, 0, 255)
    return modified_img.astype(np.uint8)


def get_contrast(img):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    contrast = np.std(grayscale_image)
    # Normalize to the range 0-100
    normalized_contrast = normalize_value(contrast, 0, 128)
    return normalized_contrast


def modify_contrast(img, contrast_factor):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mean_intensity = np.mean(grayscale_image)
    contrast_target = denormalize_value(contrast_factor, 0, 128)
    contrast_adjustment = (2 * contrast_target / 128) - 0.8
    # Adjust contrast for each channel separately
    adjusted_channels = []
    for i in range(3):
        adjusted_channel = np.clip((img[:, :, i] - mean_intensity) * (1 + contrast_adjustment) + mean_intensity, 0,
                                   255).astype(np.uint8)
        adjusted_channels.append(adjusted_channel)
    # Merge the adjusted channels back into a BGR image
    modified_img = cv.merge((adjusted_channels[0], adjusted_channels[1], adjusted_channels[2]))
    return modified_img


def get_saturation(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    saturation_values = hsv_image[:, :, 1]
    average_saturation = np.mean(saturation_values)
    normalized_saturation = normalize_value(average_saturation, 0, 255)
    return normalized_saturation


def modify_saturation(img, saturation_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    saturation_values = hsv_image[:, :, 1]
    current_avg_saturation = np.mean(saturation_values)
    target_saturation = denormalize_value(saturation_factor, 0, 255)
    if current_avg_saturation == 0:
        saturation_adjustment = 1
    else:
        saturation_adjustment = target_saturation / current_avg_saturation
    hsv_image[:, :, 1] = np.clip(saturation_values * saturation_adjustment, 0, 255).astype(np.uint8)
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def get_sharpness(img):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian_var = cv.Laplacian(grayscale_image, cv.CV_64F).var()
    # Adjusted max value for Laplacian variance based on typical image sharpness
    normalized_sharpness = normalize_value(laplacian_var, 0, 1000)
    return normalized_sharpness


def modify_sharpness(img, sharpen_factor):
    if sharpen_factor < 0:
        raise ValueError("sharpen_factor must be non-negative")
    sharpness_adjustment = denormalize_value(sharpen_factor, 0, 1000)
    sharpness_adjustment = (5 * sharpness_adjustment / 1000) - 1
    blurred_img = cv.GaussianBlur(img, (0, 0), 3)
    sharp_img = cv.addWeighted(img, 1 + sharpness_adjustment, blurred_img, -sharpness_adjustment, 0)
    return sharp_img


def get_highlights(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    highlights = hsv_image[:, :, 2].astype(np.float32)
    highlighted_pixels = highlights[highlights > 200]
    if highlighted_pixels.size == 0:
        return 0
    average_highlights = np.mean(highlighted_pixels)
    normalized_highlights = normalize_value(average_highlights, 200, 255)
    return normalized_highlights


def modify_highlights(img, highlights_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    current_avg_highlights = np.mean(value_channel)
    target_highlights = denormalize_value(highlights_factor, 50, 255)
    if current_avg_highlights == 0:
        highlights_adjustment = 1
    else:
        highlights_adjustment = target_highlights / current_avg_highlights
    value_channel = np.clip(value_channel * highlights_adjustment, 0, 255)
    hsv_image[:, :, 2] = value_channel.astype(np.uint8)
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def get_shadows(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    shadows = hsv_image[:, :, 2].astype(np.float32)
    shadowed_pixels = shadows[shadows < 50]
    if shadowed_pixels.size == 0:
        return 0
    average_shadows = np.mean(shadowed_pixels)
    normalized_shadows = normalize_value(average_shadows, 0, 50)
    return normalized_shadows


def modify_shadows(img, shadows_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    current_avg_shadows = np.mean(value_channel)
    target_shadows = denormalize_value(shadows_factor, 0, 380)
    if current_avg_shadows == 0:
        shadows_adjustment = 1
    else:
        shadows_adjustment = target_shadows / current_avg_shadows
    value_channel = np.clip(value_channel * shadows_adjustment, 0, 255)
    hsv_image[:, :, 2] = value_channel.astype(np.uint8)
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def get_exposure(img):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    average_exposure = np.mean(grayscale_image)
    normalized_exposure = normalize_value(average_exposure, 0, 255)
    return normalized_exposure


def modify_exposure(img, exposure_factor):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]
    current_avg_exposure = np.mean(value_channel)
    target_exposure = denormalize_value(exposure_factor, 0, 380)
    if current_avg_exposure == 0:
        exposure_adjustment = 1
    else:
        exposure_adjustment = target_exposure / current_avg_exposure
    value_channel = np.clip(value_channel * exposure_adjustment, 0, 255)
    hsv_image[:, :, 2] = value_channel.astype(np.uint8)
    modified_img = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return modified_img


def image_analysis(img_path, res_path):
    src = cv.imread(img_path)

    # Get image parameter values
    WB_red, WB_green, WB_blue = get_white_balance(src)
    average_brightness = get_brightness(src)
    contrast = get_contrast(src)
    average_hue = get_hue(src)
    average_saturation = get_saturation(src)
    average_perceived_brightness = get_perceived_avg_brightness(src)
    average_sharpen = get_sharpness(src)
    average_highlights = get_highlights(src)
    average_shadow = get_shadows(src)
    average_temperature = get_color_temperature(src)
    average_noisy = get_noise(src)
    average_exposure = get_exposure(src)

    # Store img parameters into a Pandas DataFrame
    results_df = pd.DataFrame({
        "File": [img_path],
        "contrast": [contrast],
        "WB_red": [WB_red],
        "WB_green": [WB_green],
        "WB_blue": [WB_blue],
        "avg_brightness": [average_brightness],
        "avg_perceived_brightness": [average_perceived_brightness],
        "avg_hue": [average_hue],
        "avg_saturation": [average_saturation],
        "avg_sharpness": [average_sharpen],
        "avg_highlights": [average_highlights],
        "avg_shadow": [average_shadow],
        "avg_temperature": [average_temperature],
        "avg_noisy": [average_noisy],
        "avg_exposure": [average_exposure]
    })

    # Convert DataFrame to JSON and save to file
    results_dict = results_df.to_dict(orient='records')[0]
    results_json = json.dumps(results_dict, indent=4)
    with open(res_path, 'a') as f:
        f.write(results_json)
        f.write('\n')

    # Successfully executed notification
    print("Image " + img_path + " parameters have been extracted to " + res_path)


def process_images_in_folders(root_dir):
    for sub_folder in os.listdir(root_dir):
        sub_folder_path = os.path.join(root_dir, sub_folder)
        if os.path.isdir(sub_folder_path):
            # Gather all image file paths in the sub-folder
            image_paths = [os.path.join(sub_folder_path, f) for f in os.listdir(sub_folder_path) if
                           os.path.isfile(os.path.join(sub_folder_path, f))]
            # Define the output JSON file path
            output_file = os.path.join(sub_folder_path, f"{sub_folder}_result.json")
            # Perform image analysis and save the results
            for image_path in image_paths:
                image_analysis(image_path, output_file)
            print(f"Processed {sub_folder} and saved results to {output_file}")


if __name__ == "__main__":
    # Define the root dataset directory
    dataset_dir = ".\dataset"
    # Process the images in each sub-folder
    process_images_in_folders(dataset_dir)
