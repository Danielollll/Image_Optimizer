import cv2 as cv
import numpy as np
import json
from matplotlib import pyplot as plt


def get_avg_brightness(img):
    average_brightness = np.mean(img)
    return average_brightness


def get_perceived_avg_brightness(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_values = hsv_image[:, :, 2]
    average_value = np.mean(value_values)
    return average_value


def get_contrast(img):
    grayscale_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    min_intensity = np.min(grayscale_image)
    max_intensity = np.max(grayscale_image)
    contrast = (float(max_intensity) - float(min_intensity)) / (float(max_intensity) + float(min_intensity))
    return contrast


def get_hue(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hue_values = hsv_image[:, :, 0]
    average_hue = np.mean(hue_values)
    return average_hue


def get_saturation(img):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    saturation_values = hsv_image[:, :, 1]
    average_saturation = np.mean(saturation_values)
    return average_saturation


def get_white_balance(img):
    reference_channel = img[:, :, 2]
    mean_reference = np.mean(reference_channel)
    mean_red = np.mean(img[:, :, 2])
    mean_green = np.mean(img[:, :, 1])
    mean_blue = np.mean(img[:, :, 0])
    balance_ratio_red = mean_reference / mean_red
    balance_ratio_green = mean_reference / mean_green
    balance_ratio_blue = mean_reference / mean_blue
    return balance_ratio_red, balance_ratio_green, balance_ratio_blue


def resize_img(img, width, height):
    resized_image = cv.resize(img, (width, height))
    return resized_image


def plot_grayscale_hist(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def plot_color_hist(img):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def image_analysis(image_path):
    src = cv.imread(image_path)
    resized_image = resize_img(src, 800, 600)

    # Get image parameter values
    WB_red, WB_green, WB_blue = get_white_balance(src)
    average_brightness = get_avg_brightness(resized_image)
    contrast = get_contrast(resized_image)
    average_hue = get_hue(resized_image)
    average_saturation = get_saturation(resized_image)
    average_perceived_brightness = get_perceived_avg_brightness(resized_image)

    # Store img parameters into a txt file in JSON
    results = {
        "File_name": image_path,
        "WB_red": WB_red,
        "WB_green": WB_green,
        "WB_blue": WB_blue,
        "avg_brightness": average_brightness,
        "contrast": contrast,
        "avg_hue": average_hue,
        "avg_saturation": average_saturation,
        "avg_perceived_brightness": average_perceived_brightness
    }
    with open('result.json', 'a') as f:
        json.dump(results, f, indent=4)
        f.write('\n')

    # Successfully executed notification
    print("参数计算完成，结果已保存到image_analysis_result.json")

    # Plot img hists
    plot_grayscale_hist(resized_image)
    plot_color_hist(resized_image)

    # Display resized img
    cv.imshow("Resized Image", resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Example
instance = r"test.jpg"
image_analysis(instance)
