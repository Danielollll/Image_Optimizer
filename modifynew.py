import numpy as np
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def load_image(input_image_path):
    """
    读取图像

    参数:
    input_image_path (str): 输入图像的路径

    返回:
    img (numpy.ndarray): 读取的图像
    """
    try:
        img = Image.open(input_image_path)
        return np.array(img)
    except Exception as e:
        print(f"无法读取图像 {input_image_path}: {e}")
        return None


def save_image(image, output_image_path):
    """
    保存图像

    参数:
    image (numpy.ndarray): 要保存的图像
    output_image_path (str): 图像保存路径
    """
    Image.fromarray(image).save(output_image_path)
    print(f"图像已保存到: {output_image_path}")


def compress_image(image, output_image_path, quality=85):
    """
    压缩图像并保存

    参数:
    image (PIL.Image.Image): PIL图像对象
    output_image_path (str): 压缩后图像保存的路径
    quality (int): 压缩质量，默认85
    """
    image.save(output_image_path, "JPEG", quality=quality)
    print(f"压缩后的图像已保存到: {output_image_path}")


def adjust_sharpness(image, output_image_path, sharpness_factor=2.0):
    """
    调整图像清晰度并保存

    参数:
    image (PIL.Image.Image): PIL图像对象
    output_image_path (str): 调整后图像保存的路径
    sharpness_factor (float): 清晰度因子，默认2.0
    """
    enhancer = ImageEnhance.Sharpness(image)
    img_enhanced = enhancer.enhance(sharpness_factor)
    img_enhanced.save(output_image_path)
    print(f"调整清晰度后的图像已保存到: {output_image_path}")


def crop_and_process_image(input_image_path, output_image_path):
    """
    裁剪图像并进行处理，包括压缩和调整清晰度

    参数:
    input_image_path (str): 输入图像路径
    output_image_path (str): 输出图像路径
    """
    image = load_image(input_image_path)
    if image is None:
        return

    fig, ax = plt.subplots()
    ax.imshow(image)

    def onselect(eclick, erelease, output_image_path):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        cropped_img = image[y1:y2, x1:x2]
        plt.close(fig)

        save_choice = input("是否保存裁剪后的图像？(y/n): ").strip().lower()
        if save_choice == 'y':
            save_image(cropped_img, output_image_path)

            # 打开保存后的图像进行进一步处理
            img = Image.open(output_image_path)

            # 询问用户是否压缩图像
            compress_choice = input("是否压缩图像？(y/n): ").strip().lower()
            if compress_choice == 'y':
                quality = get_integer_input("请输入压缩质量（1-100）：")
                compressed_image_path = os.path.join(os.path.dirname(output_image_path),
                                                     'compressed_' + os.path.basename(output_image_path))
                compress_image(img, compressed_image_path, quality)
                output_image_path = compressed_image_path

            # 询问用户是否调整清晰度
            sharpness_choice = input("是否调整图像清晰度？(y/n): ").strip().lower()
            if sharpness_choice == 'y':
                sharpness_factor = get_float_input("请输入清晰度因子（建议范围：1.0-3.0）：")
                sharpened_image_path = os.path.join(os.path.dirname(output_image_path),
                                                    'sharpened_' + os.path.basename(output_image_path))
                adjust_sharpness(img, sharpened_image_path, sharpness_factor)
        else:
            print("裁剪操作未保存")

    rect_selector = RectangleSelector(ax, lambda eclick, erelease: onselect(eclick, erelease, output_image_path), interactive=True)
    plt.show()


def get_integer_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid input. Please enter an integer.")


def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a float.")


# 示例调用
input_path = '/Users/liyue/Desktop/images/3.png'
output_path = '/Users/liyue/Desktop/images/4.png'
crop_and_process_image(input_path, output_path)
