import cv2
import os
from PIL import Image, ImageEnhance


def crop(input_image_path):
    """
    选择图像中的ROI并返回裁剪后的图像

    参数:
    input_image_path (str): 输入图像的路径

    返回:
    crop (numpy.ndarray): 裁剪后的图像
    """
    img = cv2.imread(input_image_path)

    # 检查图像是否成功读取
    if img is None:
        print(f"无法读取图像 {input_image_path}")
        return None

    # 显示原始图像
    cv2.imshow('original', img)

    # 选择ROI
    roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    print("ROI:", roi)

    # 显示ROI
    if roi != (0, 0, 0, 0):
        crop = img[y:y + h, x:x + w]
        cv2.imshow('crop', crop)
        print("裁剪后的图像显示成功")

        # 退出
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return crop
    else:
        cv2.destroyAllWindows()
        return None


def save_image(image, output_image_path):
    """
    保存图像

    参数:
    image (numpy.ndarray): 要保存的图像
    output_image_path (str): 图像保存路径
    """
    cv2.imwrite(output_image_path, image)
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

cropped_img = crop(input_path)
if cropped_img is not None:
    save_choice = input("是否保存裁剪后的图像？(y/n): ").strip().lower()
    if save_choice == 'y':
        save_image(cropped_img, output_path)

        # 打开保存后的图像进行进一步处理
        img = Image.open(output_path)

        # 询问用户是否压缩图像
        compress_choice = input("是否压缩图像？(y/n): ").strip().lower()
        if compress_choice == 'y':
            quality = get_integer_input("请输入压缩质量（1-100）：")
            compressed_image_path = os.path.join(os.path.dirname(output_path),
                                                 'compressed_' + os.path.basename(output_path))
            compress_image(img, compressed_image_path, quality)
            output_path = compressed_image_path

        # 询问用户是否调整清晰度
        sharpness_choice = input("是否调整图像清晰度？(y/n): ").strip().lower()
        if sharpness_choice == 'y':
            sharpness_factor = get_float_input("请输入清晰度因子（建议范围：1.0-3.0）：")
            sharpened_image_path = os.path.join(os.path.dirname(output_path),
                                                'sharpened_' + os.path.basename(output_path))
            adjust_sharpness(img, sharpened_image_path, sharpness_factor)
else:
    print("未选择有效的ROI进行裁剪")
