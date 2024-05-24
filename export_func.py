import os
import sys
import tkinter as tk
from PIL import Image, ImageEnhance, ImageTk
from tkinter import filedialog, messagebox
from io import BytesIO


def save_image(image, output_image_path):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    ext = os.path.splitext(output_image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        image.save(output_image_path, format='JPEG')
    elif ext == '.png':
        image.save(output_image_path, format='PNG')
    else:
        image.save(output_image_path)
    print(f"Image saved to: {output_image_path}")
    root.quit()
    root.destroy()


def compress_image(image, quality=85):
    buffer = BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def adjust_sharpness(image, sharpness_factor=2.0):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness_factor)


def resize_image(image, max_width, max_height):
    ratio = min(max_width / image.width, max_height / image.height)
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)
    return image.resize((new_width, new_height), Image.LANCZOS), new_width, new_height


def update_image():
    global img_tk
    img = processed_image.copy()

    # Apply compression
    quality = compress_scale.get()
    img = compress_image(img, int(quality))

    # Apply sharpness
    sharpness = clarity_scale.get() / 50.0  # scale from 0-100 to 0-2
    img = adjust_sharpness(img, sharpness)

    # Update the displayed image
    img_tk = ImageTk.PhotoImage(img)
    canvas_export.delete("current_image")  # Delete the previous image
    canvas_export.create_image(0, 0, anchor=tk.NW, image=img_tk, tags="current_image")  # Add the new image with a tag

    # Draw crop rectangle
    canvas_export.delete("rect")
    canvas_export.create_rectangle(start_x, start_y, end_x, end_y, outline='red', tag="rect")


def on_canvas_click(event):
    global start_x, start_y
    start_x = event.x
    start_y = event.y
    canvas_export.delete("rect")


def on_canvas_drag(event):
    global start_x, start_y, end_x, end_y
    end_x = event.x
    end_y = event.y
    canvas_export.delete("rect")
    canvas_export.create_rectangle(start_x, start_y, end_x, end_y, outline='red', tag="rect")


def on_canvas_release(event):
    global end_x, end_y
    end_x = event.x
    end_y = event.y


def crop_and_process_image(filename):
    if processed_image is None:
        messagebox.showwarning("Warning", "Please select an image first!")
        return

    # Get the coordinates of the selected region
    x1 = min(start_x, end_x)
    y1 = min(start_y, end_y)
    x2 = max(start_x, end_x)
    y2 = max(start_y, end_y)

    cropped_img = processed_image.crop((x1, y1, x2, y2))
    compress_quality = compress_scale.get()
    sharpness_factor = clarity_scale.get() / 50.0  # scale from 0-100 to 0-2
    cropped_img = compress_image(cropped_img, int(compress_quality))
    cropped_img = adjust_sharpness(cropped_img, sharpness_factor)

    base_name, extension = os.path.splitext(filename)
    suggested_filename = base_name + "_processed" + extension

    output_path = filedialog.asksaveasfilename(
        title="Choose a file",
        initialfile=suggested_filename,
        filetypes=(("JPEG files", "*.jpg;*.jpeg"), ("PNG files", "*.png"), ("All files", "*.*"))
    )

    if output_path:
        save_image(cropped_img, output_path)
        messagebox.showinfo("Info", f"Image processed and saved to {output_path}")


def create_export_window(filename):
    global original_image, processed_image, compress_scale, clarity_scale, canvas_export, start_x, start_y, end_x, end_y, canvas_frame, root, img_tk
    root = tk.Tk()
    root.title("Image Export Window")
    root.configure(bg="#004c66")

    # Convert the image buffer to a PIL Image
    original_image = Image.open(filename)

    # Resize the image to fit the canvas
    processed_image, new_width, new_height = resize_image(original_image, 800, 600)

    # Initialize cropping variables
    start_x = 0
    start_y = 0
    end_x = new_width
    end_y = new_height

    # Set the root window geometry based on the image size
    root.geometry(f"{new_width + 20}x{new_height + 200}")

    # Set up the canvas frame
    canvas_frame = tk.Frame(root, bg='white')
    canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)

    # Set up the canvas
    canvas_export = tk.Canvas(canvas_frame, width=new_width, height=new_height, bg='white')
    canvas_export.pack(fill=tk.BOTH, expand=True)

    # Bind canvas events
    canvas_export.bind("<Button-1>", on_canvas_click)
    canvas_export.bind("<B1-Motion>", on_canvas_drag)
    canvas_export.bind("<ButtonRelease-1>", on_canvas_release)

    # Display the initial image
    img_tk = ImageTk.PhotoImage(processed_image)
    canvas_export.image = img_tk
    canvas_export.create_image(0, 0, anchor=tk.NW, image=img_tk)

    # Set up compression scale
    compress_label = tk.Label(root, text="Compression", bg="yellow")
    compress_label.pack(anchor='w', padx=10)

    compress_scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, bg="yellow",
                              command=lambda x: update_image())
    compress_scale.set(85)  # Set default value
    compress_scale.pack(fill='x', padx=10)

    # Set up clarity scale
    clarity_label = tk.Label(root, text="Clarity", bg="yellow")
    clarity_label.pack(anchor='w', padx=10)

    clarity_scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, bg="yellow", command=lambda x: update_image())
    clarity_scale.set(50)  # Set default value
    clarity_scale.pack(fill='x', padx=10)

    # Set up button frame
    button_frame = tk.Frame(root, bg="#004c66")
    button_frame.pack(pady=10)

    # Set up confirm button
    confirm_button = tk.Button(button_frame, text="Confirm", command=lambda: crop_and_process_image(filename),
                               bg="yellow")
    confirm_button.pack(side='left', padx=10)

    # Set up cancel button
    cancel_button = tk.Button(button_frame, text="Cancel", command=root.destroy, bg="yellow")
    cancel_button.pack(side='right', padx=10)

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Assuming the image file path is passed directly for testing
        img_path = filedialog.askopenfilename(title="Select an Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if img_path:
            create_export_window(img_path)
    else:
        tmp_file_path = sys.argv[1]
        create_export_window(tmp_file_path)
