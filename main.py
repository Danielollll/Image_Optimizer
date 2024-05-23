import os
import cv2 as cv
import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import img_func as img_func
import export_func


def raise_frame(next_frame):
    next_frame.tkraise()


def on_close():
    root.quit()
    root.destroy()


def select_img():
    filename = filedialog.askopenfilename(
        title="Select An Image",
        filetypes=(("Image", "*.png"), ("Image", "*.jpg"), ("Image", "*.jpeg"))
    )
    if filename:
        global img_name
        img_name = os.path.basename(filename)
        img_pil = Image.open(filename)
        img = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
        if img is not None:
            raise_frame(f_main)
            plot_img_hist(img, f_main_left_top, 7, 3)
            display_image(img, f_main_right_top)
            # Get image parameter values
            WB_red, WB_green, WB_blue = img_func.get_white_balance(img)
            average_brightness = img_func.get_avg_brightness(img)
            contrast = img_func.get_contrast(img)
            average_hue = img_func.get_hue(img)
            average_saturation = img_func.get_saturation(img)
            average_perceived_brightness = img_func.get_perceived_avg_brightness(img)
            average_sharpen = img_func.get_sharpness(img)
            average_highlights = img_func.get_highlights(img)
            average_shadow = img_func.get_shadows(img)
            average_temperature = img_func.get_color_temperature(img)
            average_noisy = img_func.get_noise(img)
            average_exposure = img_func.get_exposure(img)
            # Get image parameter values
            parameters = {
                "Red": WB_red,
                "Green": WB_green,
                "Blue": WB_blue,
                "Contrast": contrast,
                "Brightness": average_brightness,
                "Perceived Brightness": average_perceived_brightness,
                "Hue": average_hue,
                "Saturation": average_saturation,
                "Sharpness": average_sharpen,
                "Highlight": average_highlights,
                "Shadow": average_shadow,
                "Temperature": average_temperature,
                "Noise": average_noisy,
                "Exposure": average_exposure
            }
            display_parameters(f_main_left_bottom, parameters, img)


def get_parameter_value(frame, parameter_name):
    for widget in frame.scrollable_frame.winfo_children():
        if isinstance(widget, ctk.CTkLabel):
            if widget["text"] == parameter_name:
                # Retrieve the corresponding textbox widget
                textbox_widget = frame.scrollable_frame.grid_slaves(row=widget.grid_info()["row"], column=1)[0]
                # Get the value from the textbox widget
                parameter_value = textbox_widget.get("1.0", "end-1c")  # Retrieve text content from the textbox
                return parameter_value
    return None  # Parameter not found


def display_parameters(frame, parameters, img):
    # Clear previous content
    for widget in frame.winfo_children():
        widget.destroy()

    global img_buffer
    img_buffer = img

    # Create a canvas
    canvas = ctk.CTkCanvas(frame, highlightthickness=0, bg="white")
    scrollbar = ctk.CTkScrollbar(frame, fg_color="white", command=canvas.yview)
    scrollable_frame = ctk.CTkFrame(canvas, fg_color="white")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    # Function to change background color on focus
    def on_focus_in(event):
        event.widget.configure(foreground="black")

    def on_focus_out(event):
        event.widget.configure(foreground="gray")

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack the canvas and scrollbar
    canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
    scrollbar.pack(side=ctk.LEFT, fill=ctk.Y)

    # Adding parameters to the scrollable frame
    for i, (param, value) in enumerate(parameters.items()):
        # Parameter name label
        label = ctk.CTkLabel(scrollable_frame, text=f"{param}", anchor="w", fg_color="white")
        label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

        # Parameter value textbox
        value_str = f"{value}"
        textbox = ctk.CTkTextbox(scrollable_frame, height=1, width=160, wrap="none", fg_color="#f0f0f0",
                                 text_color="gray")
        textbox.insert("1.0", value_str)
        textbox.grid(row=i, column=1, padx=10, pady=5, sticky="w")

        # Bind focus in and focus out events
        textbox.bind("<FocusIn>", on_focus_in)
        textbox.bind("<FocusOut>", on_focus_out)
        textbox.bind("<FocusOut>",
                     lambda event, param1=param, img1=img_buffer, textbox1=textbox: param_updator(param1, img1,
                                                                                                  textbox1))

    scrollable_frame.update_idletasks()


def param_updator(param, img_buf, textbox):
    try:
        new_val = float(textbox.get("1.0", "end-1c"))
    except ValueError:
        print(f"Invalid value for {param}: {textbox.get('1.0', 'end-1c')}")
        return
    global img_buffer
    modified_img = None
    if param == "Red":
        modified_img = img_func.modify_white_balance(img_buf, new_val, -1, -1)
    elif param == "Green":
        modified_img = img_func.modify_white_balance(img_buf, -1, new_val, -1)
    elif param == "Blue":
        modified_img = img_func.modify_white_balance(img_buf, -1, -1, new_val)
    elif param == "Contrast":
        modified_img = img_func.modify_contrast(img_buf, new_val)
    elif param == "Brightness":
        modified_img = img_func.modify_brightness(img_buf, new_val)
    elif param == "Perceived Brightness":
        modified_img = img_func.modify_perceived_avg_brightness(img_buf, new_val)
    elif param == "Hue":
        modified_img = img_func.modify_hue(img_buf, new_val)
    elif param == "Saturation":
        modified_img = img_func.modify_saturation(img_buf, new_val)
    elif param == "Sharpness":
        modified_img = img_func.modify_sharpness(img_buf, new_val)
    elif param == "Highlight":
        modified_img = img_func.modify_highlights(img_buf, new_val)
    elif param == "Shadow":
        modified_img = img_func.modify_shadows(img_buf, new_val)
    elif param == "Temperature":
        modified_img = img_func.modify_color_temperature(img_buf, new_val)
    if param == "Noise":
        modified_img = img_func.modify_noise(img_buf, new_val)
    elif param == "Exposure":
        modified_img = img_func.modify_exposure(img_buf, new_val)
    else:
        pass
    if modified_img is not None:
        img_buffer = modified_img
        plot_img_hist(img_buffer, f_main_left_top, 7, 3)
        display_image(img_buffer, f_main_right_top)


def display_image(img, frame):
    # Clear previous image
    for widget in frame.winfo_children():
        widget.destroy()

    # Convert the image to RGB format
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Get the height and width of the image
    height, width = img.shape[:2]

    # Calculate the maximum width and height
    if height > width:
        max_width = 380
    else:
        max_width = 500
    img_ratio = width / height
    max_height = int(max_width / img_ratio)

    # Display image in the frame
    img_tk = ctk.CTkImage(img_pil, size=(max_width, max_height))
    label = ctk.CTkLabel(frame, image=img_tk, text="")
    label.image = img_tk
    label.pack()


def plot_img_hist(img, frame, width, height):
    # Clear previous plots
    for widget in frame.winfo_children():
        widget.destroy()

    plt.figure(figsize=(width, height))

    # Plot histogram for grayscale hist
    plt.hist(img.ravel(), 256, [0, 256])
    # Color hist
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.xlabel("Bins")
    plt.ylabel("Pixel Number")

    # Hide the top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Get the current figure & draw
    figure = plt.gcf()
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    plt.close()


def export_img():
    if img_buffer is not None:
        # Encode the original image name using UTF-8 to ensure consistent encoding
        encoded_img_name = img_name.encode('utf-8')
        # Construct the full temporary file path with the original image name
        tmp_file_path = os.path.join(r".\temp", encoded_img_name.decode('utf-8'))
        # Save the image to the temporary file
        cv.imwrite(tmp_file_path, img_buffer)
        # Call the export function with the temporary file path
        export_func.create_export_window(tmp_file_path)


# Main window
img_buffer = None
bg_color = "white"
root = ctk.CTk(bg_color)
root.title("Image Optimizer")
root.minsize(800, 500)
root.iconbitmap('./icon/icon.ico')
root.protocol("WM_DELETE_WINDOW", on_close)

# Frames
f_wizard = ctk.CTkFrame(root, fg_color=bg_color)
f_main = ctk.CTkFrame(root, fg_color=bg_color)
f_dataset = ctk.CTkFrame(root, fg_color=bg_color)

for f in (f_wizard, f_main, f_dataset):
    f.grid(row=0, column=0, sticky="nsew")

# Configure grid to expand
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# Wizard Interface
f_wizard_bg_color = "white"

# Wizard Interface Left
f_wizard_left = ctk.CTkFrame(f_wizard, fg_color=f_wizard_bg_color)
f_wizard_left.grid(row=0, column=0, sticky="nsew")
f_wizard_left.grid_rowconfigure(0, weight=1)
f_wizard_left.grid_columnconfigure(0, weight=1)

# Wizard Interface Right
f_wizard_right = ctk.CTkFrame(f_wizard, fg_color=f_wizard_bg_color)
f_wizard_right.grid(row=0, column=1, sticky="nsew")
f_wizard_right.grid_rowconfigure(0, weight=1)
f_wizard_right.grid_columnconfigure(0, weight=1)

# Configure grid in f_wizard
f_wizard.grid_rowconfigure(0, weight=1)
f_wizard.grid_columnconfigure(0, weight=1)
f_wizard.grid_columnconfigure(1, weight=1)

# Load Wizard Interface Icons
icon_select = ctk.CTkImage(Image.open("./icon/select.png"), size=(100, 100))
icon_db_man = ctk.CTkImage(Image.open("./icon/edit.png"), size=(100, 100))

# Buttons with icons and custom styles
select_img_button = ctk.CTkButton(
    f_wizard_left, text="Select An Image", command=select_img,
    image=icon_select, compound="top", fg_color="transparent", text_color="black", hover_color="lightblue"
)
select_img_button.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

db_manage_button = ctk.CTkButton(
    f_wizard_right, text="Dataset Management", command=lambda: raise_frame(f_dataset),
    image=icon_db_man, compound="top", fg_color="transparent", text_color="black", hover_color="lightblue"
)
db_manage_button.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# Main Interface
f_main_bg_color = "white"

# Main Interface Left
f_main_left = ctk.CTkFrame(f_main, fg_color=f_main_bg_color)
f_main_left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
f_main_left.grid_rowconfigure(0, weight=1)
f_main_left.grid_rowconfigure(1, weight=1)  # Button
f_main_left.grid_columnconfigure(0, weight=1)

# Main Interface Left Top
f_main_left_top = ctk.CTkFrame(f_main_left, fg_color=f_main_bg_color)
f_main_left_top.grid(row=0, column=0, sticky="nsew")
f_main_left_top.grid_rowconfigure(0, weight=1)
f_main_left_top.grid_columnconfigure(0, weight=1)

# Main Interface Left Bottom
f_main_left_bottom = ctk.CTkFrame(f_main_left, fg_color=f_main_bg_color)
f_main_left_bottom.grid(row=1, column=0, sticky="nsew")  # Button
f_main_left_bottom.grid_rowconfigure(0, weight=1)
f_main_left_bottom.grid_columnconfigure(0, weight=1)

# Main Interface Right
f_main_right = ctk.CTkFrame(f_main, fg_color=f_main_bg_color)
f_main_right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
f_main_right.grid_rowconfigure(0, weight=1)
f_main_right.grid_rowconfigure(1, weight=1)
f_main_right.grid_columnconfigure(0, weight=1)

# Main Interface Right Top
f_main_right_top = ctk.CTkFrame(f_main_right, fg_color=f_main_bg_color)
f_main_right_top.grid(row=0, column=0, sticky="nsew")
f_main_right_top.grid_rowconfigure(0, weight=1)
f_main_right_top.grid_columnconfigure(0, weight=1)

# Main Interface Right Bottom
f_main_right_bottom = ctk.CTkFrame(f_main_right, fg_color=f_main_bg_color)
f_main_right_bottom.grid(row=1, column=0, sticky="nsew")
f_main_right_bottom.grid_rowconfigure(0, weight=1)
f_main_right_bottom.grid_columnconfigure(0, weight=1)

auto_button = ctk.CTkButton(f_main_right_bottom, text="New Image", command=lambda: select_img())
auto_button.pack(padx=20, pady=20, fill='x')

export_button = ctk.CTkButton(f_main_right_bottom, text="Export Image",
                              command=export_img)
export_button.pack(padx=20, pady=20, fill='x')

# Configure grid in f_main
f_main.grid_rowconfigure(0, weight=1)
f_main.grid_columnconfigure(0, weight=1)
f_main.grid_columnconfigure(1, weight=1)

# Database Management Interface
button3 = ctk.CTkButton(f_dataset, text="Dataset Frame", command=lambda: raise_frame(f_wizard))
button3.pack(padx=20, pady=20, fill='x')

raise_frame(f_wizard)
root.mainloop()
