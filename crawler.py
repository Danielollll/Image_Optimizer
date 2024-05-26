import os
import requests
import time
import customtkinter as ctk
from tkinter import messagebox, StringVar

# Unsplash API
api_url = "https://api.unsplash.com/search/photos"
default_access_key = ['-7mzLi8x261zMRKU-Rw_Qz_vIvaYvf2sWHV1l1pO5ZE',
                      'y0TRzwdfxA4PpZBPw3Lr8a6bYyAugiKItS7b_80UTA4']

# Crawl parameter
categories = {
    'animals': 'animals',
    'people': 'people',
    'food-drink': 'food drink',
    'nature': 'nature',
    'architecture-interior': 'architecture interior'
}

# Save directory and changelog file
os.makedirs('dataset', exist_ok=True)
changelog_path = os.path.join('dataset', 'changelog.txt')

# Initialize page and images_downloaded
category_state = {category: {'page': 1, 'images_downloaded': 0} for category in categories}

# Check if changelog file exists and read the state if it does
if os.path.exists(changelog_path):
    with open(changelog_path, 'r') as changelog_file:
        for line in changelog_file:
            category, page, images_downloaded = line.strip().split(',')
            category_state[category]['page'] = int(page)
            category_state[category]['images_downloaded'] = int(images_downloaded)
# Create changelog file
else:
    with open(changelog_path, 'w') as changelog_file:
        for category in categories:
            changelog_file.write(f'{category},1,0\n')

# Create directories for categories if they don't exist
for category in categories:
    os.makedirs(os.path.join('dataset', category), exist_ok=True)


# Crawling
def crawl_images(category, query, download_num, access_key):
    if download_num == 0:
        print("No need to update: " + category)
        return

    global category_state
    img_index = category_state[category]['images_downloaded']
    cur_page = category_state[category]['page']
    flag = 0
    if img_index % 10 != 0:
        cur_page -= 1
    real_img_index = int((cur_page - 1) * 10)

    try:
        while download_num > 0:
            params = {
                'query': query,
                'client_id': access_key,
                'per_page': 50,
                'page': cur_page
            }
            response = requests.get(api_url, params=params)

            if response.status_code == 200:
                data = response.json()
                images = data['results']
                print(f'Found {len(images)} images for category {category} on page {cur_page}')

                for idx, img in enumerate(images):
                    if download_num <= 0:
                        break
                    img_url = img['urls']['regular']
                    # Ignore crawled images
                    if flag == 0 and img_index > real_img_index:
                        real_img_index += 1
                        print(img_index)
                        print(real_img_index)
                        continue
                    else:
                        flag = 1
                    # Crawls and Store images
                    try:
                        img_data = requests.get(img_url).content
                        img_filename = os.path.join('dataset', category, f'{category}_{img_index + 1}.jpg')
                        with open(img_filename, 'wb') as handler:
                            handler.write(img_data)
                        print(f'Downloaded {img_filename}')
                        img_index += 1
                        download_num -= 1
                    except Exception as e:
                        messagebox.showerror("Download Error", f'Failed to download {img_url}: {e}')
                        break

                cur_page += 1
                time.sleep(1)
            else:
                messagebox.showerror("Download Error",
                                     f'Failed to fetch images for category {category}: {response.status_code}. '
                                     f'It can be caused by the API limit. Please try again after 1hr.')
                break
    finally:
        # Update changelog file after finishing each category
        category_state[category]['page'] = cur_page
        category_state[category]['images_downloaded'] = img_index
        with open(changelog_path, 'w') as new_changelog_file:
            for cat in categories:
                new_changelog_file.write(
                    f'{cat},{category_state[cat]["page"]},{category_state[cat]["images_downloaded"]}\n')

    print("Completed: " + category)


def start_crawling():
    start_button.configure(text="Processing", fg_color="red")
    start_button.update()

    access_key = access_key_combo.get()
    try:
        total_images_to_download = {
            'animals': int(animals_var.get()),
            'people': int(people_var.get()),
            'food-drink': int(food_drink_var.get()),
            'nature': int(nature_var.get()),
            'architecture-interior': int(architecture_interior_var.get())
        }

        # Validate input values
        total_count = sum(total_images_to_download.values())
        if total_count <= 0:
            messagebox.showerror("Invalid input", "Total number of images to download must be greater than 0.")
            return
        if total_count > 500:
            messagebox.showerror("Invalid input", "The total number of images to download must not exceed 500.")
            return

        # Start Crawling
        for curr_category, curr_query in categories.items():
            print(curr_category)
            crawl_images(curr_category, curr_query, total_images_to_download[curr_category], access_key)

    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid integer values.")
        return

    finally:
        start_button.configure(text="Start Crawling", fg_color="#3b8ed0")
        start_button.update()


if __name__ == "__main__":
    # Create the main window
    root = ctk.CTk()
    root.title("Image Crawler")
    root.iconbitmap('./icon/icon.ico')

    # Create StringVar instances for each entry
    animals_var = StringVar(value="0")
    people_var = StringVar(value="0")
    food_drink_var = StringVar(value="0")
    nature_var = StringVar(value="0")
    architecture_interior_var = StringVar(value="0")

    # Create and place widgets
    ctk.CTkLabel(root, text="Unsplash API Access Key:").grid(row=0, column=0, padx=10, pady=5)
    access_key_combo = ctk.CTkComboBox(root, values=default_access_key, width=200)
    access_key_combo.grid(row=0, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Animals:").grid(row=1, column=0, padx=10, pady=5)
    animals_entry = ctk.CTkEntry(root, textvariable=animals_var, width=100)
    animals_entry.grid(row=1, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="People:").grid(row=2, column=0, padx=10, pady=5)
    people_entry = ctk.CTkEntry(root, textvariable=people_var, width=100)
    people_entry.grid(row=2, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Food & Drink:").grid(row=3, column=0, padx=10, pady=5)
    food_drink_entry = ctk.CTkEntry(root, textvariable=food_drink_var, width=100)
    food_drink_entry.grid(row=3, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Nature:").grid(row=4, column=0, padx=10, pady=5)
    nature_entry = ctk.CTkEntry(root, textvariable=nature_var, width=100)
    nature_entry.grid(row=4, column=1, padx=10, pady=5)

    ctk.CTkLabel(root, text="Architecture & Interior:").grid(row=5, column=0, padx=10, pady=5)
    architecture_interior_entry = ctk.CTkEntry(root, textvariable=architecture_interior_var, width=100)
    architecture_interior_entry.grid(row=5, column=1, padx=10, pady=5)

    start_button = ctk.CTkButton(root, text="Start Crawling", command=start_crawling)
    start_button.grid(row=6, column=0, columnspan=2, padx=10, pady=20)

    # Start Window
    root.mainloop()
