import os
import requests
import time

# Unsplash API
api_url = "https://api.unsplash.com/search/photos"
access_key = '-7mzLi8x261zMRKU-Rw_Qz_vIvaYvf2sWHV1l1pO5ZE'

# Crawl parameter
categories = {
    'animals': 'animals',
    'people': 'people',
    'food-drink': 'food drink',
    'nature': 'nature',
    'architecture-interior': 'architecture interior'
}

# Desired number of images to download for each category
total_images_to_download = {
    'animals': 0,
    'people': 10,
    'food-drink': 500,
    'nature': 500,
    'architecture-interior': 500
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
def crawl_images(category, query, download_num):
    global category_state
    img_index = category_state[category]['images_downloaded']
    cur_page = category_state[category]['page']

    try:
        while img_index < download_num:
            params = {
                'query': query,
                'client_id': access_key,
                'per_page': 50,
                'page': cur_page + 1
            }
            response = requests.get(api_url, params=params)

            if response.status_code == 200:
                data = response.json()
                images = data['results']
                print(f'Found {len(images)} images for category {category} on page {cur_page}')

                for idx, img in enumerate(images):
                    if img_index >= download_num:
                        break
                    img_url = img['urls']['regular']
                    try:
                        img_data = requests.get(img_url).content
                        img_filename = os.path.join('dataset', category, f'{category}_{img_index + 1}.jpg')
                        with open(img_filename, 'wb') as handler:
                            handler.write(img_data)
                        print(f'Downloaded {img_filename}')
                        img_index += 1
                    except Exception as e:
                        print(f'Failed to download {img_url}: {e}')
                cur_page += 1
                time.sleep(1)
            else:
                print(f'Failed to fetch images for category {category}: {response.status_code}')
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


# Instance
for curr_category, curr_query in categories.items():
    crawl_images(curr_category, curr_query, total_images_to_download[category])
