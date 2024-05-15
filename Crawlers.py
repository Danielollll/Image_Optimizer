import os
import requests
import time

access_key = 'y0TRzwdfxA4PpZBPw3Lr8a6bYyAugiKItS7b_80UTA4'

# Crawl parameter
categories = {
    'animals': 'animals',
    'people': 'people',
    'food-drink': 'food drink',
    'nature': 'nature',
    'architecture-interior': 'architecture interior'
}

# Save directory
os.makedirs('downloads', exist_ok=True)
for category in categories:
    os.makedirs(os.path.join('downloads', category), exist_ok=True)

# Unsplash API
api_url = "https://api.unsplash.com/search/photos"

# Crawling
for category, query in categories.items():
    total_images_to_download = 200
    images_downloaded = 0
    page = 1

    while images_downloaded < total_images_to_download:
        params = {
            'query': query,
            'client_id': access_key,
            'per_page': 30,
            'page': page
        }
        response = requests.get(api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            images = data['results']
            print(f'Found {len(images)} images for category {category} on page {page}')

            for idx, img in enumerate(images):
                if images_downloaded >= total_images_to_download:
                    break
                img_url = img['urls']['regular']
                try:
                    img_data = requests.get(img_url).content
                    img_filename = os.path.join('downloads', category, f'{category}_{images_downloaded + 1}.jpg')
                    with open(img_filename, 'wb') as handler:
                        handler.write(img_data)
                    print(f'Downloaded {img_filename}')
                    images_downloaded += 1
                except Exception as e:
                    print(f'Failed to download {img_url}: {e}')
            page += 1
            time.sleep(1)
        else:
            print(f'Failed to fetch images for category {category}: {response.status_code}')
            break

print("Download completed.")
