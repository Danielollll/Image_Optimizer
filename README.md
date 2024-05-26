# Image Optimizer
UIC DPW 24S **Group 11** course project

## Set Up
1. **Install packages using *requirements.txt*:** `pip install -r requirements.txt`.
2. **Ensure `.\icon` folder is correctly placed** in the same directory with the python files.
3. The prepared JSON filter datasets are placed in `.\dataset` folder. Although the program can run without these dataset files, you may not be able to observe the dataset and do image auto optimization as it requires dataset for analysis and training.
4. **Run** `main.py`.

## Usage: Image Optimization
1. **Adjust Atomic Parameter of an Image:** Click in and change the value in a text box located at the left-bottom of `Main` frame. Then click out of the canvas or simply click the *Image histogram* on the top to modify the image & refresh the window.
2. **Auto Optimization:** Choose a filter in the combo box located at the right-bottom of the `Main` frame, and click `Auto Optimizzation` button.
3. **Export Image:** Simple click `Export` button, then a new window will appear. You may crop, compress, and choose clarity of the output image.
4. **Revert Image:** We offer `Revert To Original` button to restart the editing process.

## Usage: Dataset Management
1. **New Dataset Creation:** Simply Create a new folder in `.\dataset` and push the `Update Dataset` button on the `Dataset Manamgement` frame.
2. **Dataset Update:** Add new images into the corresponding filter folder in`.\dataset` and push the `Update Dataset` button on the `Dataset Manamgement` frame.
3. **Update Dataset By Unsplash Crawler:** 
Open the `Dataset Management` frame, click `Get More Images from Unsplash`. 
Specify which filter datasets you want to update, each inner API can only crawl *500 images/hour*. 
After crawling, click `Dataset Update` button.
