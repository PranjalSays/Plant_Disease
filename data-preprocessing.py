## import all the essential libraries and modules
import os
import pathlib
import sklearn
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from PIL import Image

## function to reshape all the images from the current size to 256x256
def reshape_images(folder_path):
    parent_folder = folder_path
    image_size = (256,256) # desired image path, can also be changed accordingly

    # check for subfolders in the parent folder
    for folder in os.listdir(parent_folder):
        subfolder_path = os.path.join(folder_path, folder)

        # if the subfolder exists
        if os.path.isdir(subfolder_path):
            print(f"Inside subfolder: {folder}")

            # iterate over each image in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # check if image has the correct extension or not
                if filename.endswith(('.jpg', 'jpeg', '.png', '.bmp')):
                    print(f"Resizing Image: {filename}")

                    img = cv2.imread(file_path) # open image using openCV
                    resized_image = cv2.resize(img, image_size) # reshape the image
                    cv2.imwrite(resized_image, file_path) # save the resized image
                else:
                    print(f"Extension not supported.")
        else:
            print(f"Subfolders doesn't exists.")


## function to convert the colored images to grayscale, for better computation
def color_to_grayscale(folder_path):
    parent_folder = folder_path  # desired image path, can also be changed accordingly

    # check for subfolders in the parent folder
    for folder in os.listdir(parent_folder):
        subfolder_path = os.path.join(folder_path, folder)

        # if the subfolder exists
        if os.path.isdir(subfolder_path):
            print(f"Inside subfolder: {folder}")

            # iterate over each image in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # check if image has the correct extension or not
                if filename.endswith(('.jpg', 'jpeg', '.png', '.bmp')):
                    print(f"Resizing Image: {filename}")

                    img = cv2.imread(file_path)  # open image using openCV
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert the image using cvtColor
                    cv2.imwrite(gray_image, file_path)  # save the resized image
                else:
                    print(f"Extension not supported")
        else:
            print(f"Subfolders doesn't exists.")

## function to normalize the gray scale images using MinMaxScaler
def grayscale_normalization(folder_path):
    parent_folder = folder_path  # desired image path, can also be changed accordingly

    # check for subfolders in the parent folder
    for folder in os.listdir(parent_folder):
        subfolder_path = os.path.join(folder_path, folder)

        # if the subfolder exists
        if os.path.isdir(subfolder_path):
            print(f"Inside subfolder: {folder}")

            # iterate over each image in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # check if image has the correct extension or not
                if filename.endswith(('.jpg', 'jpeg', '.png', '.bmp')):
                    print(f"Resizing Image: {filename}")

                    img = cv2.imread(file_path)  # open image using openCV
                    normalized_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # grayscale image normalized using MinMaxScaler
                    cv2.imwrite(gray_image, file_path)  # save the resized image
                else:
                    print(f"Extension not supported")
        else:
            print(f"Subfolders doesn't exists.")

## function to clean duplicate images and very similar images
def clean_duplicate_images(folder_path):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    # Create a dictionary to store the images and their hashes
    image_hashes = {}

    # Iterate over the image files
    for image_file in image_files:
        # Open the image using OpenCV
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the hash of the image
        image_hash = cv2.img_hash.PHash(gray_image)

        # Check if the image hash is already in the dictionary
        if image_hash in image_hashes:
            # If the image hash is already in the dictionary, it's a duplicate image
            print(f"Removing duplicate image: {image_file}")
            os.remove(image_path)
        else:
            # If the image hash is not in the dictionary, add it to the dictionary
            image_hashes[image_hash] = image_file

    # Create a list to store the images that are very similar
    similar_images = []

    # Iterate over the image files again
    for image_file in image_files:
        # Open the image using OpenCV
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Iterate over the other image files
        for other_image_file in image_files:
            # Skip the same image
            if image_file == other_image_file:
                continue

            # Open the other image using OpenCV
            other_image_path = os.path.join(folder_path, other_image_file)
            other_image = cv2.imread(other_image_path)

            # Convert the other image to grayscale
            other_gray_image = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)

            # Calculate the SSIM between the two images
            ssim_value = ssim(gray_image, other_gray_image)

            # Check if the SSIM is above a certain threshold
            if ssim_value > 0.9:
                # If the SSIM is above the threshold, the images are very similar
                print(f"Removing very similar image: {other_image_file}")
                similar_images.append(other_image_file)

    # Remove the very similar images
    for similar_image in similar_images:
        os.remove(os.path.join(folder_path, similar_image))

if name == __main__:

    folder_path = input("Please Enter parent folder path: ")
    clean_duplicate_images(folder_path)
    reshape_images(folder_path)
    color_to_grayscale(folder_path)
    grayscale_normalization(folder_path)
