from skimage.segmentation import slic, mark_boundaries
import utilities as utils
import cv2
import numpy as np


# SLIC Segmentation
# If the image given is grayscale we lower compactness to avoid grid-like segmentation
def calculate_slic_superpixels(image, superpixels_number=100, gray=False):
    if not gray:
        segments = slic(image, n_segments=superpixels_number)
    else:
        segments = slic(image, n_segments=superpixels_number, compactness=0.1)
    return segments


# Displaying SLIC segmentation using mark boundaries
def show_slic(image, segments):
    marked_image = mark_boundaries(image, segments)
    utils.display_image(marked_image, name="SLIC Superpixels (100)")


# Get the individual superpixels in a list
def get_slic_superpixels(image, superpixels_number=100, gray=False):
    print("[!] Started SLIC segmentation [!]")
    superpixels_list = []
    segments = calculate_slic_superpixels(image, superpixels_number, gray)
    for(i, segVal) in enumerate(np.unique(segments)):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        mask[segments == segVal] = 255
        superpixel = cv2.bitwise_and(image, image, mask=mask)
        superpixels_list.append(superpixel)
    print("SLIC segmentation finished!")
    return superpixels_list
