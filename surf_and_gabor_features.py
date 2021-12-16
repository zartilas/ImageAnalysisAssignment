from mahotas.features import surf
import utilities as utils
from skimage.filters import gabor


def extract_surf(image):
    interest_points = surf.surf(image)
    return interest_points


def extract_gabor(image, freq=0.6):
    filter_real, filter_imaginary = gabor(image, frequency=freq)
    return filter_real


# Extracting SURF for each superpixel and after making it grayscale
def extract_surf_for_each_superpixel(superpixels_list, gray=False):
    print("[!] Extracting SURF features [!]")
    surf_of_superpixels = []
    for superpixel in superpixels_list:
        if not gray:
            gray_superpixel = utils.rgb2gray(superpixel)
        else:
            gray_superpixel = superpixel
        points = extract_surf(gray_superpixel)
        surf_of_superpixels.append(points)
    print("SURF extraction finished!")
    return surf_of_superpixels


# Extracting Gabor for each superpixel and after making it grayscale
def extract_gabor_for_each_superpixel(superpixels_list, gray=False):
    print("[!] Extracting Gabor features [!]")
    gabor_of_superpixels = []
    for superpixel in superpixels_list:
        if not gray:
            gray_superpixel = utils.rgb2gray(superpixel)
        else:
            gray_superpixel = superpixel
        filters = extract_gabor(gray_superpixel)
        gabor_of_superpixels.append(filters)
    print("Gabor extraction finished!")
    return gabor_of_superpixels
