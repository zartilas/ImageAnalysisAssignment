import cv2
import numpy                   as np
import slic_superpixels        as ssp

# This is done to use multi-threads to have faster results.
cv2.setUseOptimized(True)
cv2.setNumThreads(cv2.getNumberOfCPUs())


def load_rgb_image(path):
    rgb_img = cv2.imread(path)
    return rgb_img


def rgb2lab(rgb):
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab_img


def lab2rgb(lab):
    rgb_image = cv2.cvtColor(lab, cv2.COLOR_LAB2LRGB)
    return rgb_image


def rgb2gray(rgb):
    gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray_img


def display_image(img, name="Image"):
    cv2.imshow(name, np.hstack([img]))
    cv2.waitKey(0)


def display_end_result(original_img, gray_img, colors, sample_img):
    segments = ssp.calculate_slic_superpixels(gray_img, gray=True)
    (h, w) = gray_img.shape
    a = []
    b = []
    for i in range(h):
        row_a = []
        row_b = []
        for j in range(w):
            color_index = segments[i][j]
            row_a.append(colors[color_index][0])
            row_b.append(colors[color_index][1])
        a.append(row_a)
        b.append(row_b)

    a = np.array(a)
    b = np.array(b)
    predicted_image = np.stack((gray_img, a, b), axis=2)

    rgb = lab2rgb(predicted_image)

    cv2.imshow("Original", np.hstack([original_img]))
    cv2.imshow("Colored LAB", np.hstack([predicted_image]))
    cv2.imshow("Colored RGB", np.hstack([rgb]))
    cv2.imshow("Grayscale Image", gray_img)

    sample_img_slic = ssp.calculate_slic_superpixels(load_rgb_image(sample_img), 100, False)
    ssp.show_slic(load_rgb_image(sample_img), sample_img_slic)

    cv2.waitKey(0)
