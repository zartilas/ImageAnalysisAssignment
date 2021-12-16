import utilities                 as utils
import image_segmentation        as csd
import slic_superpixels          as ssp
import surf_and_gabor_features   as surf_and_gabor
import svm
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    SUPERPIXELS_NUM = 100  # We create a variable with full caps to show that it's a constant
    SAMPLE_IMAGE = "assets\\sample_img.jpg"

    print("\nTRAINING STARTED")

    # initialization of arrays
    rgb_images = []
    lab_images = []

    # Loading all images that will be trained
    for i in range(1, len(os.listdir("assets"))):
        temp_image = utils.load_rgb_image("assets\\" + str(i) + ".jpg")
        rgb_images.append(temp_image)
        temp_image = utils.rgb2lab(temp_image)
        lab_images.append(temp_image)

    csd.fit_and_save_kmeans(lab_images)

    if os.path.exists("SVC.sav"):
        print("The fitted SVM was already saved, if you want to repeat the process you need to delete the file and "
              "restart.")
    else:
        slic_list = []
        surf_list = []
        gabor_list = []

        for img in rgb_images:
            # SLIC segmentation and feature extraction per superpixel
            slic = ssp.get_slic_superpixels(img, superpixels_number=SUPERPIXELS_NUM)
            slic_list.extend(slic)

            surf = surf_and_gabor.extract_surf_for_each_superpixel(slic)
            surf_list.extend(surf)

            gabor = surf_and_gabor.extract_gabor_for_each_superpixel(slic)
            gabor_list.extend(gabor)

        # Fitting and saving SVM
        svm.fit_and_save_svm(slic_list, surf_list, gabor_list)

    print("\nTESTING STARTED")

    # Load fitted models
    svc = svm.load_svm()
    kmeans = csd.load_kmeans()

    # Load test image of car and make it grayscale
    colored_img = utils.load_rgb_image(SAMPLE_IMAGE)
    gray_img = utils.rgb2gray(colored_img)

    # SLIC segmentation and feature extraction per superpixel
    slic = ssp.get_slic_superpixels(gray_img, gray=True, superpixels_number=SUPERPIXELS_NUM)
    surf = surf_and_gabor.extract_surf_for_each_superpixel(slic, True)
    gabor = surf_and_gabor.extract_gabor_for_each_superpixel(slic, True)

    print("[!] Preparing testing samples [!]")
    test_sample = svm.prepare_test_sample(slic, surf, gabor)
    print("Testing samples have been prepared!")
    predictions = svc.predict(test_sample)
    print("Colors have been predicted successfully!")
    colors = kmeans.cluster_centers_.astype("uint8")[predictions]

    utils.display_end_result(colored_img, gray_img, colors, SAMPLE_IMAGE)
