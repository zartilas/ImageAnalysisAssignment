from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import image_segmentation as csd
import utilities as imf
import pickle
from os import path


def fit_and_save_svm(superpixels_list, surf_list, gabor_list):
    if path.exists("SVC.sav"):
        print("Fitted SVM already saved -- If you wish to fit again stop execution, delete the file and rerun!")
        return
    print("[!] Started fitting SVM [!]")
    # Using LinearSVC because it's using 'one-vs-the-rest' multi-class strategy
    classifier = SVC(class_weight='balanced')
    loaded_kmeans = csd.load_kmeans()

    training_samples = []
    training_labels = []

    for i in range(len(superpixels_list)):
        features = []

        # SURF of each superpixel is the average of each pixel's SURF
        surf_of_sp = surf_list[i]
        surf_of_sp = np.average(surf_of_sp)
        surf_of_sp = np.nan_to_num(surf_of_sp)

        # Gabor of each superpixel is the average of each pixel's Gabor
        gabor_of_sp = gabor_list[i]
        gabor_of_sp = np.average(gabor_of_sp)
        gabor_of_sp = np.nan_to_num(gabor_of_sp)

        features.append(surf_of_sp)
        features.append(gabor_of_sp)

        # We find the dominant color of the superpixel
        color_of_sp = superpixels_list[i]
        color_of_sp = imf.rgb2lab(color_of_sp)
        color_of_sp = color_of_sp.reshape((color_of_sp.shape[0] * color_of_sp.shape[1]), 3)
        temp = []
        for j in color_of_sp:
            if not np.array_equal(j, [0, 128, 128]):
                temp.append([j[1], j[2]])

        color_of_sp = temp
        if len(color_of_sp) == 0:
            continue
        color_of_sp = loaded_kmeans.predict(color_of_sp)
        (values, counts) = np.unique(color_of_sp, return_counts=True)
        ind = np.argmax(counts)
        color_of_sp = values[ind]

        training_samples.append(features)
        training_labels.append(color_of_sp)

    fitted_svc = classifier.fit(training_samples, training_labels)

    pickle.dump(fitted_svc, open("SVC.sav", 'wb'))
    print("Finished fitting SVM!")


def load_svm():
    loaded_svc = pickle.load(open("SVC.sav", 'rb'))
    return loaded_svc


def prepare_test_sample(superpixels, surf, gabor):
    test_sample = []
    for i in range(len(superpixels)):
        features = []

        surf_of_sp = surf[i]
        surf_of_sp = np.average(surf_of_sp)
        surf_of_sp = np.nan_to_num(surf_of_sp)

        gabor_of_sp = gabor[i]
        gabor_of_sp = np.average(gabor_of_sp)
        gabor_of_sp = np.nan_to_num(gabor_of_sp)

        features.append(surf_of_sp)
        features.append(gabor_of_sp)
        test_sample.append(features)
    return test_sample
