from sklearn.cluster import KMeans
import numpy
import pickle
import os


# Fit kmeans and save it with pickle library.
def fit_and_save_kmeans(lab_images):
    if os.path.exists("fitted_kmeans.sav"):
        print("The fitted KMeans was already saved, if you want to repeat the process you need to delete the file and "
              "restart.")
        return
    print("[!] Fitting KMeans [!]")
    kmeans = KMeans(n_clusters=16)
    # Reshaping lab images as feature vector
    ab_values = []
    for lab_image in lab_images:
        lab_image = lab_image.reshape((lab_image.shape[0] * lab_image.shape[1]), 3)
        lab_image = numpy.delete(lab_image, 0, 1)
        ab_values.extend(lab_image)
    ab_values = numpy.array(ab_values)
    fitted = kmeans.fit(ab_values)

    print("Fitting completed!")
    pickle.dump(fitted, open("fitted_kmeans.sav", 'wb'))
    print("KMeans saved!")


def load_kmeans():
    loaded_kmeans = pickle.load(open("fitted_kmeans.sav", 'rb'))
    return loaded_kmeans
