import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import filters
from skimage.filters import threshold_otsu, frangi, gaussian
from skimage.morphology import dilation, erosion, disk
from skimage.util import img_as_float, view_as_windows
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load


SCALE_FACTOR = 0.8
MODEL_NAME = "full_knn_3.pkl"
PATCH_SIZE = 5

def preprocess_image(img, mask):
    # Wyciągamy zielony kanał i normalizujemy
    green_channel = img[:, :, 1]
    img_eq = cv2.equalizeHist(green_channel)  # to już jest obraz w skali szarości (1 kanał)

    # Wyostrzenie
    sharpened_image = filters.unsharp_mask(img_eq)
    clear_back(img_eq, mask)  # lub sharpened_image, jeśli chcesz maskować już wyostrzony


    return sharpened_image
def clear_back(img, mask, th = 100):
    '''

    :param img: obrazek
    :param mask: obrazek dzielący na tło i oko
    :param th: treshold, do czyszczenia szumu
    :return: wyczyszczony obrazek
    '''
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask > th
    img[mask == 0] = 0
    return img


def get_all_features(patches, gery_patches):
    '''

    :param patches: fragmenty obrazka
    :param gery_patches: fragemnty przetworzonego obrazka
    :return: cechy dla wszystkich fragmentów
    '''
    all_features = []
    for i in range(len(patches)):
        all_features.extend(get_feature(patches[i], gery_patches[i]))
    #print(all_features)
    return all_features



def get_feature(patch,grey_patch):
    '''

    :param patch: fragemnty obrazka (do cech kolorów)
    :param grey_patch: fragemnty przetworoznego obrazka (do momentów)
    :return: cechy jednego patcha
    '''
    features = []

    pixels = patch.reshape(-1, 3)
    std = np.std(pixels, axis=0)
    mean = np.mean(pixels, axis=0)

    moments = cv2.moments(grey_patch)
    hu_moments = cv2.HuMoments(moments).flatten()

    features.append(np.hstack([mean, std, hu_moments]))
    #print(features)
    return features

def create_patches_and_labels(image, manual, mask, patch_size=PATCH_SIZE):
    '''

    :param image: Przetwarzany obrazek
    :param manual: Obrazek z wynkami, przez człowieka
    :param mask: Obrazek któy dzieli orginalny na tło i oko
    :param patch_size: wielkość jedngo fragemntu zdjęcia
    :return: -fragemnty oryginalnego obrazka z kolorami, patche z przetworzonym obrazkiem,
    fragemnty manula, środkowy piksel manula (lable)
    '''

    #Tworzymy szary, przetworzony obrazek do modelu
    grey_image = preprocess_image(image,mask)
    h, w, _ = image.shape
    center = patch_size // 2

    patches = []
    grey_patches = []
    manual_patches = []
    labels = []


    if len(manual.shape) == 3:
        manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)

    for x in range(h - patch_size + 1):
        for y in range(w - patch_size + 1):
            patch = image[x:x+patch_size, y:y+patch_size]
            grey_patch = grey_image[x:x+patch_size, y:y+patch_size]
            manual_patch = manual[x:x+patch_size, y:y+patch_size]
            label = manual[x + center, y + center]

            patches.append(patch)
            manual_patches.append(manual_patch)
            labels.append(label)
            grey_patches.append(grey_patch)

    #print(labels[30])
    #print(manual_patches[30])

    return patches,grey_patches,manual_patches,labels





def create_mode():
    features = []
    img_path = ""
    manual_path = ""
    all_labels = []
    for i in range(1,9):
        base_img = "images/0" + str(i) + "_h.jpg"
        base_manual = "images_manual/0" + str(i) + "_h.tif"
        base_mask = "images_mask/0" + str(i) + "_h_mask.tif"
        image = cv2.imread(base_img)
        manual = cv2.imread(base_manual)
        mask = cv2.imread(base_mask)
        # print("XD")

        # Skalowanie obrazów
        image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        manual = cv2.resize(manual, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        mask = cv2.resize(mask, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

        #manual = clear_back(manual,mask,10)
        manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
        manual = manual > 10
        manual = clear_back(manual, mask, 100)
        #manual[mask == 0] = 0

        patches, grey_patches, manual_patches, labels = create_patches_and_labels(image, manual, mask)

        features.extend(get_all_features(patches, grey_patches))
        all_labels.extend(labels)
        print("one done")
    print("ALl done")

    features = np.array(features)
    all_labels= np.array(all_labels)

    rus = RandomUnderSampler(random_state=0)
    features, all_labels = rus.fit_resample(features, all_labels)
   # features_reshaped = features.reshape((features.shape[0], -1))
    #features_resampled, labels_resampled = rus.fit_resample(features_reshaped, all_labels)
#
    X_train, X_test, y_train, y_test = train_test_split(features, all_labels, test_size=0.3, random_state=0)

    print("Classifier")

    classifier = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        p=1
    )

    print("Fitting")
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)

    #print("Best parameters found:", classifier.best_params_)
    print("Accuracy on test set:", accuracy)

    dump(classifier, MODEL_NAME)

image = cv2.imread("images/01_h.jpg")
manual = cv2.imread("images_manual/01_h.tif")


def load_model():
    return load(MODEL_NAME)



def make_img_from_labels(img_size,labels, patch_size = PATCH_SIZE):
    x,y = img_size
    new_img = np.zeros((x,y));



    for i in range(x - patch_size + 1):
        for j in range(y - patch_size + 1):
            print("XD")

    return

def prediction_to_mask(predictions, image_shape, patch_size=PATCH_SIZE):
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    offset = patch_size // 2  # środkowy piksel w patchu
    idx = 0

    for y in range(0, height - patch_size + 1):
        for x in range(0, width - patch_size + 1):
            # wstawiamy wynik w środkowy piksel patcha
            mask[y + offset, x + offset] = 255 if predictions[idx] else 0
            idx += 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.title("Maska z predykcji")
    plt.axis('off')
    plt.show()

    # zapis maski jako jpg
    output_filename = "knn.jpg"
    cv2.imwrite(output_filename, mask)
    print(f"Zapisano wynik KNN na {output_filename}")

    return mask
def predict_img(knn_clasifier):
    base_img = "images/10_h.jpg"
    base_manual = "images_manual/10_h.tif"
    base_mask = "images_mask/10_h_mask.tif"
    image = cv2.imread(base_img)
    manual = cv2.imread(base_manual)
    mask = cv2.imread(base_mask)

    # Skalowanie
    image = cv2.resize(image, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    manual = cv2.resize(manual, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    mask = cv2.resize(mask, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)

    manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
    manual = manual > 10
    manual = clear_back(manual, mask, 100)
    print("zaczaynamt")
    patches, grey_patches, manual_patches, labels = create_patches_and_labels(image, manual, mask)
    print("Koniec patchowania, wyciagamy feature")
    features = get_all_features(patches, grey_patches)
    print("wycignelsimy feature, przestepujemy do predykcji")
    features = np.array(features)

    predictions = knn_clasifier.predict(features)
    print("Koniec predykcji")
    print(predictions)
    print(len(predictions))
    print(np.nonzero(predictions))
    prediction_to_mask(predictions, image.shape[:2])




#create_patches_and_labels(image,manual)
# create_mode()

knn_clasifier = load_model()
print("wczytany")

predict_img(knn_clasifier)


# najpierw create_mode(), jak odpalony to zakomentować go i dać predict_img
# predict_img na guzik





