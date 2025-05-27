import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi, threshold_otsu, gaussian
from skimage.color import rgb2gray
import skimage.morphology as mp
from skimage import exposure

def detect_vessels(img, image_fov):
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()  # spłaszczenie, żeby łatwiej indeksować

    # 1. Normalizacja histogramu tylko zielonego kanału
    colorimage_b = img[:, :, 0]
    colorimage_g = cv2.equalizeHist(img[:, :, 1])
    colorimage_r = img[:, :, 2]
    img = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)

    img = colorimage_g
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Po equalizacji zielonego")
    axs[0].axis("off")

    # 2. Konwersja BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Usuwanie szumów (gaussian)
    img = gaussian(img, sigma=10, channel_axis=-1)
    axs[1].imshow(img)
    axs[1].set_title("Po filtrze Gaussa")
    axs[1].axis("off")

    # 4. Skala szarości
    img = rgb2gray(img)
    axs[2].imshow(img, cmap='gray')
    axs[2].set_title("Skala szarości")
    axs[2].axis("off")

    # 5. Filtr Frangi
    img = frangi(img)
    axs[3].imshow(img, cmap='gray')
    axs[3].set_title("Filtr Frangi")
    axs[3].axis("off")

    # 6. Morfologiczna dylatacja i erozja
    img = mp.dilation(img, mp.disk(4))
    img = mp.erosion(img, mp.disk(4))
    axs[4].imshow(img, cmap='gray')
    axs[4].set_title("Morfologia (dylacja+erozja)")
    axs[4].axis("off")

    # 7. Progowanie Otsu
    thresh = threshold_otsu(img)
    binary_img = img > thresh * 0.1
    axs[5].imshow(binary_img, cmap='gray')
    axs[5].set_title("Po progowaniu Otsu")
    axs[5].axis("off")

    # 8. Nakładanie maski FOV (ograniczenie do obszaru siatkówki)
    final_masked = np.copy(binary_img)
    final_masked[image_fov == 0] = 0
    axs[6].imshow(image_fov, cmap='gray')
    axs[6].set_title("Maska FOV")
    axs[6].axis("off")

    # 9. Obraz końcowy
    axs[7].imshow(final_masked, cmap='gray')
    axs[7].set_title("Obraz końcowy")
    axs[7].axis("off")

    plt.tight_layout()
    plt.show()

    return final_masked


image_name = 'images/01_h.jpg'
image_fov = 'images_mask/01_h_mask.tif'

image = cv2.imread(image_name)
fov = cv2.imread(image_fov, cv2.IMREAD_GRAYSCALE)

result = detect_vessels(image, fov)