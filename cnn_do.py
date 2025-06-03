import cv2
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt


def reconstruct_from_patches(pred_patches, image_shape, patch_size=64, stride=32):
    h, w = image_shape
    recon_image = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    patch_idx = 0
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            recon_image[i:i+patch_size, j:j+patch_size] += pred_patches[patch_idx, :, :, 0]
            weight[i:i+patch_size, j:j+patch_size] += 1.0
            patch_idx += 1

    # uniknamy dzielenia przez zero
    weight[weight == 0] = 1.0
    recon_image /= weight
    return recon_image


def extract_patches(img, mask, patch_size=64, stride=32):
    img_patches = []
    mask_patches = []
    h, w, _ = img.shape
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            img_patch = img[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]
            img_patches.append(img_patch)
            mask_patches.append(mask_patch)
    img_patches = np.array(img_patches)
    mask_patches = np.array(mask_patches)
    return img_patches, mask_patches


def predict_and_save(image, manual, output_filename="cnn_prediction.jpg"):
    img_patches, mask_patches = extract_patches(image, manual)
    
    img_patches = img_patches.astype(np.float32) / 255.0
    mask_patches = mask_patches.astype(np.float32) / 255.0
    
    if mask_patches.ndim == 3:
        mask_patches = np.expand_dims(mask_patches, axis=-1)
    
    model = tf.keras.models.load_model("unet_model.h5")
    y_pred = model.predict(img_patches)
    
    image_shape = image.shape[:2]
    full_mask = reconstruct_from_patches(y_pred, image_shape, patch_size=64, stride=32)
    
    binary_mask = (full_mask > 0.5).astype(np.uint8)
    
    binary_mask_255 = binary_mask * 255 
    cv2.imwrite(output_filename, binary_mask_255)
    print(f"CNN zapisane jako {output_filename}")
    
    # Wyświetlanie
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Maska ekspercka (manual)")
    plt.imshow(manual, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Maska predykcji")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return binary_mask, output_filename

if __name__ == "__main__":
    base_img = "images/10_h.jpg"
    base_manual = "images_manual/10_h.tif"
    base_mask = "images_mask/10_h_mask.tif"
    image = cv2.imread(base_img)
    manual = cv2.imread(base_manual)
    mask = cv2.imread(base_mask)
    manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
    
    predict_and_save(image, manual, "cnn_.jpg")