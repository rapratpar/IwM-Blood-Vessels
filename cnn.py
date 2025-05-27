import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(b)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c3 = layers.Conv2D(32, 3, activation='relu', padding='same')(c3)

    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)
    c4 = layers.Conv2D(16, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)
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

img_patches = []
mask_patches = []
for i in range(1, 5):
    base_img = "images/0" + str(i) + "_h.jpg"
    base_manual = "images_manual/0" + str(i) + "_h.tif"
    base_mask = "images_mask/0" + str(i) + "_h_mask.tif"
    image = cv2.imread(base_img)
    manual = cv2.imread(base_manual)
    mask = cv2.imread(base_mask)
    manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)

    ip, mp = extract_patches(image, manual)
    img_patches.extend(ip);
    mask_patches.extend(mp);

img_patches = np.array(img_patches)
mask_patches = np.array(mask_patches)
print(img_patches.shape)

# Normalizacja PRZED podziałem
img_patches = img_patches.astype(np.float32) / 255.0
mask_patches = mask_patches.astype(np.float32) / 255.0

# Dodaj kanał do masek, jeśli go nie mają
if mask_patches.ndim == 3:
    mask_patches = np.expand_dims(mask_patches, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(img_patches, mask_patches, test_size=0.2, random_state=0)
input_shape = X_train.shape[1:]

print(mask_patches[40])
model = build_unet(input_shape);
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1  # 10% z treningu jako walidacja
)

# Ocena modelu na danych testowych
#test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
#print(f"\n✅ Test Loss: {test_loss:.4f}")
#print(f"✅ Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import precision_score, recall_score

# Predykcje modelu na zbiorze testowym
y_pred = model.predict(X_test)
# Zamień predykcje na binarne maski (0 lub 1) za pomocą threshold, np. 0.5
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

# Jeśli y_test i y_pred mają wymiar (samples, height, width, 1)
# spłaszczamy je do 1D wektorów
y_true_flat = y_test.flatten()
y_pred_flat = y_pred_bin.flatten()

precision = precision_score(y_true_flat, y_pred_flat)
recall = recall_score(y_true_flat, y_pred_flat)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

model.save("unet_model.h5")
