import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the project directory to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a simple GUI with three buttons
def main():
    # Create the main window
    root = tk.Tk()
    root.title("Wykrywanie naczyń dna siatkówki oka")
    root.geometry("500x300")
    
    # Create a frame for buttons
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    # Create buttons
    btn_detect = tk.Button(
        frame, 
        text="Algorytm", 
        command=run_xd_detection,
        width=25,
        height=2
    )
    btn_detect.pack(pady=10)
    
    btn_cnn = tk.Button(
        frame, 
        text="CNN", 
        command=run_cnn_do,
        width=25,
        height=2
    )
    btn_cnn.pack(pady=10)
    
    btn_predict = tk.Button(
        frame, 
        text="KNN", 
        command=run_main_prediction,
        width=25,
        height=2
    )
    btn_predict.pack(pady=10)
    
    # Start the main loop
    root.mainloop()

def run_xd_detection():
    """Run the vessel detection from xd.py"""
    try:
        # Use the default files as in xd.py
        image_name = 'images/01_h.jpg'
        image_fov = 'images_mask/01_h_mask.tif'

        # Run the function from xd module
        import cv2
        from xd import detect_vessels
        
        image = cv2.imread(image_name)
        fov = cv2.imread(image_fov, cv2.IMREAD_GRAYSCALE)
        
        if image is None or fov is None:
            messagebox.showerror("Error", f"Failed to read image or mask file: {image_name}, {image_fov}")
            return
        
        detect_vessels(image, fov)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def run_cnn_do():
    """Run the CNN model from cnn_do.py"""
    try:
        # Use the default files as in cnn_do.py
        base_img = "images/10_h.jpg"
        base_manual = "images_manual/10_h.tif"
        base_mask = "images_mask/10_h_mask.tif"
        
        # Run the functions from cnn_do module
        import cv2
        import numpy as np
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from cnn_do import extract_patches, reconstruct_from_patches
        
        image = cv2.imread(base_img)
        manual = cv2.imread(base_manual)
        mask = cv2.imread(base_mask)
        
        if image is None or manual is None or mask is None:
            messagebox.showerror("Error", f"Failed to read image, manual or mask: {base_img}, {base_manual}, {base_mask}")
            return
            
        manual = cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
        
        # Process with the CNN
        img_patches, mask_patches = extract_patches(image, manual)
        img_patches = img_patches.astype('float32') / 255.0
        mask_patches = mask_patches.astype('float32') / 255.0
        
        if mask_patches.ndim == 3:
            mask_patches = np.expand_dims(mask_patches, axis=-1)
            
        model = tf.keras.models.load_model("unet_model.h5")
        y_pred = model.predict(img_patches)
        
        image_shape = image.shape[:2]
        full_mask = reconstruct_from_patches(y_pred, image_shape)
        binary_mask = (full_mask > 0.5).astype('uint8')
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Expert Mask")
        plt.imshow(manual, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def run_main_prediction():
    """Run the prediction from main.py"""
    try:
        # Load the model and predict directly as in main.py
        from main import load_model, predict_img
        
        knn_classifier = load_model()
        if knn_classifier is None:
            messagebox.showerror("Error", "Failed to load KNN model")
            return
            
        predict_img(knn_classifier)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()