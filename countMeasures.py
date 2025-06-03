import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def calculate_metrics(prediction_path, manual_path, threshold=128):
    """
    Calculate accuracy, sensitivity (recall) and specificity for vessel detection
    
    Args:
        prediction_path: Path to the predicted vessel mask image
        manual_path: Path to the ground truth manual segmentation
        threshold: Threshold for binarization (default 128)
        
    Returns:
        Dictionary with accuracy, sensitivity and specificity values
    """
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    manual = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
    
    if prediction is None:
        raise FileNotFoundError(f"Could not load prediction image: {prediction_path}")
    if manual is None:
        raise FileNotFoundError(f"Could not load manual image: {manual_path}")
    
    if prediction.shape != manual.shape:
        prediction = cv2.resize(prediction, (manual.shape[1], manual.shape[0]))
    
    pred_bin = prediction > threshold
    manual_bin = manual > threshold
    
    y_true = manual_bin.flatten()
    y_pred = pred_bin.flatten()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Metrics for {os.path.basename(prediction_path)} vs {os.path.basename(manual_path)}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def visualize_comparison(prediction_path, manual_path, threshold=128):
    """
    Visualize the comparison between prediction and ground truth
    
    Args:
        prediction_path: Path to the predicted vessel mask image
        manual_path: Path to the ground truth manual segmentation
        threshold: Threshold for binarization (default 128)
    """
    prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
    manual = cv2.imread(manual_path, cv2.IMREAD_GRAYSCALE)
    
    if prediction is None or manual is None:
        print("Error loading images")
        return
    
    if prediction.shape != manual.shape:
        prediction = cv2.resize(prediction, (manual.shape[1], manual.shape[0]))
    
    pred_bin = prediction > threshold
    manual_bin = manual > threshold
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Ground Truth")
    plt.imshow(manual_bin, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(pred_bin, cmap='gray')
    plt.axis('off')
    
    comparison = np.zeros((manual.shape[0], manual.shape[1], 3), dtype=np.uint8)
    
    comparison[np.logical_and(manual_bin, pred_bin), 1] = 255
    
    comparison[np.logical_and(np.logical_not(manual_bin), pred_bin), 0] = 255
    
    comparison[np.logical_and(manual_bin, np.logical_not(pred_bin)), 2] = 255
    
    plt.subplot(1, 3, 3)
    plt.title("Comparison (Green:TP, Red:FP, Blue:FN)")
    plt.imshow(comparison)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    prediction_file = "xd.jpg"
    manual_file = "images_manual/01_h.tif"
    
    metrics = calculate_metrics(prediction_file, manual_file)
    
    visualize_comparison(prediction_file, manual_file)
    