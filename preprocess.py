import cv2
import numpy as np
import imutils

# Crop
def cropping(img):

    # 1. Blur
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # 2. Threshold and erosion + dilation to remove small noise
    _, th = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    th = cv2.erode(th, None, iterations=2)
    th = cv2.dilate(th, None, iterations=2)

    # 3. Find contour
    contours = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return img

    c = max(contours, key=cv2.contourArea)

    # 4. Find extreme points
    exLeft = tuple(c[c[:,:,0].argmin()][0])
    exRight = tuple(c[c[:,:,0].argmax()][0])
    exTop = tuple(c[c[:,:,1].argmin()][0])
    exBot = tuple(c[c[:,:,1].argmax()][0])

    cropped = img[exTop[1]:exBot[1], exLeft[0]:exRight[0]]

    return cropped

# Padding
def make_square(img):
    h, w = img.shape
    size = max(h, w)

    # Buat canvas square
    square = np.zeros((size, size), dtype=img.dtype)

    # Hitung offset biar center
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return square


# Preprocessing
def preprocess_image(img_path, return_steps=False):
    steps = {}
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    steps["Original"] = img

    # Cropping
    img_crop = cropping(img)
    steps["ROI Extraction"] = img_crop

    # Padding
    img_pad = make_square(img_crop)
    steps["Padding"] = img_pad

    # Denoise
    img_denoise = cv2.bilateralFilter(img_pad, 2, 20, 20)
    steps["Noise Reduction"] = img_denoise

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_denoise)
    steps["Contrast Enhancement"] = img_clahe

    # Resize
    img_resize = cv2.resize(img_clahe, (224, 224))
    steps["Resized"] = img_resize

    # Normalize
    img_final = img_resize / 255.0

    # Add channel dim
    img_final = np.expand_dims(img_final, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    if return_steps:
        return img_final, steps
    else:
        return img_final