# Ex1_1 Read and display an image with different quality

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (color by default in OpenCV is BGR)
img = cv2.imread('nature.jpg')

# Convert to grayscale
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to binary image (threshold at 0.5)
_, img_bw = cv2.threshold(img_grayscale, 127, 255, cv2.THRESH_BINARY)

# Display images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # convert BGR->RGB for matplotlib
plt.title("Color Image", fontsize=15)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_grayscale, cmap='gray')
plt.title("Grayscale Image", fontsize=15)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_bw, cmap='gray')
plt.title("Binary Image", fontsize=15)
plt.axis("off")

plt.tight_layout()
plt.savefig("Image_in_Different_Types.jpg")
plt.show()

# Save images with different quality
cv2.imwrite("nature100.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # default
cv2.imwrite("nature70.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])    # 70% quality
cv2.imwrite("nature10.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])    # 10% quality
