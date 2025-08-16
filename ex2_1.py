# Ex2_1 Adding two images

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test images (OpenCV đọc theo BGR)
img1 = cv2.imread('prarie.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('giraffe.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Resize img2 để bằng kích thước img1
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Image addition (saturation arithmetic giống imadd của MATLAB)
img3 = cv2.add(img1, img2_resized)

# Display images
FS = 15
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img1)
plt.title("prarie", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img2)
plt.title("giraffe", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img3)
plt.title("Addition", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Image_Addition.jpg")
plt.show()
