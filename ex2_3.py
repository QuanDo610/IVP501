# Ex2_3 Multiplying two or more images

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test image (normalize to [0,1])
img = cv2.imread('earth1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float64) / 255.0

# Multiply images (similar to immultiply)
img_3D = img * img * img * img      # equivalent to immultiply(immultiply(...))
img_3d = np.power(img, 8)           # equivalent to Img .^ 8

# Plot images
FS = 15
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("2-D Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_3D)
plt.title("3-D Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_3d)
plt.title("3-d Image", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Image_Multiplication.jpg")
plt.show()
