# Ex3_1 Flipping and Rotating Images

import cv2
import numpy as np
import matplotlib.pyplot as plt

FS = 15  # fontsize

# ==========================================================
# a. Flipping an image up-down and left-right
# ==========================================================
img = cv2.imread('atrium.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_ud = cv2.flip(img, 0)   # flip up-down
img_lr = cv2.flip(img, 1)   # flip left-right

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_ud)
plt.title("Flipped Up-Down", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_lr)
plt.title("Flipped Left-Right", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Flipping_Image.png")
plt.show()


# ==========================================================
# b. Rotating an image with an angle theta (counter-clockwise)
# ==========================================================
img2 = cv2.imread('eight.png', cv2.IMREAD_GRAYSCALE)

# Rotation with OpenCV: cv2.getRotationMatrix2D + warpAffine
(h, w) = img2.shape[:2]
center = (w // 2, h // 2)

theta1 = 30
M1 = cv2.getRotationMatrix2D(center, theta1, 1.0)
img_rot1 = cv2.warpAffine(img2, M1, (w, h))

theta2 = 90
M2 = cv2.getRotationMatrix2D(center, theta2, 1.0)
img_rot2 = cv2.warpAffine(img2, M2, (w, h))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img2, cmap="gray")
plt.title("Original Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_rot1, cmap="gray")
plt.title("Rotated 30-deg", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_rot2, cmap="gray")
plt.title("Rotated 90-deg", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Rotating_Image.png")
plt.show()
