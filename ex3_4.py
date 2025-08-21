import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale để dễ xử lý mức xám)
Img = cv2.imread('lion.jpg', cv2.IMREAD_GRAYSCALE)

FS = 15  # font size for titles

# Hiển thị ảnh gốc
plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title("Original Image", fontsize=FS)
plt.axis("off")

######################################
# a. Negative Transformation
x = np.arange(255, -1, -1, dtype=np.uint8)  # LUT ngược
Img1 = x[Img]  # áp LUT
plt.subplot(3, 2, 2)
plt.plot(np.arange(256), x, linewidth=1.5)
plt.xlim([0, 255]); plt.ylim([0, 255]); plt.grid(True)
plt.title("Negative Transform Function", fontsize=FS)

plt.subplot(3, 2, 3)
plt.imshow(Img1, cmap='gray')
plt.title("Negative Image", fontsize=FS)
plt.axis("off")

######################################
# b. Logarithmic Transformation
x = np.arange(256, dtype=np.float32)
c = 128 / np.log(256)
y = c * np.log(x + 1)
y = np.uint8(np.clip(y, 0, 255))

Img2 = y[Img]

plt.subplot(3, 2, 4)
plt.plot(np.arange(256), y, linewidth=1.5)
plt.xlim([0, 255]); plt.ylim([0, 255]); plt.grid(True)
plt.title("Logarithmic Transform Function", fontsize=FS)

plt.subplot(3, 2, 5)
plt.imshow(Img2, cmap='gray')
plt.title("Logarithmic Image", fontsize=FS)
plt.axis("off")

######################################
# c. Piece-wise Linear Transformation
LUT = np.zeros(256, dtype=np.uint8)
LUT[0:100] = 2 * np.arange(0, 100)
LUT[100:200] = 150
LUT[200:256] = np.uint8(0.9 * np.arange(200, 256) - 12)

Img3 = LUT[Img]

plt.subplot(3, 2, 6)
plt.plot(np.arange(256), LUT, linewidth=1.5)
plt.xlim([0, 255]); plt.ylim([0, 255]); plt.grid(True)
plt.title("Piece-wise Linear Transform Function", fontsize=FS)

plt.figure(figsize=(5,5))
plt.imshow(Img3, cmap='gray')
plt.title("Piece-Wise Linear Image", fontsize=FS)
plt.axis("off")

######################################
# Save result
plt.savefig("GrayLevelTransformations.png", dpi=300)
plt.show()
