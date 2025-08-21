# Ex4_2 Global Histogram Equalization
import cv2
import numpy as np
import matplotlib.pyplot as plt

FS = 15  # fontsize

# Load image (grayscale)
# Img = cv2.imread('tree.jfif', cv2.IMREAD_GRAYSCALE)
# Img = cv2.imread('bay.jpg', cv2.IMREAD_GRAYSCALE)
# Img = cv2.imread('brain.jpg', cv2.IMREAD_GRAYSCALE)
Img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 8))

# a. Original image and its histogram
count, bins = np.histogram(Img.flatten(), bins=256, range=[0, 256])

plt.subplot(2, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.bar(bins[:-1], count, width=1.0, color='black')
plt.title('Original Histogram', fontsize=FS)
plt.xlabel('Gray Level', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, count.max() + 500])

# b. Histogram equalization
eq_Img = cv2.equalizeHist(Img)  # OpenCV function

count_eq, bins_eq = np.histogram(eq_Img.flatten(), bins=256, range=[0, 256])

plt.subplot(2, 2, 3)
plt.imshow(eq_Img, cmap='gray')
plt.title('Equalized Image', fontsize=FS)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.bar(bins_eq[:-1], count_eq, width=1.0, color='black')
plt.title('Equalized Histogram', fontsize=FS)
plt.xlabel('Gray Level', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, count_eq.max() + 500])

plt.tight_layout()
plt.savefig("Histogram_Equalization.jpg", dpi=300)
plt.show()
