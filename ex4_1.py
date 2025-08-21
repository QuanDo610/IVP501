# Ex4_1 Histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale)
# Img = cv2.imread('tree.jfif', cv2.IMREAD_GRAYSCALE)
# Img = cv2.imread('bay.jpg', cv2.IMREAD_GRAYSCALE)
Img = cv2.imread('brain.jpg', cv2.IMREAD_GRAYSCALE)

# Compute histogram
count, bins = np.histogram(Img.flatten(), bins=256, range=[0, 256])

# Plot
FS = 15  # fontsize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image', fontsize=FS)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(bins[:-1], count, width=1.0, color='black')
plt.title('Histogram', fontsize=FS)
plt.xlabel('Gray Level', fontsize=FS)
plt.ylabel('# of pixels', fontsize=FS)
plt.grid(True)
plt.xlim([0, 255])
plt.ylim([0, count.max() + 500])

plt.tight_layout()
plt.savefig("Histogram.jpg", dpi=300)
plt.show()
