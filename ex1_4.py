# Ex1_4 Contrast Enhancement

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test image (as float in range [0,1])
img = cv2.imread('waterfall.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV đọc ảnh theo BGR -> chuyển RGB
img = img.astype(np.float64) / 255.0        # chuẩn hóa về [0,1]

# Contrast enhancement by gamma correction
gamma = 1.8   # gamma=1.0: unchanged; <1.0: decrease; >1.0: increase
enhanced_img = np.power(img, gamma)

# Display images
FS = 15
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_img)
plt.title("Increase Contrast", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Increase_Contrast.jpg")
plt.show()
