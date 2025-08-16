# Ex2_2 Subtracting two images

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test images as float in range [0,1]
live = cv2.imread('live.jpg')
live = cv2.cvtColor(live, cv2.COLOR_BGR2RGB)
live = live.astype(np.float64) / 255.0

mask = cv2.imread('mask.jpg')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
mask = mask.astype(np.float64) / 255.0

# Absolute difference and gamma adjustment (power)
diff_img = np.abs(live - mask) ** 1.4

# Tăng độ sáng và tương phản
# Cách 1: Điều chỉnh gamma để làm sáng (giảm gamma)
diff_img = diff_img ** 0.6  # Gamma < 1 để làm sáng

# Cách 2: Normalize và scale để tăng tương phản
diff_img = (diff_img - diff_img.min()) / (diff_img.max() - diff_img.min())
diff_img = diff_img * 1.2  # Tăng độ sáng thêm 20%
diff_img = np.clip(diff_img, 0, 1)  # Giới hạn trong [0,1]

# Plot images
FS = 15
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(live)
plt.title("live", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask)
plt.title("mask", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(diff_img)
plt.title("Subtracting", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Image_Subtraction.jpg")
plt.show()
