# Ex1_3 Brightness/Darkness Adjustment

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test image (as float in range [0,1])
img = cv2.imread('rose.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB
img = img.astype(np.float64) / 255.0

# Brightness adjustment by intensity scaling
scale = 1.4   # scale=1.0: unchanged; <1.0: darker; >1.0: brighter
scaled_img = np.clip(img * scale, 0, 1)  # clip to [0,1] to avoid overflow

# Display images
FS = 15
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image", fontsize=FS)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(scaled_img)
plt.title("Brighter Image", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Brighter.jpg")
plt.show()
