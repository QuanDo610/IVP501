# Ex2_4 Dividing two images

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load test images (grayscale để dễ xử lý giống MATLAB)
notext = cv2.imread('gradient.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)
withtext = cv2.imread('gradient_with_text.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)

# Division (avoid division by zero)
epsilon = 1e-8
D = withtext / (notext + epsilon)

# Thresholding giống như MATLAB (D > 1)
D = (D > 1).astype(np.uint8) * 255   # chuyển về ảnh nhị phân (0 hoặc 255)

# Plot images
FS = 15
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(notext, cmap='gray')
plt.title("Without Text", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(withtext, cmap='gray')
plt.title("With Text", fontsize=FS)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(D, cmap='gray')
plt.title("Detected Text", fontsize=FS)
plt.axis("off")

plt.tight_layout()
plt.savefig("Image_Division.jpg")
plt.show()
