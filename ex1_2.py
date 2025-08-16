# Ex1_2 Sampling and Quantization

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread('tiger.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64)

plt.figure(figsize=(12, 6))
FS = 15  # font size

for NumOfBit in range(1, 9):
    # Quantization
    NumOfLevel = 2 ** NumOfBit
    LevelGap = 256 / NumOfLevel
    quantized_img = np.uint8(np.ceil(img / LevelGap) * LevelGap - 1)

    # Display
    plt.subplot(2, 4, NumOfBit)
    plt.imshow(quantized_img, cmap='gray', vmin=0, vmax=255)
    if NumOfBit == 1:
        name = f"{NumOfBit}-bit"
    else:
        name = f"{NumOfBit}-bits"
    plt.title(name, fontsize=FS)
    plt.axis("off")

plt.tight_layout()
plt.savefig("Image_Quantization.png")
plt.show()
