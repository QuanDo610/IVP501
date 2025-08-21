# Ex4_4 Local Histogram Equalization
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, util

FS = 15  # fontsize

# Load ảnh và chuyển sang grayscale
# Img = cv2.imread('dental.jpg', cv2.IMREAD_GRAYSCALE)
Img = cv2.imread('newspaper.jpg', cv2.IMREAD_GRAYSCALE)

# Global histogram equalization
eq_Img = cv2.equalizeHist(Img)

# Local histogram equalization (chia block và apply histeq)
def local_histeq(img, block_size=(49, 294)):
    h, w = img.shape
    bh, bw = block_size
    out = np.zeros_like(img)

    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = img[i:i+bh, j:j+bw]
            eq_block = cv2.equalizeHist(block)
            out[i:i+bh, j:j+bw] = eq_block
    return out

lc_Img = local_histeq(Img)

# Hiển thị
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(eq_Img, cmap='gray')
plt.title('Global', fontsize=FS)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(lc_Img, cmap='gray')
plt.title('Local', fontsize=FS)
plt.axis('off')

plt.tight_layout()
plt.savefig('Global_vs_Local_Eq.jpg', dpi=300)
plt.show()
