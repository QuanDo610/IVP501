import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread('moon.jpg', cv2.IMREAD_GRAYSCALE)

# Compute histogram
count, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])
maxCount = np.max(count)

# Define clipping ratios
clip_ratios = [1, 0.7, 0.4, 0.05]
limited_eq_imgs = []
LUTs = []

for ratio in clip_ratios:
    clip = ratio * maxCount
    # Clip histogram
    clipped_count = np.minimum(count, clip).astype(int)

    # Construct virtual image
    clipped_img = []
    for level in range(256):
        clipped_img.extend([level] * clipped_count[level])
    clipped_img = np.array(clipped_img, dtype=np.uint8)

    # Mapping function via histogram equalization
    hist, bins_eq = np.histogram(clipped_img, bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # normalize
    T = np.round(cdf * 255).astype(np.uint8)

    # Apply mapping to original image
    LUTs.append(T)
    limited_eq_img = cv2.LUT(img, T)
    limited_eq_imgs.append(limited_eq_img)

# Show images
plt.figure(figsize=(12, 4))
for i, ratio in enumerate(clip_ratios):
    plt.subplot(1, len(clip_ratios), i+1)
    plt.imshow(limited_eq_imgs[i], cmap='gray')
    plt.title(f'Clip at {ratio} max', fontsize=12)
    plt.axis('off')
plt.savefig('CLAHE.jpg')

# Show histogram and LUTs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(np.arange(256), count, width=1)
plt.xlim([0, 255])
plt.ylim([0, np.max(count) + 500])
plt.grid(True)
plt.title("Original Histogram")

plt.subplot(1, 2, 2)
for i in range(len(clip_ratios)):
    plt.plot(LUTs[i], label=f'Clip {clip_ratios[i]}')
plt.xlim([0, 255])
plt.ylim([0, 255])
plt.legend()
plt.grid(True)
plt.title("Mapping Functions (LUTs)")

plt.savefig('LUTs.jpg')
plt.show()
