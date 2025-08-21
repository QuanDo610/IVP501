# Ex3_3 Noise Reduction Using Image Averaging
import cv2
import numpy as np
import matplotlib.pyplot as plt

# đọc ảnh và chuyển về dạng double [0,1]
Img = cv2.imread('quadnight.jfif')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
Img = Img.astype(np.float32) / 255.0

# số lượng ảnh cần lặp
nImages = [1, 2, 5, 10, 50, 100]
FS = 15  # fontsize

plt.figure(figsize=(12, 6))

for i, N in enumerate(nImages):
    avg_Img = np.zeros_like(Img, dtype=np.float32)

    # cộng dồn ảnh có nhiễu Gaussian
    for j in range(N):
        noise = np.random.normal(0, np.sqrt(0.5), Img.shape)  # mean=0, var=0.5
        noisy_Img = Img + noise
        noisy_Img = np.clip(noisy_Img, 0, 1)  # giữ giá trị trong [0,1]
        avg_Img += noisy_Img

    avg_Img = avg_Img / N  # tính trung bình

    plt.subplot(2, len(nImages)//2, i+1)
    plt.imshow(avg_Img)
    plt.axis('off')
    plt.title(f"{N} image(s)", fontsize=FS)

plt.tight_layout()
plt.savefig("Image_Averaging.png")
plt.show()
