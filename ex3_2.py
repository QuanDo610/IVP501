# Ex3_2 Cropping and Zooming Images
import cv2
import matplotlib.pyplot as plt

FS = 15  # fontsize of caption

# ---------- a. Cropping an image ----------
# Img = cv2.imread('mandrill.tif')
# Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # chuyển sang RGB để hiển thị đúng màu

# x1, x2, y1, y2 = 150, 450, 50, 550
# Img_cropped = Img[y1:y2, x1:x2]

# plt.figure(1)
# plt.subplot(1,2,1), plt.imshow(Img), plt.title("Original Image", fontsize=FS)
# plt.subplot(1,2,2), plt.imshow(Img_cropped), plt.title("Cropped Image", fontsize=FS)
# plt.savefig("Cropped_Image.png")

# ---------- b. Zooming with different interpolation methods ----------
Img = cv2.imread('bird.jpg')
Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)

# resize with different methods
Img_z1 = cv2.resize(Img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)   # bicubic
Img_z2 = cv2.resize(Img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST) # nearest
Img_z3 = cv2.resize(Img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # bilinear

plt.figure(2, figsize=(10, 10))
plt.subplot(2,2,1), plt.imshow(Img), plt.title("Original Image", fontsize=FS)
plt.subplot(2,2,2), plt.imshow(Img_z1), plt.title("Bicubic", fontsize=FS)
plt.subplot(2,2,3), plt.imshow(Img_z2), plt.title("Nearest", fontsize=FS)
plt.subplot(2,2,4), plt.imshow(Img_z3), plt.title("Bilinear", fontsize=FS)

plt.savefig("Zooming_with_different_interpolation_methods.png")
plt.show()
