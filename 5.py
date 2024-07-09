import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the image
image = cv2.imread('/home/user/OS/lab1/PROJECT/1.jpeg')
# Plot the original image
plt.subplot(1, 3 ,1)
plt.title("Original picture")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
# Resize the image to a specific width and height
resized_image = cv2.resize(image, (2100, 1500))
# Save the resized image
cv2.imwrite('Resized_image.jpg', resized_image)
# Resize the larger image to the size of the smaller image
image_resized = cv2.resize(image, (resized_image.shape[1], resized_image.shape[0]))
# Compute the SSIM metric
image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
resized_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
ssim_value = ssim(image_gray, resized_gray)
# Print the SSIM value
print(f"Значение SSIM между исходным и измененным изображениями: {ssim_value}")
# Plot the resized image
plt.subplot(1, 3, 2)
plt.title("Resized picture")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
# Show the images
plt.show()

