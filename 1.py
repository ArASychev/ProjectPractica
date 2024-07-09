import matplotlib.pyplot as plt 
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
# Загрузка изображения 1.jpeg
image = cv2.imread('/home/user/OS/lab1/PROJECT/1.jpeg')
# Отображение оригинального изображения
plt.subplot(1, 2, 1) 
plt.title("Original picture") 
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
contrast = 2 # Уровень контраста 
brightness = 5 # Уровень яркости
image2 = cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness) 
cv2.imwrite('bright_contrast_image.jpg', image2) 
# Отображение измененного изображения
plt.subplot(1, 2, 2) 
plt.title("Contrast & Brightness adjusted picture") 
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)) 
# Рассчет и вывод значения SSIM
ssim_value = ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
print("Значение SSIM между оригинальным и улучшенным изображениями:", ssim_value)
plt.show()

