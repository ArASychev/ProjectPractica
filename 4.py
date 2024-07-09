import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
# Загрузка исходного изображения
image_original = cv2.imread('/home/user/OS/lab1/PROJECT/1.jpeg')
# Отображение исходного изображения
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
# Преобразование изображения из BGR в HSV
image_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
# Регулировка оттенка, насыщенности и значения изображения
image_hsv[:, :, 0] = image_hsv[:, :, 0] * 0.7
image_hsv[:, :, 1] = image_hsv[:, :, 1] * 1.5
image_hsv[:, :, 2] = image_hsv[:, :, 2] * 0.5
# Преобразование изображения обратно в BGR
image_enhanced = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
# Сохранение улучшенного изображения
cv2.imwrite('enhanced_coloured.jpg', image_enhanced)
# Вычисление метрики SSIM
image_original_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
image_enhanced_gray = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2GRAY)
ssim_value = ssim(image_original_gray, image_enhanced_gray)
# Отображение улучшенного изображения
plt.subplot(1, 2, 2)
plt.title("Enhanced Coloured")
plt.imshow(cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB))
# Отображение изображений
plt.show()
# Вывод значения SSIM в консоль
print(f"Значение SSIM между исходным и улучшенным изображениями: {ssim_value}")

