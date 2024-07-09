import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

# Загрузка изображения 1.jpeg
image = cv2.imread('/home/user/OS/lab1/PROJECT/1.jpeg')

# Отображение изображения 1.jpeg
plt.subplot(1, 2, 1)
plt.title("Original picture")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Удаление шума с помощью медианного фильтра
filtered_image = cv2.medianBlur(image, 15)

# Сохранение изображения
cv2.imwrite("median_blur.jpg", filtered_image)

# Отображение изображения median_blur.jpg
plt.subplot(1, 2, 2)
plt.title("Median Blur picture")
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))

# Вычисление метрики SSIM
ssim = compare_ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY))

# Вывод значения SSIM в терминал
print(f"Значение SSIM между оригинальным и улучшенным изображениями: {ssim:.2f}")

plt.show()

