import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import metrics

# Загрузка изображения 1.jpeg
image = cv2.imread('/home/user/OS/lab1/PROJECT/1.jpeg')

# Отображение изображения 1.jpeg
plt.subplot(1, 2, 1)
plt.title("Original picture")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Создание ядра для повышения резкости 
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Повышение резкости изображения
sharpened_image = cv2.filter2D(image, -1, kernel)

cv2.imwrite('sharpened_image.jpg', sharpened_image)

# Отображение изображения sharpened_image.jpg
plt.subplot(1, 2, 2)
plt.title("Sharpening picture")
plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.show()

# Вычисление метрики SSIM
ssim_value = metrics.structural_similarity(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY))
            print("Значение SSIM между оригинальным и улучшенным изображениями:", ssim_value)
