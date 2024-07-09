from skimage import io, transform
from skimage.metrics import mean_squared_error
# Загрузка изображений
image1 = io.imread('/home/user/OS/lab1/PROJECT/1.jpeg', as_gray=True)
image2 = io.imread('/home/user/OS/lab1/PROJECT/1_after.jpeg', as_gray=True)
# Изменение размеров изображений до одинаковых размеров
image2_resized = transform.resize(image2, image1.shape, anti_aliasing=True)
# Вычисление Mean Squared Error между изображениями
mse = mean_squared_error(image1, image2_resized)
print(f"Mean Squared Error между изображениями: {mse}")
