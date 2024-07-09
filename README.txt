# ProjectPractica
Статья на тему «Исследование методов улучшения изображения» 
Введение
В современном мире, где визуальные данные играют все более значимую роль, вопрос улучшения качества изображений становится невероятно актуальным. Исследование методов улучшения изображений на основе глубокого обучения представляет собой важную область в развитии компьютерного зрения и визуальных технологий. С появлением новых методов и технологий в области нейросетей и генеративно-состязательных сетей (GAN) возможности по повышению качества изображений достигают новых высот. В рамках данного исследования, целью является анализ эффективности различных подходов и методов улучшения изображений с применением глубокого обучения, а также изучение их влияния на создание фотореалистичных реконструкций изображений. Новизна этого исследования заключается в стремлении выявить оптимальные методы и параметры для достижения высококачественных и реалистичных визуальных результатов. Рассмотрим более подробно процесс исследования методов улучшения изображения на основе глубокого обучения и его актуальность в современном обществе.

Описание:
В современном информационном обществе, где визуальные данные играют все более важную роль, тема методов улучшения изображений с использованием глубокого обучения становится крайне актуальной и значимой. Это обусловлено несколькими факторами.
Развитие технологий компьютерного зрения: С постоянным развитием методов глубокого обучения, архитектур нейронных сетей и подходов обработки изображений, возможности улучшения качества изображений значительно повышаются. Новые модели, такие как сверточные нейронные сети (CNN) и генеративно-состязательные сети (GAN), позволяют создавать более реалистичные и высококачественные изображения.
Растущий спрос на высококачественные изображения: В медицине, дизайне, мультимедиа, маркетинге и других областях жизни нас окружает огромное количество визуальной информации. Повышенный спрос на высококачественные изображения ставит перед исследователями и разработчиками задачу создания эффективных методов улучшения, способных справиться с этой потребностью.
Необходимость в разработке новых методов: С увеличением объема и сложности визуальных данных существующие методы улучшения становятся недостаточно эффективными. Континуальное развитие новых методов улучшения изображений на основе глубокого обучения позволяет идти в ногу с требованиями современной визуальной обработки данных.
Таким образом, изучение и совершенствование методов улучшения изображений с применением глубокого обучения не только является актуальным направлением исследований, но и имеет практическое значение для широкого спектра областей деятельности, где качество изображений играет ключевую роль в успешной реализации задач и проектов.

Исследования, которые проводились раньше по теме исследования:
До меня было проведено множество исследований. Вот некоторые из них:
1)	"Deep Image Prior" (Ulyanov et al., 2018): Работа, где предложен метод улучшения изображений, не требующий обучающего набора данных, а использующий архитектуру нейронной сети как априорное знание.
2)	"Super-Resolution Convolutional Neural Network" (Dong et al., 2016): Исследование, посвященное разработке супер-разрешающей нейронной сети для увеличения разрешения изображений.
3)	"Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (Johnson et al., 2016): Работа, где внедрены перцептивные потери для стилизации изображений в реальном времени и увеличения разрешения.
4)	"Deep Photo Style Transfer" (Luan et al., 2017): Исследование о стилевой передаче изображений с использованием глубокого обучения для сохранения содержания и изменения стиля в изображениях.
5)	"EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis" (Sajjadi et al., 2017): Работа, где предложен метод супер-разрешения изображений на основе синтеза текстур с использованием нейронных сетей.
6)	“Deep Learning Based Single Image Super-resolution: A Survey” (Ha, Viet Khanh et al., 2019): Обзор последних достижений в моделях и методах, основанных на глубоком обучении, которые были применены к задачам сверхвысокого разрешения одного изображения. Обобщение и сравнение различных моделей.
7)	“A Survey on Image Denoising Techniques” (Irene Mary Mathew et al., 2023): Рассматриваются способы денойзинга изображений, объясняются методы, используемые в последнее время, обсуждаются различные показатели, используемые для оценки моделей.

Исходные гипотезы и исследовательские вопросы:
          Для нашего исследования сформулируем исходные гипотезы и исследовательские вопросы:
1)	Исходные гипотезы: Предполагается, что интеграция различных методов глубокого обучения в процесс улучшения изображений даст хорошие результаты в изменении стиля и увеличении разрешения.
2)	Исследовательские вопросы: Какие конкретные методы улучшения изображений, основанные на глубоком обучении, будут наиболее эффективными для различных типов изображений и задач? Какие метрики наилучшим образом отражают улучшение качества изображений с помощью этих методов?

Задачи:
1)	Выбор методов улучшения изображения.
2)	Подготовка данных.
3)	Реализация выбранных методов.
4)	Эксперименты и оценка результатов.
5)	Сравнительный анализ методов.
6)	Выводы.

Глава 1. Пять методов улучшения изображения

Метрика №1:
Для первых пяти методов улучшения изображения воспользуемся метрикой SSIM. Подробнее опишем что из себя представляет SSIM. 
SSIM (Structural Similarity Index) - это метрика качества изображения, которая используется для измерения структурного сходства между двумя изображениями. SSIM оценивает изменения в содержании изображения, а не только изменения в яркости и цвете.
SSIM учитывает три аспекта восприятия изображений:
1)	Сходство: насколько сильно структуры изображения похожи друг на друга.
2)	Яркость: как хорошо яркость изображения соответствует друг другу.
3)	Контраст: как хорошо контраст изображения соответствует друг другу.
Результат SSIM находится в диапазоне от -1 до 1, где значение означает, что два изображения идентичны, а значение -1 указывает на полное отсутствие сходства между изображениями.
Таким образом, SSIM позволяет количественно оценить, насколько изображения структурно похожи друг на друга, что полезно при сравнении качества изображений в различных обработках, таких как сжатие, фильтрация или улучшение разрешения.

Некоторые методы улучшения изображения:
1)	Настройка яркости и контрастности изображения.
•	Настройка яркости: Изменение уровня освещения изображения для делания его более темным или светлым.
•	Настройка контрастности: Регулирование разницы между темными и светлыми участками изображения для увеличения резкости и четкости визуального восприятия.
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
          После выполнения программы выводится:
Значение SSIM между оригинальным и улучшенным изображениями: 0.6230794191408137
Итак, значение SSIM равное 0.623… говорит о том, что улучшенное изображение имеет некоторое сходство с оригинальным, однако есть места, где они различаются.
2)	Повышение резкости изображения.
Процесс усиления границ и контурных деталей на изображении для придания более четкого и детального внешнего вида.
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
После выполнения программы выводится:
Значение SSIM между оригинальным и улучшенным изображениями: 0.98575440758749.
Итак, значение 0.986 говорит о том, что большинство деталей, текстур и структур из оригинального изображения были успешно сохранены в улучшенном варианте. Это означает, что улучшенное изображение очень близко к оригинальному и несет в себе почти все его структурные аспекты.
3)	Удаление цифрового шума изображения.
Применение фильтров и алгоритмов для уменьшения помех, которые могут появиться на изображении в результате съемки или сжатия данных.  
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
После выполнения программы выводится:
Значение SSIM между исходным и улучшенным изображениями: 0.9859748003379373
Итак, значение 0.986… говорит о том, что большинство деталей, текстур и структур из оригинального изображения были успешно сохранены в улучшенном варианте. Это означает, что улучшенное изображение очень близко к оригинальному и несет в себе почти все его структурные аспекты.
4)	Улучшение цвета изображения.
Коррекция цветового баланса, насыщенности, оттенков и тонов на изображении для достижения более точного и приятного визуального представления.
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
После выполнения программы выводится:
Значение SSIM между исходным и улучшенным изображениями: 0.9859748003379373
Итак, значение 0.986… говорит о том, что большинство деталей, текстур и структур из оригинального изображения были успешно сохранены в улучшенном варианте. Это означает, что улучшенное изображение очень близко к оригинальному и несет в себе почти все его структурные аспекты.
5)	Изменение размера и масштабирования изображения.
Процесс изменения размера изображения для увеличения или уменьшения его четкости и разрешения, а также изменения масштаба для подгонки изображения под нужные размеры или пропорции.
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
После выполнения программы выводится:
Значение SSIM между исходным и измененным изображениями: 1.0
Итак, так как значение SSIM между исходным и измененным изображениями, равное 1.0, значит, что эти изображения практически идентичны по структурному сходству.              

Сравнительный анализ этих пяти методов:
Проанализируем все использованные методы, используя для сравнения качества их работы метрику SSIM.
Для метода настройки яркости и контрастности изображения (метод 1)  метрика качества составила 0.623. Данный показатель указывает на то, что, несмотря на наличие некоторого сходства с оригиналом, между изображениями всё же присутствуют структурные различия, и содержание изображения в полной мере сохранить не удалось.
Остальные методы показали гораздо более высокое сходство между оригинальным изображением и его улучшенной версией.
При повышении резкости изображения (метод 2) значение SSIM составило 0.986.  Такое значение говорит о значительном сохранении деталей, структур и текстур исходного изображения, успешном сохранении его структурных аспектов.
Аналогичный результат 0.986 продемонстрировало улучшение цвета изображения (метод 3).
Удаление цветового шума изображения (метод 4) характеризуется показателем метрики качества, равным 0.99. Это значение свидетельствует о высоком сходстве между двумя изображениями и практически полном отсутствии структурных различий между ними.
Наиболее высокий результат SSIM=1.0, означающий полную идентичность содержания оригинального изображения и улучшенного, был получен при изменении размера и масштабировании изображения (метод 5).

Вывод №1:
Из сравнительного анализа методов видно, что полная идентичность между содержанием исходного и улучшенного изображения достигается только при изменении размера и масштабировании (метод 5). Методы 2, 3, 4 также показали высокое структурное сходство между изображениями, близкое к идентичности. Менее всего структура изображения сохраняется при использовании метода 1, присутствуют некоторые различия. Таким образом, большая часть (4 из 5) рассмотренных методов демонстрируют полную или близкую к полной идентичность между двумя изображениями.
Но главная проблема этих методов, что они по отдельности неэффективно улучшают изображения, для этого рассмотрим более эффективный метод улучшения изображения, который улучшает изображения по нескольким критериям, а не только по одному.

Глава 2. Метод Real-ESRGAN
Метрика №2 (для метода Real-ESRGAN):
Мы будем рассматривать метод Real-ESRGAN (который мы подробнее рассмотри дальше). Для анализа изображения до улучшения и после мы воспользуемся метрикой Mean Squared Error. Рассмотрим ее подробнее.
Метрика Mean Squared Error (MSE) является одним из популярных методов для измерения различий между двумя наборами данных. В контексте обработки изображений, MSE используется для измерения разницы между яркостями пикселей на двух изображениях.
Для вычисления MSE сначала находится разница между соответствующими пикселями на двух изображениях. Затем эта разница в яркости возведена в квадрат, чтобы избежать отрицательных значений. В конце концов, суммируются квадраты разностей для всех пикселей и делят на количество пикселей, чтобы получить среднее значение среднеквадратичной ошибки. Чем меньше значение MSE, тем более похожи два изображения на друг друга.
MSE широко используется в компьютерном зрении и обработке изображений для сравнения качества изображений, обнаружения аномалий, сжатия изображений и других приложений, где важно количественно измерить разницу между изображениями. Вот для анализа изображений воспользуемся кодом программы на Python.
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

Метод Real-ESRGAN:
Для улучшения изображения воспользуемся методом Real-ESRGAN, который будет улучшать изображение по нескольким критериям. Подробнее рассмотри этот метод.
import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    args = parser.parse_args()
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)
    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)
if __name__ == '__main__':
    main()
Данный код представляет собой скрипт на языке Python для демонстрации инференса (вычисления вывода) с использованием Real-ESRGAN - улучшенной версии Enhanced Super-Resolution Generative Adversarial Network (ESRGAN) для повышения разрешения изображений.
Вот краткое пошаговое объяснение кода:
1)	Импортируются необходимые библиотеки и модули: argparse для обработки аргументов командной строки, cv2 для работы с изображениями, glob для работы с файлами, os для работы с операционной системой, а также модули для архитектур нейронных сетей и загрузки моделей.
2)	Определяется функция main(), в которой задается вся логика выполнения скрипта.
3)	Создается парсер аргументов командной строки, в котором определяются параметры запуска скрипта, такие как пути к изображениям, моделям, параметры обработки изображения и др.
4)	По заданному имени модели выбирается соответствующая модель нейронной сети и определяется путь к её весам.
5)	Создается экземпляр RealESRGANer для выполнения улучшения качества изображения на заданной модели.
6)	Если указан флаг face_enhance, то используется GFPGAN для улучшения лиц на изображении.
7)	Проверяется наличие входного изображения или папки с изображениями для обработки.
8)	Происходит обработка каждого изображения: сначала оно считывается с помощью OpenCV, затем улучшается RealESRGANer'ом или GFPGAN'ом (в случае улучшения лиц), и в конечном итоге сохраняется улучшенное изображение.
9)	Скрипт содержит логику обработки возможных ошибок, например, связанных с нехваткой памяти на GPU.
Этот скрипт предназначен для улучшения изображений с помощью Real-ESRGAN и GFPGAN, а также демонстрирует пример использования этих моделей для повышения качества изображений.
# flake8: noqa
# This file is used for deploying replicate models
# running: cog predict -i img=@inputs/00017_gray.png -i version='General - v3' -i scale=2 -i face_enhance=True -i tile=0
# push: cog push r8.im/xinntao/realesrgan
import os
os.system('pip install gfpgan')
os.system('python setup.py develop')
import cv2
import shutil
import tempfile
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer
try:
    from cog import BasePredictor, Input, Path
    from gfpgan import GFPGANer
except Exception:
    print('please install cog and realesrgan package')
class Predictor(BasePredictor):
    def setup(self):
        os.makedirs('output', exist_ok=True)
        # download weights
        if not os.path.exists('weights/realesr-general-x4v3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./weights')
        if not os.path.exists('weights/GFPGANv1.4.pth'):
            os.system('wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./weights')
        if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights'
            )
        if not os.path.exists('weights/RealESRGAN_x4plus_anime_6B.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P ./weights'
            )
        if not os.path.exists('weights/realesr-animevideov3.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth -P ./weights')
    def choose_model(self, scale, version, tile=0):
        half = True if torch.cuda.is_available() else False
        if version == 'General - RealESRGANplus':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = 'weights/RealESRGAN_x4plus.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'General - v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            model_path = 'weights/realesr-general-x4v3.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'Anime - anime6B':
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            model_path = 'weights/RealESRGAN_x4plus_anime_6B.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        elif version == 'AnimeVideo - v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            model_path = 'weights/realesr-animevideov3.pth'
            self.upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=model, tile=tile, tile_pad=10, pre_pad=0, half=half)
        self.face_enhancer = GFPGANer(
            model_path='weights/GFPGANv1.4.pth',
            upscale=scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=self.upsampler)
    def predict(
        self,
        img: Path = Input(description='Input'),
        version: str = Input(
            description='RealESRGAN version. Please see [Readme] below for more descriptions',
            choices=['General - RealESRGANplus', 'General - v3', 'Anime - anime6B', 'AnimeVideo - v3'],
            default='General - v3'),
        scale: float = Input(description='Rescaling factor', default=2),
        face_enhance: bool = Input(
            description='Enhance faces with GFPGAN. Note that it does not work for anime images/vidoes', default=False),
        tile: int = Input(
            description=
            'Tile size. Default is 0, that is no tile. When encountering the out-of-GPU-memory issue, please specify it, e.g., 400 or 200',
            default=0)
    ) -> Path:
        if tile <= 100 or tile is None:
            tile = 0
        print(f'img: {img}. version: {version}. scale: {scale}. face_enhance: {face_enhance}. tile: {tile}.')
        try:
            extension = os.path.splitext(os.path.basename(str(img)))[1]
            img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            elif len(img.shape) == 2:
                img_mode = None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            self.choose_model(scale, version, tile)

            try:
                if face_enhance:
                    _, _, output = self.face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.')

            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            # save_path = f'output/out.{extension}'
            # cv2.imwrite(save_path, output)
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            cv2.imwrite(str(out_path), output)
        except Exception as error:
            print('global exception: ', error)
        finally:
            clean_folder('output')
        return out_path
def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
Данный скрипт на Python предназначен для развертывания и использования набора моделей Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) и GFPGAN (Generative Face Performance Enhancement) для улучшения качества изображений. Вот пошаговое объяснение кода:
1) В начале скрипта импортируются необходимые библиотеки и модули, а также задаются комментарии flake8 для игнорирования ошибок.
2) В функции setup() инициализируется структура папок, куда будут сохраняться результаты, и загружаются веса моделей, если они еще не были загружены.
3) В методе choose_model() выбирается модель в зависимости от указанной версии и масштаба увеличения. Для каждой версии модели инициализируется соответствующая нейронная сеть RealESRGANer или GFPGANer для дальнейшего применения.
4) В методе predict() задается логика прогнозирования (применения модели к входному изображению): считывается изображение, выбирается подходящая модель, улучшается изображение (возможно с улучшением лиц, если указано), сохраняется улучшенное изображение и возвращается его путь.
5) Для управления памятью создан метод clean_folder(), который чистит временную папку от созданных временных файлов после завершения обработки.
Этот скрипт предоставляет удобный способ использования готовых моделей Real-ESRGAN и GFPGAN для улучшения качества изображений. В случае возникновения ошибок скрипт также обеспечивает информирование о них.
Вот примеры некоторых изображений до и после улучшения при помощи метода Real-ESRGAN (см.рис.1-3):
 
Рис.1
 
Рис.2
 
Рис.3

Сравнительный анализ метода Real-ESRGAN
В ходе исследования можно увидеть, что практически у всех исследуемых изображений значение метрики равно 0.01 (за некоторым исключением), что указывает на небольшую среднеквадратичную разницу в яркости пикселей между этими изображениями. Это говорит о том, что изображения очень похожи и имеют незначительные различия в яркости.

Вывод №2:
1)	Высокое сходство: Значение MSE близкое к 0.01 подразумевает, что пиксели изображений очень близки по яркости. Это может означать, что изображения могут быть очень похожи или даже идентичны.
2)	Высокое качество: Маленькое значение MSE указывает на то, что потери информации при обработке изображений были минимальными. Это характерно для качественных методов обработки изображений.
3)	Применимость: Такое значение MSE полезно в задачах, где важно сохранить детали и текстуру в изображении, таких как медицинская диагностика, реконструкция изображений высокого разрешения и др.
4)	Важность контекста: Важно помнить, что значение MSE само по себе не дает полной картины. В контексте конкретного проекта или задачи, дополнительный анализ и другие метрики могут быть также необходимы для качественной оценки результатов.


Глава 3. Заключение
В ходе исследования можно понять, что интеграция различных методов глубокого обучения, таких как ESRGAN и его улучшенная версия Real-ESRGAN, в процесс улучшения изображений позволяет добиваться высококачественных результатов в изменении стиля и увеличении разрешения, что как раз и подтверждает гипотезу. 
Благодаря метрикам SSIM и MSE удалось провести сравнительный анализ методов улучшения изображения.
