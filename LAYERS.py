import pydicom
import matplotlib.pyplot as plt
import os #перебор файлов
import imageio #для создания gif-анимации

print('Ввод директории для считывания файлов формата .dcm: ') #C:\Users\User\Desktop\kursach\Head - 33311\3D_Ax_eSWAN_9 - пример ввода
directory_in = input() #папка со входными данными
print('Ввод директории для сохранения графиков: ') #C:\Users\User\Desktop\kursach\graphs\3D_Ax_eSWAN_9 - пример ввода
directory_out = input() #папка для вывода графиков

imgs = [] #массив картинок для gif

#создание отдельных картинок
print('Создание графиков начато. . .')
for filename in os.listdir(directory_in):
    if filename.endswith(".dcm"):  #filename - название файла без пути
        filepath = os.path.join(directory_in, filename) #filepath - полный путь
        data = pydicom.read_file(filepath) #создание графика
        plt.title(filepath)
        plt.imshow(data.pixel_array)
        imgname = directory_out + '\\' + filename[:-4] + '.png' #сохранение файла в требуемую директорию с изменением расширения
        print(imgname)
        plt.savefig(imgname)
    else:
        continue
print('Сохранение графиков завершено')

print('Создание gif-анимации начато') #склеивание картинок в gif.
#было бы более целесообразно брать картинки сразу при их создании, не пробегаясь второй раз, но с этой библиотекой так не выходит.

for filename in os.listdir(directory_out):
    if filename.endswith(".png"):  #filename - название файла без пути
        filepath = os.path.join(directory_out, filename) #filepath - полный путь
        imgs.append(imageio.imread(filepath)) #добавление картиночки в массив


directory_out_gif = directory_out+'\move.gif' #создание директории для вывода gif.
imageio.mimsave(directory_out_gif, imgs)

print('Создание gif-анимации завершено')
