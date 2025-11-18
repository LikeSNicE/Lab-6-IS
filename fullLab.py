# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# import os
# import cv2
# from glob import glob
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import cv2 as cv
# import PIL
# ## проверка наличия рентгеновских лучей и соответствующих им масок
# from glob import glob
# import re
# from collections import defaultdict
# import pandas as pd
# import matplotlib.pyplot as plt

# import cv2, ast
# import torchvision
# import matplotlib.pyplot as plt
# from os.path import isfile
# import torch.nn.init as init
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import os
# from PIL import Image, ImageFilter
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from torch.utils.data import Dataset
# from torchvision import transforms
# from torch.optim import Adam, SGD, RMSprop
# import time
# from torch.autograd import Variable
# import torch.functional as F
# from tqdm import tqdm
# from sklearn import metrics
# import urllib
# import pickle
# #import cv2
# import torch.nn.functional as F
# from torchvision import models
# import seaborn as sns
# import random
# #from apex import amp
# import sys

# from collections import defaultdict
# import re
# import segmentation_models_pytorch as smp



# path = "./Lung Segmentation/"


# lung_image_paths = glob(os.path.join(path, "CXR_png/*.png"))
# mask_image_paths = glob(os.path.join(path, "masks/*.png"))



# related_paths = defaultdict(list)

# for img_path in lung_image_paths:
#     # Имя файла без пути и расширения
#     img_name = os.path.splitext(os.path.basename(img_path))[0]  # пример: CXR_001

#     # Ищем маски, содержащие ту же основу имени
#     for mask_path in mask_image_paths:
#         mask_name = os.path.splitext(os.path.basename(mask_path))[0]

#         # если mask_name содержит img_name, значит это соответствующая маска
#         if img_name in mask_name:
#             related_paths["image_path"].append(img_path)
#             related_paths["mask_path"].append(mask_path)
#             break  # нашли маску → выходим из цикла


# paths_df = pd.DataFrame.from_dict(related_paths)

# xray_num = 5
# img_path = paths_df["image_path"][xray_num]
# mask_path = paths_df["mask_path"][xray_num]

# img = PIL.Image.open(img_path)
# mask = PIL.Image.open(mask_path)

# fig = plt.figure(figsize = (10,10))

# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(img, cmap = "gray")
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(mask, cmap = "gray")

# from tqdm import tqdm
# import cv2
# def prepare_train_test(df = pd.DataFrame(), resize_shape = tuple(), color_mode = "rgb"):
#     img_array = list()
#     mask_array = list()

#     for image_path in tqdm(paths_df.image_path):
#         resized_image = cv2.resize(cv2.imread(image_path),resize_shape)
#         resized_image = resized_image/255.
#         if color_mode == "gray":
#             img_array.append(resized_image[:,:,0])
#         elif color_mode == "rgb":
#             img_array.append(resized_image[:,:,:])
#       # img_array.append(resized_image)

#     for mask_path in tqdm(paths_df.mask_path):
#         resized_mask = cv2.resize(cv2.imread(mask_path),resize_shape)
#         resized_mask = resized_mask/255.
#         mask_array.append(resized_mask[:,:,0])
#         # mask_array.append(resized_image)

#     return img_array, mask_array

# img_array, mask_array = prepare_train_test(df = paths_df, resize_shape = (224,224), color_mode = "gray")

# 704/100*80

# # Узнайте реальный размер вашего массива
# print(f"Общий размер данных: {len(img_array)}") # Должно вывести 132

# # Укажите правильную точку разреза
# split_point = 105

# X_train, X_test = img_array[:split_point], img_array[split_point:]
# y_train, y_test = mask_array[:split_point], mask_array[split_point:]

# print(f"Размер обучающей выборки (X_train): {len(X_train)}")
# print(f"Размер тестовой выборки (X_test): {len(X_test)}")

# len(X_train) + len(X_test)

# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader,Dataset
# from PIL import Image

# import numpy as np
# import matplotlib.pyplot as plt
# class MyDataset(Dataset):
#     def __init__(self, data, targets, transform):
#         self.data = data
#         self.targets = targets
#         self.transforms = transform

#     def __getitem__(self, index):
#         x = self.data[index]
#         y = self.targets[index]
#         if self.transforms:
#             x = self.transforms(x)
#         return x, y

#     def __len__(self):
#         return len(self.data)
#     # %matplotlib inline

# from torchvision import transforms as T

# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
# #     if train:
# #         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)


# dataset = MyDataset(X_train,y_train, get_transform(True))
# train_loader = DataLoader(dataset, batch_size=4)

# testset = MyDataset(X_test,y_test, get_transform(True))
# val_loader = DataLoader(testset, batch_size=4)

# import tqdm
# for img,target in tqdm.tqdm(train_loader):
#     z = 1

# plt.imshow(img[0][0])

# plt.imshow(target[0])

# #№ !pip install segmentation-models-pytorch
# # import segmentation_models_pytorch as smp

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = smp.Unet(
#     encoder_name="resnet50",
#     in_channels=1,
#     classes=1,
# )
# model = model.to(device)


# def dice_loss(pred, target):
#     pred = torch.sigmoid(pred)  # переводим логиты в вероятности

#     intersection = (pred * target).sum()
#     pred_sum = pred.sum()
#     target_sum = target.sum()

#     return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))


# criterion = dice_loss
# optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

# losses = []
# vlosses = []

# iter_num = 0
# n_epoch = 50
# start_time = time.time()

# for epoch_num in range(n_epoch):
#     loss_ = []
#     vloss_ = []
#     for i_batch, sampled_batch in enumerate(train_loader):
#         image_batch, label_batch = sampled_batch
#         image_batch = image_batch.float()
#         image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
#         outputs = model(image_batch)
#         outputs = torch.squeeze(outputs, 1)
#         loss = criterion(outputs, label_batch.float())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         loss_.append(loss.item())
#     losses.append(np.mean(loss_,axis = 0))
#     print('train epoch ',epoch_num)
#     print('-------- train dice ce total', np.mean(loss_,axis = 0))
#     with torch.no_grad():
#         for i_batch, sampled_batch1 in enumerate(val_loader):
#             image_batch1, label_batch1 = sampled_batch1
#             image_batch1 = image_batch1.float()
#             image_batch1, label_batch1 = image_batch1.cuda(), label_batch1.cuda()
#             outputs1 = model(image_batch1)
#             outputs1 = torch.squeeze(outputs1, 1)
#             vloss = criterion(outputs1, label_batch1.float())
#             vloss_.append(vloss.item())
#         vlosses.append(np.mean(vloss_,axis = 0))
#         print('-------- test loss ', np.mean(vloss_,axis = 0))
# end_time = time.time()

# # Create a range of epochs for x-axis
# epochs = range(1, len(losses) + 1)

# # Plot training and validation losses on the same graph
# plt.plot(epochs, losses, label='Training Loss')
# plt.plot(epochs, vlosses, label='Validation Loss')

# # Add labels and title
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Losses over Epochs')

# # Add legend
# plt.legend()

# # Show the plot
# plt.show()

# model.eval()

# d = []

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import random # <--- ШАГ 1: Импортируем библиотеку для случайных чисел

# # --- Сначала убедимся, что у нас есть функция для предсказания ---
# # (эта функция осталась без изменений с прошлого раза)

# def predict_one_image(model, image_numpy):
#     """
#     Принимает модель и одно изображение (массив NumPy) и возвращает
#     предсказанную для него маску (тоже массив NumPy).
#     """
#     model.eval()
#     if image_numpy.ndim == 3:
#         img_tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).unsqueeze(0)
#     else:
#         img_tensor = torch.from_numpy(image_numpy).unsqueeze(0).unsqueeze(0)

#     img_tensor = img_tensor.float().cuda()

#     with torch.no_grad():
#         output_tensor = model(img_tensor)

#     mask_numpy = output_tensor.cpu().numpy().squeeze()
#     return mask_numpy

# # --- А теперь основной код, который делает выбор и предсказание ---

# # ПРЕДПОЛАГАЕТСЯ, ЧТО У ВАС УЖЕ ЕСТЬ:
# # model - ваша обученная нейросеть
# # X_test - ваш тестовый набор изображений
# # y_test - ваш тестовый набор настоящих масок

# # --- ШАГ 2: Генерируем случайный индекс и выбираем данные ---

# # Получаем общее количество изображений в тестовом наборе
# num_test_images = len(X_test)
# # Генерируем случайный индекс от 0 до (количество - 1)
# random_index = random.randint(0, num_test_images - 1)

# # Выбираем случайное изображение и его настоящую маску по этому индексу
# random_image = X_test[random_index]
# true_mask = y_test[random_index]

# print(f"Отображается случайное изображение с индексом: {random_index}")

# # --- ШАГ 3: Делаем предсказание для выбранного изображения ---

# predicted_mask = predict_one_image(model, random_image)

# # --- ШАГ 4: Улучшенная визуализация (сразу 3 картинки) ---

# fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Создаем 3 области для рисования

# # 1. Исходное изображение
# axes[0].imshow(np.squeeze(random_image))
# axes[0].set_title('Исходное изображение')
# axes[0].axis('off')

# # 2. Настоящая маска (правильный ответ)
# axes[1].imshow(np.squeeze(true_mask), cmap='gray')
# axes[1].set_title('Настоящая маска (Ground Truth)')
# axes[1].axis('off')

# # 3. Предсказанная моделью маска
# axes[2].imshow(predicted_mask, cmap='gray')
# axes[2].set_title('Предсказанная маска')
# axes[2].axis('off')

# plt.show()

# model.eval()

# import numpy as np
# from scipy.spatial.distance import cdist

# def calculate_dice_coefficient(mask_true, mask_pred, smooth=1e-6):
#     """
#     Вычисляет коэффициент Дайса (Dice Coefficient).
#     Это мера сходства двух выборок.
#     Формула: (2 * |A ∩ B|) / (|A| + |B|)
#     """
#     # Убедимся, что маски бинарные (0 или 1)
#     mask_true = (mask_true > 0).astype(np.bool_)
#     mask_pred = (mask_pred > 0).astype(np.bool_)

#     # Сводим маски в одномерный массив
#     intersection = np.sum(mask_true & mask_pred)
#     sum_of_masks = np.sum(mask_true) + np.sum(mask_pred)

#     # Вычисляем коэффициент Дайса, добавляя smooth для избежания деления на ноль
#     dice_coefficient = (2. * intersection + smooth) / (sum_of_masks + smooth)

#     return dice_coefficient

# def precision_score(groundtruth_mask, pred_mask):
#     """
#     Вычисляет точность (Precision).
#     Формула: TP / (TP + FP)
#     """
#     # Убедимся, что маски бинарные
#     groundtruth_mask = (groundtruth_mask > 0).astype(np.bool_)
#     pred_mask = (pred_mask > 0).astype(np.bool_)

#     TP = np.sum(groundtruth_mask & pred_mask) # True Positives
#     FP = np.sum(~groundtruth_mask & pred_mask) # False Positives

#     denominator = TP + FP
#     if denominator == 0:
#         # Если нет предсказанных положительных, точность равна 1.0 (идеально)
#         return 1.0

#     precision = TP / denominator
#     return round(precision, 3)

# def recall_score(groundtruth_mask, pred_mask):
#     """
#     Вычисляет полноту (Recall).
#     Формула: TP / (TP + FN)
#     """
#     # Убедимся, что маски бинарные
#     groundtruth_mask = (groundtruth_mask > 0).astype(np.bool_)
#     pred_mask = (pred_mask > 0).astype(np.bool_)

#     TP = np.sum(groundtruth_mask & pred_mask) # True Positives
#     FN = np.sum(groundtruth_mask & ~pred_mask) # False Negatives

#     denominator = TP + FN
#     if denominator == 0:
#         # Если нет реальных положительных, полнота равна 1.0 (идеально)
#         return 1.0

#     recall = TP / denominator
#     return round(recall, 3)

# def calculate_iou(mask_true, mask_pred, smooth=1e-6):
#     """
#     Вычисляет IoU (Intersection over Union) или индекс Жаккара.
#     Формула: |A ∩ B| / |A ∪ B|
#     """
#     # Убедимся, что маски бинарные
#     mask_true = (mask_true > 0).astype(np.bool_)
#     mask_pred = (mask_pred > 0).astype(np.bool_)

#     intersection = np.sum(mask_true & mask_pred)
#     union = np.sum(mask_true | mask_pred)

#     # Вычисляем IoU, добавляя smooth для избежания деления на ноль
#     iou = (intersection + smooth) / (union + smooth)
#     return iou

# def hausdorff_distance(mask1, mask2):
#     """
#     Вычисляет расстояние Хаусдорфа между двумя наборами точек.
#     Это максимальное из расстояний от точки в одном наборе до ближайшей точки в другом.
#     """
#     # Убедимся, что маски бинарные
#     mask1 = (mask1 > 0).astype(np.bool_)
#     mask2 = (mask2 > 0).astype(np.bool_)

#     # Находим координаты всех точек (пикселей) в каждой маске
#     coords1 = np.argwhere(mask1)
#     coords2 = np.argwhere(mask2)

#     # Обработка случая, когда одна или обе маски пусты
#     if len(coords1) == 0 and len(coords2) == 0:
#         return 0.0  # Расстояние между двумя пустыми наборами равно 0
#     if len(coords1) == 0 or len(coords2) == 0:
#         return np.inf # Расстояние от непустого до пустого набора - бесконечность

#     # Вычисляем матрицу попарных расстояний между точками
#     dist_matrix = cdist(coords1, coords2, 'euclidean')

#     # Вычисляем расстояние Хаусдорфа
#     # h(A, B) = max(min(dist(a, b))) для всех a в A
#     h1 = np.max(np.min(dist_matrix, axis=1))
#     # h(B, A) = max(min(dist(a, b))) для всех b в B
#     h2 = np.max(np.min(dist_matrix, axis=0))

#     hausdorff_dist = max(h1, h2)

#     return hausdorff_dist

# # 1. Создаем пустые списки для хранения оценок
# dice = []
# hd = []
# prec = []
# rec = []
# iou = []

# # 2. Проходим в цикле по ВСЕМУ тестовому набору
# # tqdm.tqdm - для красивого индикатора прогресса
# for i in tqdm.tqdm(range(len(X_test))):

#     # --- ЭТО ГЛАВНЫЕ ИЗМЕНЕНИЯ ---

#     # Получаем i-тое изображение и настоящую маску
#     image_to_test = X_test[i]
#     mask_true = y_test[i]

#     # ДЕЛАЕМ НОВОЕ ПРЕДСКАЗАНИЕ для текущего изображения
#     mask_pred = predict_one_image(model, image_to_test)

#     # ------------------------------------

#     # 3. Вычисляем метрики, сравнивая настоящую маску с ПРАВИЛЬНОЙ предсказанной
#     #    и передаем в функции целые 2D-массивы
#     dice.append(calculate_dice_coefficient(mask_true, mask_pred))
#     hd.append(hausdorff_distance(mask_true, mask_pred)) # <-- Исправлено
#     prec.append(precision_score(mask_true, mask_pred))
#     rec.append(recall_score(mask_true, mask_pred))
#     iou.append(calculate_iou(mask_true, mask_pred))

# # 4. После завершения цикла вычисляем и печатаем средние значения
# print("\n--- Средние метрики по всему тестовому набору ---")
# print(f"Средний Dice Coefficient: {np.mean(dice):.4f}")
# print(f"Средний IoU: {np.mean(iou):.4f}")
# print(f"Средняя Precision: {np.mean(prec):.4f}")
# print(f"Средняя Recall: {np.mean(rec):.4f}")
# print(f"Среднее Hausdorff Distance: {np.mean(hd):.4f}")




import os
import re
from glob import glob
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import SGD

import segmentation_models_pytorch as smp
from scipy.spatial.distance import cdist
import random
import time

# ----------------------
# Пути к папкам
# ----------------------
path = "./Lung Segmentation/"
lung_image_paths = glob(os.path.join(path, "CXR_png/*.png"))
mask_image_paths = glob(os.path.join(path, "masks/*.png"))

# ----------------------
# Связываем изображения и маски
# ----------------------
related_paths = defaultdict(list)

for img_path in lung_image_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # имя файла без расширения
    for mask_path in mask_image_paths:
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        if img_name in mask_name:  # маска содержит имя изображения
            related_paths["image_path"].append(img_path)
            related_paths["mask_path"].append(mask_path)

paths_df = pd.DataFrame.from_dict(related_paths)
print(f"Найдено {len(paths_df)} изображений с масками.")

# ----------------------
# Проверка нескольких изображений
# ----------------------
xray_num = 5
img_path = paths_df["image_path"][xray_num]
mask_path = paths_df["mask_path"][xray_num]

img = Image.open(img_path)
mask = Image.open(mask_path)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img, cmap="gray")
ax1.set_title("Изображение")
ax1.axis("off")

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(mask, cmap="gray")
ax2.set_title("Маска")
ax2.axis("off")
plt.show()

# ----------------------
# Подготовка данных
# ----------------------
def prepare_train_test(df, resize_shape=(224,224), color_mode="gray"):
    img_array = []
    mask_array = []

    for image_path in tqdm(df.image_path, desc="Обработка изображений"):
        resized_image = cv2.resize(cv2.imread(image_path), resize_shape)
        resized_image = resized_image / 255.
        if color_mode == "gray":
            img_array.append(resized_image[:,:,0])
        else:
            img_array.append(resized_image)

    for mask_path in tqdm(df.mask_path, desc="Обработка масок"):
        resized_mask = cv2.resize(cv2.imread(mask_path), resize_shape)
        resized_mask = resized_mask / 255.
        mask_array.append(resized_mask[:,:,0])

    return img_array, mask_array

img_array, mask_array = prepare_train_test(paths_df, resize_shape=(224,224), color_mode="gray")
print(f"Общий размер данных: {len(img_array)}")

# ----------------------
# Разделение на обучающую и тестовую выборки
# ----------------------
split_point = int(len(img_array) * 0.8)
X_train, X_test = img_array[:split_point], img_array[split_point:]
y_train, y_test = mask_array[:split_point], mask_array[split_point:]

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# ----------------------
# Dataset и DataLoader
# ----------------------
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def get_transform():
    return transforms.Compose([transforms.ToTensor()])

train_dataset = MyDataset(X_train, y_train, get_transform())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = MyDataset(X_test, y_test, get_transform())
val_loader = DataLoader(test_dataset, batch_size=4)

# ----------------------
# Устройство (CPU)
# ----------------------
device = torch.device("cpu")

# ----------------------
# Создание модели
# ----------------------
model = smp.Unet(
    encoder_name="resnet50",
    in_channels=1,
    classes=1
)
model = model.to(device)

# ----------------------
# Функция Dice Loss
# ----------------------
def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.*intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)

criterion = dice_loss
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

# ----------------------
# Обучение модели
# ----------------------
losses = []
vlosses = []
n_epoch = 10
start_time = time.time()

for epoch_num in range(n_epoch):
    model.train()
    loss_ = []
    for image_batch, label_batch in train_loader:
        image_batch = image_batch.float().to(device)
        label_batch = label_batch.float().to(device)

        outputs = model(image_batch)
        outputs = torch.squeeze(outputs, 1)
        loss = criterion(outputs, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_.append(loss.item())
    losses.append(np.mean(loss_))
    print(f"Epoch {epoch_num+1}/{n_epoch}, Train Dice Loss: {np.mean(loss_):.4f}")

    # Validation
    model.eval()
    vloss_ = []
    with torch.no_grad():
        for image_batch, label_batch in val_loader:
            image_batch = image_batch.float().to(device)
            label_batch = label_batch.float().to(device)
            outputs = model(image_batch)
            outputs = torch.squeeze(outputs, 1)
            loss = criterion(outputs, label_batch)
            vloss_.append(loss.item())
    vlosses.append(np.mean(vloss_))
    print(f"Validation Dice Loss: {np.mean(vloss_):.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# ----------------------
# График потерь
# ----------------------
epochs = range(1, n_epoch+1)
plt.plot(epochs, losses, label='Train Loss')
plt.plot(epochs, vlosses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# ----------------------
# Предсказание и визуализация
# ----------------------
def predict_one_image(model, image_numpy):
    model.eval()
    if image_numpy.ndim == 2:
        img_tensor = torch.from_numpy(image_numpy).unsqueeze(0).unsqueeze(0)
    else:
        img_tensor = torch.from_numpy(image_numpy).permute(2,0,1).unsqueeze(0)
    img_tensor = img_tensor.float().to(device)
    with torch.no_grad():
        output_tensor = model(img_tensor)
    return output_tensor.cpu().numpy().squeeze()

random_index = random.randint(0, len(X_test)-1)
random_image = X_test[random_index]
true_mask = y_test[random_index]
pred_mask = predict_one_image(model, random_image)

fig, axes = plt.subplots(1,3,figsize=(18,6))
axes[0].imshow(random_image, cmap='gray'); axes[0].set_title('Исходное')
axes[1].imshow(true_mask, cmap='gray'); axes[1].set_title('Ground Truth')
axes[2].imshow(pred_mask, cmap='gray'); axes[2].set_title('Predicted')
for ax in axes:
    ax.axis('off')
plt.show()

# ----------------------
# Метрики
# ----------------------
def calculate_dice_coefficient(mask_true, mask_pred, smooth=1e-6):
    mask_true = (mask_true>0).astype(np.bool_)
    mask_pred = (mask_pred>0).astype(np.bool_)
    intersection = np.sum(mask_true & mask_pred)
    return (2.*intersection + smooth)/(np.sum(mask_true)+np.sum(mask_pred)+smooth)

def precision_score(mask_true, mask_pred):
    mask_true = (mask_true>0).astype(np.bool_)
    mask_pred = (mask_pred>0).astype(np.bool_)
    TP = np.sum(mask_true & mask_pred)
    FP = np.sum(~mask_true & mask_pred)
    return 1.0 if TP+FP==0 else TP/(TP+FP)

def recall_score(mask_true, mask_pred):
    mask_true = (mask_true>0).astype(np.bool_)
    mask_pred = (mask_pred>0).astype(np.bool_)
    TP = np.sum(mask_true & mask_pred)
    FN = np.sum(mask_true & ~mask_pred)
    return 1.0 if TP+FN==0 else TP/(TP+FN)

def calculate_iou(mask_true, mask_pred, smooth=1e-6):
    mask_true = (mask_true>0).astype(np.bool_)
    mask_pred = (mask_pred>0).astype(np.bool_)
    intersection = np.sum(mask_true & mask_pred)
    union = np.sum(mask_true | mask_pred)
    return (intersection+smooth)/(union+smooth)

def hausdorff_distance(mask1, mask2):
    mask1 = (mask1>0).astype(np.bool_)
    mask2 = (mask2>0).astype(np.bool_)
    coords1 = np.argwhere(mask1)
    coords2 = np.argwhere(mask2)
    if len(coords1)==0 and len(coords2)==0: return 0.0
    if len(coords1)==0 or len(coords2)==0: return np.inf
    dist_matrix = cdist(coords1, coords2, 'euclidean')
    return max(np.max(np.min(dist_matrix, axis=1)), np.max(np.min(dist_matrix, axis=0)))

# Вычисляем метрики на тесте
dice = []
iou = []
prec = []
rec = []
hd = []

for i in tqdm(range(len(X_test))):
    mask_true = y_test[i]
    mask_pred = predict_one_image(model, X_test[i])
    dice.append(calculate_dice_coefficient(mask_true, mask_pred))
    iou.append(calculate_iou(mask_true, mask_pred))
    prec.append(precision_score(mask_true, mask_pred))
    rec.append(recall_score(mask_true, mask_pred))
    hd.append(hausdorff_distance(mask_true, mask_pred))

print("\n--- Средние метрики по тестовому набору ---")
print(f"Dice: {np.mean(dice):.4f}, IoU: {np.mean(iou):.4f}")
print(f"Precision: {np.mean(prec):.4f}, Recall: {np.mean(rec):.4f}")
print(f"Hausdorff Distance: {np.mean(hd):.4f}")
