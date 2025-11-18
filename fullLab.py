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
