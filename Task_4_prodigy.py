!pip install numpy pandas matplotlib seaborn scikit-learn opencv-python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os


data_dir = r"C:\Users\Pranali Kolhe\Downloads\leapGestRecog"  # Update this path
categories = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]  # Update categories

data = []
labels = []

for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img_array, (64, 64))  # Resize to 64x64
            data.append(resized_img.flatten())  # Flatten the 2D image to 1D
            labels.append(class_num)
        except Exception as e:
            pass

data = np.array(data)
labels = np.array(labels)

# Check the shapes of data and labels
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Check the first few entries to ensure data is loaded correctly
print(data[:5])
print(labels[:5])


# Ensure data is a 2D array
if data.ndim == 1:
    data = data.reshape(-1, 1)

scaler = StandardScaler()
data = scaler.fit_transform(data)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Check the shapes of the splits
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


model = SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
