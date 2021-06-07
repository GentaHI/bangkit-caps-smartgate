from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Masukan path spesifik ke dataset masker")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())


EPOCHS = 25
BS = 32

# ambil list directory foto di dalam folder yang ditulis
print("Data fotonya akan diambil")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# looping untuk preporcess foto
for imagePath in imagePaths:

	label = imagePath.split(os.path.sep)[-2]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update data dan label
	data.append(image)
	labels.append(label)

# konversi ke array numpy
print("Foto sudah di optimasi")
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot encoding untuk label
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# 85% training, 15% testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.15, stratify=labels, random_state=42)

# Augmentasi foto 
aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.25,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.25,
	horizontal_flip=True,
	fill_mode="nearest")

#load MobileNetV2 untuk melakukan fine-tuning, head tidak include
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# membuat head model untuk di train 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Gabungkan model sebelum memulai train
model = Model(inputs=baseModel.input, outputs=headModel)

# semua layer kecuali base layer yang akan di train
for layer in baseModel.layers:
	layer.trainable = False

print("Memulai compiling")
opt = Adam(lr=1e-4, decay= 1e-4 / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train head
print("Memulai training")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Memulai prediksi dengan testing
print("Evaliasi ulang model")
predIdxs = model.predict(testX, batch_size=BS)

# mencari probabiliti prediksi tertinggi 
predIdxs = np.argmax(predIdxs, axis=1)

# print hasil
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# simpan model
print("Menyimpan model")
model.save(args["model"], save_format="h5")

# buat plot training loss & accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])