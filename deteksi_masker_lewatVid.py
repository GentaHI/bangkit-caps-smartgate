from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


def detect_and_predict_mask(frame, faceNet, maskNet):
	# ambil data dimensi frame nya dan buat blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# mengambil deteksi wajah dari network
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# deklarasi list wajah, lokasi dir, dan prediksi dari network deteksi masker
	wajah = []
	locs = []
	preds = []

	
	for i in range(0, detections.shape[2]):
		# extract nilai confidence nya
		confidence = detections[0, 0, i, 2]

		# di filter confidence rendahnya (dibawah 50%)
		if confidence > args["confidence"]:
			# deklarasi bounding box untuk objek deteksi
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# memastikan bounding box nya tetap berada di dimensi frame deteksi wajah
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# ekstrak face ROI, convert BGR ke RGB channel
			# ordering, resize 224x224, preprocess
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# tambahkan wajah dan bounding box
			wajah.append(face)
			locs.append((startX, startY, endX, endY))

	# hanya lakukan prediksi hanya jika ada satu wajah terdeteksi
	if len(wajah) > 0:
		# Melakukan prediksi pada semua wajah sekaligus
		wajah = np.array(wajah, dtype="float32")
		preds = maskNet.predict(wajah, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="untuk menemukan lokasi model deteksi wajah di dir")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="untuk menemukan lokasi mask_detector.model yg sudh di train")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="mendeklarasikan minimum probabilitas untuk menyaring prob yang lemah")
args = vars(ap.parse_args())

#Memuat model deteksi wajah

#     RESNETSSD_FACEDETECTOR face detector based on SSD framework with reduced ResNet-10 backbone
    
#     homepage = https://github.com/opencv/opencv/blob/3.4.0/samples/dnn/face_detector/how_to_train_face_detector.txt
    
#     file = test/dnn/ResNetSSD_FaceDetector/deploy.prototxt
#     url  = https://github.com/opencv/opencv/raw/3.4.0/samples/dnn/face_detector/deploy.prototxt
#     hash = 006BAF926232DF6F6332DEFB9C24F94BB9F3764E
    
#     file = test/dnn/ResNetSSD_FaceDetector/res10_300x300_ssd_iter_140000.caffemodel
#     url  = https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
#     hash = 15aa726b4d46d9f023526d85537db81cbc8dd566
#     size = 10.1 MB
print("Membuka face detector model")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Memuat model face detector untuk deteksi masker
print("Membuka model deteksi masker")
maskNet = load_model(args["model"])

# memulai penangkapan video dari cam, time.sleep untuk persiapan kamera
print("Memulai video")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# looping selama stream cam berlanjut
while True:
	# mengambil frame stream camera dan resize 400 pixel
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# memulai deteksi wajah dan prediksi pemakaian maskernya
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# looping terhadap koordinat deteksi wajah
	for (box, pred) in zip(locs, preds):
		# membagi bounding box dan prediksi
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# deklarasi kelas label dan warna bounding box dan teksnya 
		label = "Bermasker" if mask > withoutMask else "Tanpa masker"
		# merah untuk tanpa masker, dan hijau untuk bermasker
		color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)
			
		# menambahkan nilai probabilitas pada teks
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# melihatkan label dan kotak bounding box pada frame stream camera
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# tampilkan frame output deteksi
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# ketik huruf "q" untuk menutup dan selesai dengan deteksi
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
