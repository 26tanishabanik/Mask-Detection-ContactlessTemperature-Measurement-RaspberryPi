# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import telegram
import telegram_send
import board
import adafruit_mlx90614
import busio as io
from time import sleep

#i2c = io.I2C(board.SCL, board.SDA, frequency = 100000)
#mlx = adafruit_mlx90614.MLX90614(i2c)

bot = telegram.Bot(token='1728241325:AAGy6Ooyd-gq7UQxYFPSwyKGvcQaienbGOc')
i=0
def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)


	faces = []
	locs = []
	preds = []


	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]


		if confidence > 0.5:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)


			faces.append(face)
			locs.append((startX, startY, endX, endY))


	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)


prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


maskNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()


while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)


		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		temperature="The temperature is : " + str(obj_temp)


		sleep(1)
		cv2.putText(frame, label+temperature, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		#cv2.putText(frame, str(obj_temp), (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.imwrite('frame'+str(i)+'.jpg',frame)
		bot.send_photo(chat_id=1269931525, photo=open('frame'+str(i)+'.jpg', 'rb'))
		os.remove('frame'+str(i)+'.jpg')


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()