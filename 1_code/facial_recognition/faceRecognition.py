# python faceRecognition.py
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
print("Opening / Loading database of feature vectors...")
faces = pickle.loads(open("faces.db", "rb").read())
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
vs = VideoStream(src=0).start()
time.sleep(4)
while True:
	frame = imutils.resize(vs.read(), width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	boundingBoxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	vectors = face_recognition.face_encodings(rgb, boundingBoxes)
	names = []
	for vector in vectors:
		matches = face_recognition.compare_faces(data["encodings"],
			vector)
		name = "UNKNOWN"
		if True in matches:
			matchedIds = [i for (i, a) in enumerate(matches) if a]
			counts = {}
			for i in matchedIds:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
		names.append(name)
	for ((top, right, bottom, left), name) in zip(boundingBoxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	cv2.imshow("Frame", frame)
cv2.destroyAllWindows()
vs.stop()
