# python trainFaces.py

from imutils import paths
import face_recognition
import cv2
import os
import argparse
import pickle

print("Traversing faces...")
imagesP = list(paths.list_images("dataset"))
vectors = []
names = []

for (i, path) in enumerate(imagesP):
	print("Processing feature vector of image {}/{}".format(i + 1,
		len(imagesP)))
	name = path.split(os.path.sep)[-2]
	image = cv2.imread(path)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	boxes = face_recognition.face_locations(rgb,
		model="hog")
	vecs = face_recognition.face_encodings(rgb, boxes)
	for vec in vecs:
		vectors.append(vec)
		names.append(name)
		
print("Saving trained faces...")
data = {"encodings": vectors, "names": names}
f = open("faces.db", "wb")
f.write(pickle.dumps(data))
f.close()
