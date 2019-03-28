cd ~
source .profile
workon cv
sudo modprobe bcm2835-v4l2
cd Desktop/pi-face-recognition-jeff
python trainFaces.py
python faceRecognition.py
