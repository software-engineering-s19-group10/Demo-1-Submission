import cv2, numpy, logging, base64, io, binascii
from websocket_server import WebsocketServer
from time import sleep
from PIL import Image

PORT = 8080

# Testing purposes
def cap():
    cap = cv2.VideoCapture(0)
    print (2)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        print(3)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

def start_sending(client, server):
    # Create a VideoSender and start sending video
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("Could not open video feed")
        return

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    
    print("Connected")


    print("Sending Data Now")
    # Keep sending forever until ???
    while True:
        sleep(1)
        # Get the frame
        # Capture frame-by-frame
        ret, frame = vid.read()

        if ret:
            # Convert to bytes
            pil_im = Image.fromarray(frame)
            b = io.BytesIO()
            pil_im.save(b, 'jpeg')
            im_bytes = b.getvalue()

            # Send the frame
            server.send_message(client, binascii.b2a_base64(im_bytes))

            # Throttling to 0.1 FPS
            # sleep(0.01)
        else:
            continue


# TODO: Program will take one command line argument which will be the auth key
if __name__=="__main__":
    # cap()
    server = WebsocketServer(PORT)
    server.set_fn_new_client(start_sending)
    server.run_forever()
