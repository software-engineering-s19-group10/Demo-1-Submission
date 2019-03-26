# Video Streaming for Smart Lock 
## How to use
The program for streaming will run continuously on the Pi
and will receive a request for the video at port 8080. 

To request streaming,send the authorization key (see the docs for the server and database) only to 
port 8080 on the Pi. NOTE: The auth key isn't implemented yet so ignore this.

The Pi should start immediately sending the video to the port and IP
address that the request was received from. 

To terminate the video feed, just close the window.
