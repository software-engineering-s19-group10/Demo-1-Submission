
// Set port and URL
PORT = 8080;
URL = "7de6ea1a.ngrok.io";

// Create the socket
console.log("Creating Socket.");
const socket = new WebSocket('ws://' + URL);
console.log("Socket Created.");


// Event Handler for opening the connection
socket.onopen = function(event) {
    //console.log("Connected.");
}

/*
// function to convert to base 64. got off stackoverflow... as expected
function b64EncodeUnicode(
    // first we use encodeURIComponent to get percent-encoded UTF-8,
    // then we convert the percent encodings into raw bytes which
    // can be fed into btoa.
    console.log("Converting to base 64.");
    return btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g,
        function toSolidBytes(match, p1) {
            return String.fromCharCode('0x' + p1);
    }));
}
*/

window.onload = function(event) {
    console.log("Window loaded");

    let imgelement = document.getElementById("latestImage")
    socket.onmessage = function(event) {
        console.log("Frame received.");

        // get frame as a string 
        let frameSTR = event.data;        

        // convert to base 64
        // apparently not necessary
        //let jpgbuffer = b64EncodeUnicode(frameSTR);
        //console.log("Frame converted to base 64.");

        console.log("Adding JPEG headers");
        var datajpg = "data:image/jpg;base64," + frameSTR;
        console.log("JPEG Headers Added.");

        imgelement.src = datajpg;

    }
}


window.onbeforeunload = function(event) {
    console.log("Closing connection.");

    socket.close();
}
