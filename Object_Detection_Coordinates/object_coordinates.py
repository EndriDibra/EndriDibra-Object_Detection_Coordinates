# importing libraries
import cv2
import cvzone
import numpy as np
import time
import threading
from flask import Flask
import lxml.etree as ET
from flask import Response


# creating a Flask app
app = Flask(__name__)

# lists for X, Y coordinates
x_list, y_list = [], []


def camera_size(camera, width, height):

    camera.set(3, width)
    camera.set(4, height)


def capture_frame(camera):

    success, frame = camera.read()

    if not success:
        
        raise ValueError("ERROR! Camera failed to be opened.")
    
    return frame


def detect_circles(gray_frame):

    return cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.2, 100)


def frame_processing(frame):

    frame = cv2.blur(frame, (3, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = detect_circles(gray)

    return frame, gray, circles


def frame_using_circles(frame, circles):

    output = frame.copy()

    if circles is not None:

        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    return output


def fps_visualization(image, fps: float) -> np.ndarray:

    if len(np.shape(image)) < 3:

        text_color = (255, 255, 255)  # white

    else:

        text_color = (0, 255, 0)  # green

    # pixels
    row_size = 20 
    
    # pixels 
    left_margin = 24  
    
    font_size = 1
    font_thickness = 1

    fps_text = 'FPS: {:.1f}'.format(fps)
    text_location = (left_margin, row_size)

    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)

    return image


def color_detection(frame):
    
    global x_list, y_list

    output = frame.copy()

    output = cv2.resize(output, (640, 480))

    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    result = cv2.bitwise_and(output, output, mask=mask_blue)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_blue:

        contour_area = cv2.contourArea(contour)

        if contour_area > 1000:
            
            x, y, w, h = cv2.boundingRect(contour)
            
            bbox = int(x), int(y), int(w), int(h)
            
            cvzone.cornerRect(output, bbox)
            
            cv2.putText(output, "Blue", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Store coordinates of the detected object
            x_list.append(x)
            y_list.append(y)

    cv2.imshow("Color1", output)
    cv2.imshow("Color2", result)


def main():

    global x_list, y_list, camera

    try:
        
        # opening the default camera
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # checking if camera works
        if not camera.isOpened():

            raise ValueError("ERROR! Failed to open camera.")

        camera_size(camera, 320, 240)

        # looping through camera frames
        while camera.isOpened():

            global x_list, y_list

            start_time = time.time()

            frame = capture_frame(camera)
            frame, _, circles = frame_processing(frame)

            output = frame_using_circles(frame, circles)

            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds

            # Overlay FPS and display frames
            cv2.imshow("Frame", np.hstack([fps_visualization(frame, fps), fps_visualization(output, fps)]))

            # recognizes color
            color_detection(frame)

            # closing opned windows by pressing letter/key q:quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                
                break

    except Exception as e:

        print(e)

    finally:

        cv2.destroyAllWindows()
        camera.release()

# first page of flask server
@app.route("/")
def index():

    return ("Hello! Please add /get_coordinates on your url address "
            "There you will find the X,Y coordinates of the detected ball")


# second and main page of flask server, there are stored
# the x,y coordinates of the detected circle/ball
@app.route('/get_coordinates', methods=['GET'])
def get_coordinates():
   
    global x_list, y_list

    # saving x,y coordinates from their list
    x_coordinates = [int(x) for x in x_list]
    y_coordinates = [int(y) for y in y_list]

    # creating a XML file to store x,y cooridinates under X,Y columns
    xml_data = b'<?xml version="1.0" encoding="UTF-8"?>\n'
    
    xml_data += b'<coordinates>\n'
    
    # parsing and stroring the x,y coordinates in their corresponding columns
    for x, y in zip(x_coordinates, y_coordinates):
    
        xml_data += f'<point><x>{x}</x><y>{y}</y></point>\n'.encode('utf-8')
    
    xml_data += b'</coordinates>'

    # loading the XML data
    xml_tree = ET.fromstring(xml_data)

    # loading the XSLT stylesheet
    xslt_file = "style.xsl"
    xslt_tree = ET.parse(xslt_file)

    # creating an XSLT processor and apply the transformation
    transform = ET.XSLT(xslt_tree)
    html_tree = transform(xml_tree)

    # here is the output, of the transformed HTML
    html_output = ET.tostring(html_tree, pretty_print=True)

    return Response(html_output, mimetype='text/html')

# running process
if __name__ == '__main__':

    # Start a thread to capture frames and detect the ball
    capture_thread = threading.Thread(target=main)
   
    # Set the thread as daemon
    capture_thread.daemon = True
    capture_thread.start()

    # Run the Flask app, and is available to all connected devices of the sames network
    app.run(host='0.0.0.0', port=8000)