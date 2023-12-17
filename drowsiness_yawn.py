from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# Function to sound alarm or speak a message
def sound_alarm(message):
    global alarm_active
    global alarm_active2
    global speaking

 # If alarm is active, continuously sound the alarm
    while alarm_active:
        print('Calling...')
        speech_command = 'espeak "' + message + '"'
        os.system(speech_command)

  # If the second alarm is active and not currently speaking, deliver the message
    if alarm_active2:
        print('Calling...')
        speaking = True
        speech_command = 'espeak "' + message + '"'
        os.system(speech_command)
        speaking = False

def calculate_eye_aspect_ratio(eye_landmarks):
    # Calculate the Euclidean distances between specific landmarks (points) of the eye
    # Vertical eye landmark 1 to 5
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5]) 

    # Vertical eye landmark 2 to 4
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4]) 

    # Horizontal eye landmark 0 to 3
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])  

    # Compute the eye aspect ratio using the formula
    eye_ratio = (A + B) / (2.0 * C)  # Formula for calculating eye aspect ratio
    return eye_ratio

def calculate_final_eye_aspect_ratio(facial_shape):
    # Get the start and end indices of the left and right eye landmarks in the facial shape
    left_eye_start, left_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start, right_eye_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # Extract left and right eye landmarks from the facial shape
    left_eye_landmarks = facial_shape[left_eye_start:left_eye_end]
    right_eye_landmarks = facial_shape[right_eye_start:right_eye_end]

    # Calculate the eye aspect ratio for the left and right eyes separately
    left_eye_ratio = calculate_eye_aspect_ratio(left_eye_landmarks)
    right_eye_ratio = calculate_eye_aspect_ratio(right_eye_landmarks)

    # Compute the average eye aspect ratio of both eyes
    eye_aspect_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

    # Return the calculated eye aspect ratio along with left and right eye landmarks
    return (eye_aspect_ratio, left_eye_landmarks, right_eye_landmarks)

def calculate_lip_distance(facial_shape):
    # Extract specific landmarks for the top and bottom lip from the facial shape
    # Landmarks for the upper lip (50-52)
    top_lip = facial_shape[50:53]  

    # Additional landmarks for the upper lip (61-63)
    top_lip = np.concatenate((top_lip, facial_shape[61:64]))  

    # Landmarks for the lower lip (56-58)
    bottom_lip = facial_shape[56:59] 

    # Additional landmarks for the lower lip (65-67)
    bottom_lip = np.concatenate((bottom_lip, facial_shape[65:68]))  

    # Calculate the mean (average) points for the top and bottom lips
    # Mean point for the upper lip
    top_mean = np.mean(top_lip, axis=0) 
    # Mean point for the lower lip
    bottom_mean = np.mean(bottom_lip, axis=0)  

    # Calculate the vertical distance between the mean points of the top and bottom lips
    # Absolute difference in y-coordinates
    distance = abs(top_mean[1] - bottom_mean[1]) 
    return distance

# Command-line arguments setup
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

# Constants for eye and yawn detection thresholds
EYE_AR_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESHOLD = 17
alarm_active = True
alarm_active2 = True
speaking = True
FRAME_COUNTER = 0

# Loading the face detector and facial landmark predictor
print("-> Loading the predictor and detector...")
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Starting video stream
print("-> Starting Video Stream")
video_stream = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = video_stream.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
detected_faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, 
    minNeighbors=5, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)

# Loop through each detected face
for (x, y, w, h) in detected_faces:
    # Create a rectangle representing the face
    face_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    
    # Detect facial landmarks
    facial_shape = shape_predictor(gray, face_rect)
    facial_shape = face_utils.shape_to_np(facial_shape)

    # Calculate eye aspect ratio and lip distance for the current face
    eye = calculate_final_eye_aspect_ratio(facial_shape)
    eye_aspect_ratio = eye[0]
    left_eye = eye[1]
    right_eye = eye[2]

    lip_distance_value = calculate_lip_distance(facial_shape)

    # Visualize eye and lip contours on the frame
    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

    lip_contour = facial_shape[48:60]  # Landmarks for lip contour
    cv2.drawContours(frame, [lip_contour], -1, (0, 255, 0), 1)  # Draw lip contour on the frame

  # Check for drowsiness and yawn alerts based on thresholds
  # Check if the calculated eye aspect ratio falls below the predefined threshold

if eye_aspect_ratio < EYE_AR_THRESHOLD:
    # Increment the frame counter for consecutive frames with low eye aspect ratio
    FRAME_COUNTER += 1 

    # Check if the frame counter meets or exceeds the threshold for drowsiness detection
    if FRAME_COUNTER >= EYE_AR_CONSEC_FRAMES:
        # Activate the alarm if it's not already active and start a thread to sound the alarm
        if not alarm_active:
            alarm_active = True
            alarm_thread = Thread(target=sound_alarm, args=('Wake up, please!',))
            alarm_thread.daemon = True
            alarm_thread.start()

        # Display a drowsiness alert on the frame
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	else:
		# Reset the frame counter if eye aspect ratio is above the threshold
    		FRAME_COUNTER = 0  
		
		# Deactivate the alarm as the condition for drowsiness is not met
   	        alarm_active = False  

# Check if the calculated lip distance value exceeds the predefined yawn threshold
if lip_distance_value > YAWN_THRESHOLD:
    # Display a "Yawn Alert" message on the frame if a yawn is detected
    cv2.putText(frame, "Yawn Alert", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Check conditions for triggering a fresh air alarm
    if not alarm_active2 and not speaking:
        # Activate the second alarm and start a thread to sound the alarm message
        alarm_active2 = True
        alarm_thread = Thread(target=sound_alarm, args=('Take some fresh air, please!',))
        alarm_thread.daemon = True
        alarm_thread.start()
else:
    alarm_active2 = False  # Deactivate the second alarm if yawn condition is not met


# Display eye aspect ratio and lip distance values on the frame
        cv2.putText(frame, "EAR: {:.2f}".format(eye_aspect_ratio), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(lip_distance_value), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

# Exit loop when 'q' key is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video_stream.stop()
