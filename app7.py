import os
import cv2
import torch
import numpy as np
from pygame import mixer
from torchvision import transforms
from torch import nn
from PIL import Image  # Import PIL for image conversion

# Initialize pygame mixer for sound (optional)
try:
    mixer.init()
    sound = mixer.Sound('alarm.wav')
except Exception as e:
    sound = None  # If sound cannot be initialized, set to None

# Load Haar Cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Define the model architecture for eye state classification
class EyeBlinkModel(nn.Module):
    def __init__(self):
        super(EyeBlinkModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)  
        self.fc2 = nn.Linear(128, 4)  # Change to match the number of classes used during training

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        return x

# Initialize the model and load pre-trained weights
model = EyeBlinkModel()
try:
    model.load_state_dict(torch.load('cnnCat2.pth', map_location='cpu', weights_only=True))
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

model.eval()

# Define transformations for eye images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Score variables
score_count = 0  # Count of frames where at least one eye is closed
alarm_triggered = False  # To prevent multiple alarm triggers

while True:
    ret, frame = cap.read()

    # Check if frame is successfully captured
    if not ret:
        print("Failed to capture image from camera")
        break

    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Draw background rectangle for score display
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        # Get face region for eye detection
        roi_gray_face = gray_frame[y:y + h, x:x + w]

        left_eye_coords = leye_cascade.detectMultiScale(roi_gray_face)
        right_eye_coords = reye_cascade.detectMultiScale(roi_gray_face)

        left_eye_closed = False
        right_eye_closed = False

        # Process left eye detection and classification
        for (ex, ey, ew, eh) in left_eye_coords:
            l_eye_roi = roi_gray_face[ey:ey + eh, ex:ex + ew]
            l_eye_roi_pil = Image.fromarray(l_eye_roi)  # Convert NumPy array to PIL Image
            l_eye_roi_resized = transform(l_eye_roi_pil).unsqueeze(0)  

            with torch.no_grad():
                lpred = model(l_eye_roi_resized)
                _, predicted_left_eye = torch.max(lpred.data, 1)
                left_eye_closed = predicted_left_eye.item() == 0  # Closed if prediction is 0
            break

        # Process right eye detection and classification
        for (ex, ey, ew, eh) in right_eye_coords:
            r_eye_roi = roi_gray_face[ey:ey + eh, ex:ex + ew]
            r_eye_roi_pil = Image.fromarray(r_eye_roi)  # Convert NumPy array to PIL Image
            r_eye_roi_resized = transform(r_eye_roi_pil).unsqueeze(0)  

            with torch.no_grad():
                rpred = model(r_eye_roi_resized)
                _, predicted_right_eye = torch.max(rpred.data, 1)
                right_eye_closed = predicted_right_eye.item() == 0  # Closed if prediction is 0
            break

        # Display eye states on the frame if both eyes are detected
        if left_eye_closed or right_eye_closed:
            score_count += 1
            
            if score_count >= 5 and not alarm_triggered: 
                try:
                    if sound:
                        sound.play()
                        alarm_triggered = True 
                except Exception as e:
                    print("Sound alarm would play!")
            
            state_text_left = "Closed" if left_eye_closed else "Open"
            state_text_right = "Closed" if right_eye_closed else "Open"
            
            cv2.putText(frame,f'Left Eye: {state_text_left}', (x + ex -10 , y -10),
                        cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,255 ,255), 1)
            cv2.putText(frame,f'Right Eye: {state_text_right}', (x + ex -10 , y -30),
                        cv2.FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,255 ,255), 1)

        else:
            score_count -= max(1 /30.0 , score_count)   # Decrease time if eyes are open.
            alarm_triggered=False  

        score_text_position_y=height -20  
        cv2.putText(frame,'Score:' + str(max(score_count ,0)), (10 , score_text_position_y), font ,1 ,(255 ,255 ,255) ,1 ,cv2.LINE_AA)

    # Display the resulting frame with bounding boxes and labels 
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed 
    key_pressed = cv2.waitKey(1) & 0xFF  
    if key_pressed == ord('q'):
        break 

cap.release() 
cv2.destroyAllWindows()
