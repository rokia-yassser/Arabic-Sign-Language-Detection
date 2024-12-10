import cv2
import numpy as np
import mediapipe as mp
import pickle
from PIL import Image, ImageTk
from gtts import gTTS
import playsound 
import os

# Load the Random Forest model and labels
labels = {  
"Alef": "Alef",
"Beh":"Beh",  
"Teh": "Teh",  
"Theh": "Theh",  
"Jeem": "Jeem",  
"Hah": "Hah",  
"Khah": "Khah",  
"Dal": "Dal",  
"Thal": "Thal",  
"Reh": "Reh",  
"Zain": "Zain",  
"Seen": "Seen",  
"Sheen": "Sheen",  
"Sad": "Sad",  
"Dad": "Dad",  
"Tah": "Tah",  
"Zah": "Zah",  
"Ain": "Ain",  
"Ghain": "Ghain",  
"Feh": "Feh",  
"Qaf": "Qaf",  
"Kaf": "Kaf",  
"Lam": "Lam",  
"Meem": "Meem",  
"Noon": "Noon",  
"Heh": "Heh",  
"Waw": "Waw",  
"Yeh": "Yeh",
"Al":"Al",
"Laa":"Laa",
"Teh_Marbuta":"Teh_Marbuta",
"1": "Back Space",
"2": "Clear", 
"3": "Space",
"4": ""
}

letter_map = {  
"Alef": "ا",
"Beh":"ب",  
"Teh": "ت",  
"Theh": "ث",  
"Jeem": "ج",  
"Hah": "ح",  
"Khah": "خ",  
"Dal": "د",  
"Thal": "ذ",  
"Reh": "ر",  
"Zain": "ز",  
"Seen": "س",  
"Sheen": "ش",  
"Sad": "ص",  
"Dad": "ض",  
"Tah": "ط",  
"Zah": "ظ",  
"Ain": "ع",  
"Ghain": "غ",  
"Feh": "ف",  
"Qaf": "ق",  
"Kaf": "ك",  
"Lam": "ل",  
"Meem": "م",  
"Noon": "ن",  
"Heh": "ه",  
"Waw": "و",  
"Yeh": "ي",
"Al":"ال",
"Laa":"لا",
"Teh_Marbuta":"ة"
}

with open("./ASL_model.p", "rb") as f:
    model = pickle.load(f)

rf_model = model["model"]

# Initialize Mediapipe components
mp_hands = mp.solutions.hands  # Hand tracking solution
mp_drawing = mp.solutions.drawing_utils  # Drawing utility
mp_drawing_styles = mp.solutions.drawing_styles  # Pre-defined drawing styles

# Configure the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,  # Use dynamic mode for video streams
    max_num_hands=1,  # Track at most one hand
    min_detection_confidence=0.9  # Set a high detection confidence threshold
)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Strings to store the concatenated sentence
predicted_text = " "
same_characters = ""
final_characters = ""
s = ""
old = ""
new = ""
count = 0

def speak_text(text):
    tts = gTTS(text=text, lang='ar')  # Adjust language as needed
    tts.save("speech.mp3")
    playsound.playsound("speech.mp3")

# Function to update each frame and predict the character
def update_frame(video_label, text_area):
    global predicted_text, same_characters, final_characters, count, old, new
    ret, frame = cap.read()  # Capture frame-by-frame
    if ret:
        # Process the frame to display hand landmarks and predict the character
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = hands.process(frame_rgb)
        hand_landmarks = processed_image.multi_hand_landmarks
        height, width, _ = frame.shape

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect landmark coordinates for prediction
                x_coordinates = [landmark.x for landmark in hand_landmark.landmark]
                y_coordinates = [landmark.y for landmark in hand_landmark.landmark]
                min_x, min_y = min(x_coordinates), min(y_coordinates)
                
                normalized_landmarks = []
                for coordinates in hand_landmark.landmark:
                    normalized_landmarks.extend([
                        coordinates.x - min_x,
                        coordinates.y - min_y
                    ])
                
                # Predict the character using the model
                sample = np.asarray(normalized_landmarks).reshape(1, -1)
                predicted_character = rf_model.predict(sample)[0]

                if predicted_character != "4":
                    predicted_text += predicted_character
                    
                    for i in range(len(predicted_text)-1, -1, -1):
                                if predicted_text[i].isupper():
                                    if(predicted_text[i:] == "Marbuta"):
                                        new = "Teh_Marbuta"
                                        break
                                    else:
                                        new = predicted_text[i:]  # Print from the capital letter to the end of the string
                                        break
                                elif(predicted_text[i] == '1' or predicted_text[i] == '2' or predicted_text[i] == '3' or predicted_text[i] == '4'):
                                    new = predicted_text[i]
                                    break
                    
                    # Append the predicted character to the sentence
                    if (old != new): 
                        count = 0
                        same_characters = ""
                        old = new
                    else:
                        same_characters += predicted_character
                        count += 1
                    
                    # Display the concatenated sentence in the text area
                    if count == 10: 

                        if predicted_character == "1":
                            if final_characters:
                                final_characters = list(final_characters)
                                final_characters.pop()
                                final_characters = "".join(final_characters)
                                text_area.delete("1.0", 'end')
                                text_area.insert("1.0", final_characters)

                        elif predicted_character == "2":
                            final_characters = ""
                            text_area.delete("1.0", 'end')

                        elif predicted_character == "3":
                            final_characters += " "
                            text_area.delete("1.0", 'end')
                            text_area.insert("1.0", final_characters)

                        else:                            
                            for i in range(len(predicted_text)-1, -1, -1):
                                if predicted_text[i].isupper():
                                    if(predicted_text[i:] == "Marbuta"):
                                        s = "Teh_Marbuta"
                                        break
                                    else:
                                        s = predicted_text[i:]  # Print from the capital letter to the end of the string
                                        break
                                    
                            final_characters += letter_map.get(s)
                            text_area.delete("1.0", 'end')
                            text_area.insert("1.0", final_characters)
                            s = ""

                        count = 0
                        same_characters = ""
                        

                    # Coordinates and colors
                    text_position = (20, 20)  # Top-left corner of the text background
                    background_color = (0, 150, 250)  # Background color (orange)
                    text_color = (0, 0, 0)  # Text color (black)
                    font_scale = 1
                    thickness = 2

                    # Calculate the width and height of the text box
                    (text_width, text_height), baseline = cv2.getTextSize(predicted_character, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    # Calculate bottom-right corner for the background rectangle based on text size
                    background_top_left = text_position
                    background_bottom_right = (text_position[0] + text_width + 180, text_position[1] + text_height + 10)

                    # Draw the filled rectangle as the background for text
                    cv2.rectangle(frame, background_top_left, background_bottom_right, background_color, -1)

                    # Draw the text on top of the rectangle
                    cv2.putText(
                        img=frame,
                        text=labels[predicted_character],
                        org=(text_position[0] + 5, text_position[1] + text_height),  # Adjust for padding within rectangle
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale,
                        color=text_color,
                        thickness=thickness,
                        lineType=cv2.LINE_AA
                    )

        # Convert the frame to ImageTk format and update the label
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
        video_label.configure(image=imgtk)

    video_label.after(10, lambda: update_frame(video_label, text_area))  # Repeat every 10 ms
    #print(final_characters, count)


# Function to release the video capture
def release_video():  
    cap.release()
    cv2.destroyAllWindows()
    