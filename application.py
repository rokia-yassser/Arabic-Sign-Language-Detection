from io import BytesIO
from pygame import mixer
import customtkinter as ctk
from realtime_detection import update_frame, release_video  
from gtts import gTTS
from playsound import playsound
import os
import warnings
warnings.filterwarnings("ignore")

# Initialize customtkinter with a theme and color mode
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create the main window
root = ctk.CTk()
root.title("Webcam Viewer with Prediction")

# Set window size and position
w, h = 720, 800
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - w) // 2
y = (screen_height - h) // 2
root.geometry(f"{w}x{h}+{x}+{y}")

# Add a label to display the webcam feed within the application window
video_label = ctk.CTkLabel(root, text="")
video_label.pack(pady=10)

# Add a text area for displaying the predicted character sentence


def on_speak_button_click():
    text_to_speak = text_area.get("1.0", ctk.END)  # Get the text from the text area
    if text_to_speak:
        mp3_fp = BytesIO()
        tts = gTTS(text_to_speak, lang='ar')
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        mixer.init()
        mixer.music.load(mp3_fp, "mp3")
        mixer.music.play()
    else:
        print("No text to speak.")

# Start the video stream within the application

speak_button = ctk.CTkButton(root, text="Speak Text", command=on_speak_button_click)
speak_button.pack(pady=10)

text_area = ctk.CTkTextbox(root, width=530, height=200, font=("Arial", 40, "bold"))
text_area.pack(pady=10)

update_frame(video_label, text_area)

# Start the main application loop
root.mainloop()

# Release the video capture when the window is closed
release_video()
