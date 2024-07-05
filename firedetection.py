import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
from IPython.display import Audio
import time
from playsound import playsound

class FireClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Classifier")

        self.image_path = None
        self.image_label = tk.Label(root)
        self.result_label = tk.Label(root, text="Result: ")

        browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        classify_button = tk.Button(root, text="Classify", command=self.classify_image)
        browse_button.pack(pady=10)
        self.image_label.pack()
        classify_button.pack(pady=10)
        self.result_label.pack()

        
    def browse_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.display_image()

            
    def display_image(self):
        image = Image.open(self.image_path)
        image = image.resize((300, 300), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo


    def play_alarm_sound(self):
        alarm_sound_path = 'C:/Users/Jadam Harika/DLPROJECT/'+"mixkit-classic-alarm-995.wav"
        #audio_data = open(alarm_sound_path, "rb").read()
        #print("Audio Started")
        #Audio(data=audio_data, autoplay=True)
        #print("Audio ended")

    # Let the sound play for a few seconds (adjust as needed)
        #time.sleep()
        try:
            playsound(alarm_sound_path)
        except Exception as e:
            print("error playing sound")


        
    def classify_image(self):
        if self.image_path:
            # Load and preprocess the image for model prediction
            model = tf.keras.models.load_model('C:/Users/Jadam Harika/DLPROJECT/'+'fire_detection_model.h5')

            # Load and preprocess the image
            img = cv2.imread(self.image_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize pixel values
            img = np.reshape(img, (1, 224, 224, 3))  # Add batch dimension

            # Make prediction
            prediction = model.predict(img)

            # Display result

            if prediction[0][0] > 0.8:
                result = "Fire detected!"
                self.play_alarm_sound()
            else:
                result = "Fire not detected!"
            self.result_label.configure(text=f"Result: {result}")

        else:
            self.result_label.configure(text="Please select an image first.")

if __name__ == "__main__":
    # Create the Tkinter window
    root = tk.Tk()
    app = FireClassifierApp(root)
    root.mainloop()
    


