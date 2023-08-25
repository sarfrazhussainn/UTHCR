import numpy as np
from tkinter import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import ImageGrab
from tkinter import filedialog

class_names = {
    0: 'அ',
    1: 'ஆ',
    2: 'ஓ',
    3: 'ஃ',
    4: 'இ',
    5: 'ஈ',
    6: 'உ',
    7: 'ஊ',
    8: 'எ',
    9: 'ஏ',
    10: 'ஐ',
    11: 'ஒ'
}

model = load_model('UTHCR.h5')

def paint(event):
    brush_size = 7
    brush_color = 'black'
    
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill=brush_color, outline=brush_color)

def clear():
    canvas.delete("all")

def predict():
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab((x, y, x1, y1)).resize((64, 64)).convert('L')
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    
    probs = model.predict(x)
    
    predicted_class = np.argmax(probs)
    predicted_class_name = class_names[predicted_class]
    result_label.configure(text=f'Predicted Tamil Character is: {predicted_class_name} ({probs[0][predicted_class]:.4f})')
    output_box.delete(0, END)
    output_box.insert(0, predicted_class_name)

def import_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.tiff")])
    if file_path:
        img = image.load_img(file_path, target_size=(64, 64), color_mode='grayscale')
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        probs = model.predict(x)

        predicted_class = np.argmax(probs)
        predicted_class_name = class_names[predicted_class]
        result_label.configure(text=f'Predicted class: {predicted_class_name} ({probs[0][predicted_class]:.4f})')
        output_box.delete(0, END)
        output_box.insert(0, predicted_class_name)
        
root = Tk()
root.geometry('400x450')
root.title('Handwritten Tamil Character Recognition')

title_box = Label(root, text='Unconstrained Tamil Handwritten Character Recognition', font=('Arial', 20, 'bold'), pady=10)
title_box.pack(side=TOP)

canvas = Canvas(root, width=1000, height=650, bg='white')
canvas.pack(side=TOP)

button_frame = Frame(root)
button_frame.pack(side=TOP)

clear_button = Button(button_frame, text='Clear', font=('Arial', 14, 'bold'), command=clear, padx=10, pady=5, bg='white', fg='black')
clear_button.pack(side=LEFT, padx=20)

predict_button = Button(button_frame, text='Predict', font=('Arial', 14, 'bold'), command=predict, padx=10, pady=5, bg='white', fg='black')
predict_button.pack(side=LEFT, padx=20)

import_button = Button(button_frame, text='Import', command=import_image, font=('Arial', 14, 'bold'), padx=10, pady=5, bg='white', fg='black')
import_button.pack(side=LEFT)

result_label = Label(root, text='', pady=10)
result_label.pack(side=BOTTOM)

output_box = Entry(root, font=('Arial', 30), width=8)
output_box.pack(side=BOTTOM)

canvas.bind('<B1-Motion>', paint)
root.mainloop()