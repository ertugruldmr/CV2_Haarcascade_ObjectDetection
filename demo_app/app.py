import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

options = ["eye", "eye_tree_eyeglasses", "frontalcatface", "frontalcatface_extended", 
            "frontalface_alt", "frontalface_alt2","frontalface_alt_tree",
            "frontalface_default","fullbody", "lefteye_2splits", "lowerbody",
            "profileface", "righteye_2splits", "smile", "upperbody"]


def detect(img, option):

  # preparing the corresponding file
  file_path = f"haarcascade_{option}.xml"
  valid_model_path = cv2.data.haarcascades + file_path

  # preparingt the image
  old_size = img.shape[:2]
  img = cv2.resize(img, (400, 600))
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  green, thickness = (0, 255, 0), 2

  # loading the model
  haarcascader = cv2.CascadeClassifier(valid_model_path)

  # detection
  detections = haarcascader.detectMultiScale(gray)

  # Drawing the ROI
  for (x, y, w, h) in detections:

    l_t, r_b = (x, y),(x+w, y+h)
    img = cv2.rectangle(img, l_t, r_b, green, thickness)

  
  img = cv2.resize(img, old_size)

  return img

# Creating the GUI components of demo app
with gr.Blocks() as demo:
    
    # creating the components
    img = gr.Image(type="numpy", source="upload")
    option = gr.Dropdown(choices = options, value="frontalface_default", label="Detector")
    detect_btn = gr.Button("Detect")
    output = gr.Image(type="numpy")

    # connecting the button functions
    detect_btn.click(detect, inputs=[img, option], outputs=output)

    # setting the exmples
    gr.Examples("examples", img)
            
# Launching the demo
if __name__ == "__main__":
    demo.launch()
