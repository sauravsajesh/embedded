from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import svgwrite
import base64
import io
import os
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects

app = Flask(__name__)

# Initialize the webcam capture object
cap = cv2.VideoCapture(0)

# Load Edge TPU model
model_path = 'path/to/your/model.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

@app.route('/')
def index():
    return render_template('index.html')

def process_image(image):
    # Run object detection on the image
    _, scale = input_size(interpreter)
    resized_image = cv2.resize(image, (scale.width, scale.height))
    common.set_input(interpreter, resized_image)
    interpreter.invoke()
    objs = get_objects(interpreter, 0.4)  # Adjust score threshold as needed

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for obj in objs:
        bbox = obj.bbox.scale(scale.width / image.shape[1], scale.height / image.shape[0])
        x0, y0, x1, y1 = bbox
        mask[y0:y1, x0:x1] = 255

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    grayscale_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    line_art = 255 - dilated_edges

    return line_art

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            return "Failed to capture image from webcam"

        # Save captured image as JPEG for debugging
        cv2.imwrite('static/captured_image.jpg', frame)

        # Process captured image to SVG
        line_art = process_image(frame)
        cv2.imwrite('static/line_drawing.png', line_art)

        # Redirect to display SVG
        return redirect(url_for('display_svg'))
    except Exception as e:
        print("Error in capture route:", e)
        return redirect(url_for('index'))

@app.route('/display_svg')
def display_svg():
    try:
        input_png_path = 'static/line_drawing.png'
        output_svg_path = 'static/output.svg'

        # Convert PNG to SVG
        png_to_svg(input_png_path, output_svg_path)

        return render_template('display_svg.html', svg_path=output_svg_path)
    except Exception as e:
        print("Error in display_svg route:", e)
        return redirect(url_for('index'))

def png_to_svg(input_path, output_path):
    with Image.open(input_path) as img:
        width, height = img.size
        img = img.convert('RGBA')
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        dwg = svgwrite.Drawing(output_path, size=(width, height))
        img_svg = dwg.image(href=f"data:image/png;base64,{img_base64}", size=(width, height))
        dwg.add(img_svg)
        dwg.save()

if __name__ == "__main__":
    app.run(debug=True)

