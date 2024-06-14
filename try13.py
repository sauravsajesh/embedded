import argparse
import time
from PIL import Image
import io
import base64
import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import svgwrite
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_to_line_drawing(image):
    # Convert PIL Image to OpenCV image
    open_cv_image = np.array(image)
    # Convert RGB to BGR (OpenCV uses BGR format)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    grayscale_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    line_art = 255 - dilated_edges
    
    # Convert the image to a PNG format in memory
    success, encoded_image = cv2.imencode('.png', line_art)
    if not success:
        raise Exception("Failed to encode image to PNG format")

    # Convert to base64 string
    base64_image = base64.b64encode(encoded_image).decode('utf-8')
    
    return base64_image

def draw_objects_svg(objs, labels, image, image_size):
    """Draws the object images with background removed and their labels on an SVG image."""
    dwg = svgwrite.Drawing(size=image_size)
    for obj in objs:
        bbox = obj.bbox
        # Crop the object from the image
        cropped_image = image.crop((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))
        # Convert the cropped image to a base64 string
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()
        
        # Create a data URL for the image
        img_data_url = f"data:image/png;base64,{encoded_image}"
        
        # Embed the image in the SVG
        dwg.add(dwg.image(href=img_data_url, insert=(bbox.xmin, bbox.ymin), size=(bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin)))
        
        # Add the label
        #dwg.add(dwg.text(f"{labels.get(obj.id, obj.id)}: {obj.score:.2f}", insert=(bbox.xmin, bbox.ymin - 10),
                      #   fill='red', font_size='10'))
    
    return dwg

def run_inference(model_path, image_path, labels_path, threshold=0.4, count=5):
    labels = read_label_file(labels_path) if labels_path else {}
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    image = Image.open(image_path)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes loading the model into Edge TPU memory.')
    for _ in range(count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, threshold, scale)
        print('%.2f ms' % (inference_time * 1000))

    print('-------RESULTS--------')
    if not objs:
        print('No objects detected')

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

    return objs, labels, image, image.size

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            svg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.svg')
            file.save(input_path)
            
            # Use the paths and parameters from argparse
            model_path = args.model
            labels_path = args.labels
            threshold = args.threshold
            count = args.count
            
            objs, labels, image, image_size = run_inference(model_path, input_path, labels_path, threshold, count)
            
            if objs:
                line_drawing_base64 = convert_to_line_drawing(image)
                line_drawing_image = Image.open(io.BytesIO(base64.b64decode(line_drawing_base64)))
                dwg = draw_objects_svg(objs, labels, line_drawing_image, image_size)
                dwg.saveas(svg_path)
            
            return render_template('index.html', 
                                   original_image=url_for('uploaded_file', filename=filename),
                                   svg_image=url_for('uploaded_file', filename='output.svg'))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file')
    parser.add_argument('-l', '--labels', help='File path of labels file')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Score threshold for detected objects')
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of times to run inference')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
