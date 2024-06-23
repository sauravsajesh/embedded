import cv2
import numpy as np
import svgwrite
from PIL import Image
import io

def resize_image(image, max_dimension=800):
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(grayscale_image, 5)  # Apply median filter to reduce noise
    return blurred_image

def sobel_filter(image):
    height, width = image.shape
    pix_data = np.zeros((height, width), dtype=np.uint8)
    
    for x in range(1, width-1):
        for y in range(1, height-1):
            px = (
                -1 * image[y-1, x-1] + 1 * image[y-1, x+1] +
                -2 * image[y  , x-1] + 2 * image[y  , x+1] +
                -1 * image[y+1, x-1] + 1 * image[y+1, x+1]
            )
            
            py = (
                -1 * image[y-1, x-1] + -2 * image[y-1, x] + -1 * image[y-1, x+1] +
                1 * image[y+1, x-1] + 2 * image[y+1, x] + 1 * image[y+1, x+1]
            )
            
            pix_data[y, x] = 255 if (px**2 + py**2) > 80**2 else 0  # Adjust threshold as needed
    
    return pix_data

def get_dots_v(pix_data):
    height, width = pix_data.shape
    dots = []
    for y in range(height-1):
        row = []
        for x in range(1, width):
            if pix_data[y, x] == 255:
                x0 = x
                while x < width and pix_data[y, x] == 255:
                    x += 1
                row.append((x + x0) // 2)
        dots.append(row)
    return dots

def get_dots_h(pix_data):
    height, width = pix_data.shape
    dots = []
    for x in range(width-1):
        row = []
        for y in range(1, height):
            if pix_data[y, x] == 255:
                y0 = y
                while y < height and pix_data[y, x] == 255:
                    y += 1
                row.append((y + y0) // 2)
        dots.append(row)
    return dots

def connect_dots(dots, is_vertical):
    contours = []
    max_dist = 15  # Maximum allowed distance between dots to be considered part of the same contour
    
    for idx, dot_row in enumerate(dots):
        for i in range(len(dot_row)):
            if idx == 0:
                contours.append([(dot_row[i], idx)] if is_vertical else [(idx, dot_row[i])])
            else:
                closest = -1
                cdist = max_dist
                for j in range(len(dots[idx-1])):
                    prev_dot = dots[idx-1][j]
                    d = abs(dot_row[i] - prev_dot)
                    if d < cdist:
                        closest = prev_dot
                        cdist = d
                if cdist >= max_dist:
                    contours.append([(dot_row[i], idx)] if is_vertical else [(idx, dot_row[i])])
                else:
                    found = False
                    for contour in contours:
                        last = contour[-1]
                        if (last[0] == closest and last[1] == idx-1) if is_vertical else (last[1] == closest and last[0] == idx-1):
                            contour.append((dot_row[i], idx) if is_vertical else (idx, dot_row[i]))
                            found = True
                            break
                    if not found:
                        contours.append([(dot_row[i], idx)] if is_vertical else [(idx, dot_row[i])])
        contours = [contour for contour in contours if contour[-1][1 if is_vertical else 0] >= idx-1 or len(contour) >= 4]
    return contours

def simplify_contour(contour, epsilon=1.0):
    contour_np = np.array(contour, dtype=np.float32)
    simplified_np = cv2.approxPolyDP(contour_np, epsilon, False)
    return [(int(point[0][0]), int(point[0][1])) for point in simplified_np]

def convert_to_line_drawing(image):
    preprocessed_image = preprocess_image(image)
    edges = sobel_filter(preprocessed_image)
    
    dots_h = get_dots_h(edges)
    dots_v = get_dots_v(edges)
    
    contours_h = connect_dots(dots_h, False)
    contours_v = connect_dots(dots_v, True)
    
    contours = contours_h + contours_v
    
    # Remove small contours (noise)
    min_contour_length = 5  # Minimum number of points in a contour
    contours = [contour for contour in contours if len(contour) >= min_contour_length]
    
    # Simplify contours
    epsilon = 1  # Adjust the approximation accuracy as needed
    simplified_contours = [simplify_contour(contour, epsilon) for contour in contours]
    
    return simplified_contours

def draw_contours(image, contours):
    output_image = np.ones_like(image) * 255
    for contour in contours:
        for i in range(len(contour)-1):
            cv2.line(output_image, contour[i], contour[i+1], (0, 0, 0), 1)
    return output_image

def save_contours_as_svg(contours, image_size, output_path):
    width, height = image_size
    
    dwg = svgwrite.Drawing(output_path, size=(width, height))
    for contour in contours:
        points = [(point[0], point[1]) for point in contour]
        dwg.add(dwg.polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none'))
    dwg.save()

if __name__ == "__main__":
    input_image_path = 'j.jpg'
    output_svg_path = 'output.svg'

    # Load and resize the image
    image = cv2.imread(input_image_path)
    resized_image = resize_image(image)
    
    # Use the resized image for processing
    contours = convert_to_line_drawing(resized_image)
    
    output_image = draw_contours(resized_image, contours)
    output_image_path = 'line_drawing.png'
    cv2.imwrite(output_image_path, output_image)
    
    # Use the resized dimensions for SVG output
    save_contours_as_svg(contours, resized_image.shape[1::-1], output_svg_path)