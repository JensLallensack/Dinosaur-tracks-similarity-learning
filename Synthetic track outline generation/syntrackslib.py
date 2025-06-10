
import drawsvg as draw
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import svgwrite
import vtracer
import xml.etree.ElementTree as ET
from PIL import Image
import cairosvg
import math
import os
import shutil
import copy
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, wsvg, parse_path
import svgpathtools as sp
from scipy.signal import savgol_filter
import xml.etree.ElementTree as ET

import warnings
warnings.filterwarnings("ignore")

####Parameters####
position_noise = 0.8
drawing_size = 800

##### Helper functions #####

def unlist(value):
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 1:
            value = value[0]
    return value

#Check if something is an actual number
def isnumber(number):
    if isinstance(number, (int, float, complex)) and not math.isnan(number):
        number = True
    else:
        number = False
    return(number)

def whentodraw(when_to_draw,i,n):
    if when_to_draw <= 0:
        if i <= n*when_to_draw:
            drawit = True
        else:
            drawit = False
    if when_to_draw >= 0:
        if i >= n*when_to_draw:
            drawit = True
        else:
            drawit = False
    return drawit

def interpad(pad1,pad2):
	diffx = abs(pad1[0]-pad2[0])
	diffy = abs(pad1[1]-pad2[1])
	newx = max(pad1[0],pad2[0])-(diffx/2)
	newy = max(pad1[1],pad2[1])-(diffy/2)
	return(newx,newy)

def interpad_radius(radius_pad1, radius_pad2, asymmetry, noise):
    #add noise to asymmetry
    asym_noisy = np.random.uniform(asymmetry - noise, asymmetry + noise)
    if(asym_noisy) <= 0:
        asym_noisy = asymmetry
    #get the radius for a new interpad that is asymmetrically interpolated from the enclosing pads
    larger_pad_radius = max(radius_pad1, radius_pad2)
    smaller_pad_radius = min(radius_pad1, radius_pad2)
    diff = larger_pad_radius-smaller_pad_radius
    diff = diff * asym_noisy
    new_interpad_radius = smaller_pad_radius+diff
    return new_interpad_radius

#Return random value. Radius: percentage, relative to image dimensions
def rand(value,radius=position_noise):
	radius_px = (drawing_size/100)*radius
	randomvalue = np.random.randint(value - radius_px, value + radius_px)
	return(randomvalue)

#Return random values of tuple (two numbers) Radius: percentage, relative to image dimensions
def rand_t(value,radius):
	x, y = tuple(value)
	radius_px = (drawing_size/100)*radius
	randomvalue_x = np.random.uniform(x - radius_px, x + radius_px)
	randomvalue_y = np.random.uniform(y - radius_px, y + radius_px)
	returntuple = (randomvalue_x, randomvalue_y)
	return(returntuple)

def rand_t_df(points, radius_x=position_noise, radius_y=position_noise):
    radius_x_px = (drawing_size / 100) * radius_x
    radius_y_px = (drawing_size / 100) * radius_y
    points['x'] = points['x'].apply(lambda x: np.random.randint(x - radius_x_px, x + radius_x_px))
    points['y'] = points['y'].apply(lambda y: np.random.randint(y - radius_y_px, y + radius_y_px))
    return points

def rand_t_individual(points,noise):
    #accepts a data frame of coordinates ("points") and the noise values (tuple).
    for l in range(len(points)):
        radius_x_px = (drawing_size / 100) * noise[l]
        radius_y_px = (drawing_size / 100) * noise[l]
        coordinatevalue = points.at[l,'x']
        points.at[l,'x'] = np.random.randint(coordinatevalue - radius_x_px, coordinatevalue + radius_x_px)
        coordinatevalue = points.at[l,'y']
        points.at[l,'y'] = np.random.randint(coordinatevalue - radius_y_px, coordinatevalue + radius_y_px)
    return points

def interpad(pad1,pad2):
	diffx = abs(pad1[0]-pad2[0])
	diffy = abs(pad1[1]-pad2[1])
	newx = max(pad1[0],pad2[0])-(diffx/2)
	newy = max(pad1[1],pad2[1])-(diffy/2)
	return(newx,newy)

def threshold(image):
	threshold_value = np.random.randint(160 - 20, 160 + 20)  # Slightly vary threshold value
	_, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
	return binary_image

def png_to_svg():
	path = vtracer.convert_image_to_svg_py("image.png", "image.svg", colormode='binary')

def svg_to_png(input_svg_path, output_png_path):
    cairosvg.svg2png(url=input_svg_path, write_to=output_png_path)

def add_noise_to_svg(input_svg_path, output_svg_path):
    # Parse the SVG file
    tree = ET.parse(input_svg_path)
    root = tree.getroot()
    # # Get the width and height of the SVG canvas
    width = int(root.attrib['width'])
    height = int(root.attrib['height'])
    # Define the noise range (5% of the total dimensions)
    noise_range = int(min(width, height) * 0.01)
    # Iterate through all path elements
    for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
        # Get the 'd' attribute representing the path data
        path_data = path.attrib['d']
        # Split the path data into commands and parameters
        commands = path_data.split()
        # Iterate through the parameters
        for i in range(len(commands)):
            # Check if the parameter is a coordinate (x or y)
            if commands[i].replace('.', '', 1).isdigit():
                # Add random noise within the noise range
                noise = random.randint(-noise_range, noise_range)
                commands[i] = str(float(commands[i]) + noise)
        # Join the modified commands back into path data
        modified_path_data = ' '.join(commands)
        # Update the 'd' attribute with the modified path data
        path.attrib['d'] = modified_path_data
    # Write the modified SVG file
    tree.write(output_svg_path)


def invert_image(input_image_path, output_image_path):
    # Load image
    img = Image.open(input_image_path)
    # Convert image to grayscale if it's not already
    img = img.convert('L')
    # Invert colors
    inverted_img = Image.eval(img, lambda x: 255 - x)
    # Save inverted image
    inverted_img.save(output_image_path)


def fill_holes(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Fill each contour with black
    for contour in contours:
        cv2.fillPoly(gray, [contour], color=0)
    # Save the result
    cv2.imwrite(output_path, gray)


def white_to_alpha(input_image_path, output_image_path):
    # Open image
    img = Image.open(input_image_path)
	# Calculate crop dimensions
    width, height = img.size
    left = int(width * 0.05)
    top = int(height * 0.05)
    right = int(width * 0.95)
    bottom = int(height * 0.95)
    img = img.crop((left, top, right, bottom))
    # Convert image to RGBA if not already
    img = img.convert("RGBA")
    # Get pixel data
    pixdata = img.load()
    # Replace white pixels (r > 240, g > 240, b > 240 is a common threshold for white)
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y][0] > 240 and pixdata[x, y][1] > 240 and pixdata[x, y][2] > 240:
                pixdata[x, y] = (255, 255, 255, 0)  # Set to fully transparent
    # Save modified image
    img.save(output_image_path)

def alpha_to_white(image_path, output_path):
    # Load the image with alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Split the image into its channels
    b, g, r, a = cv2.split(img)
    # Since the image is grayscale, all channels b, g, r are the same
    # We can just use one of them, say b (or r or g), since they are identical
    gray = b
    # Create a white background image
    white_background = np.ones_like(gray, dtype=np.uint8) * 255
    combined = cv2.add(gray, white_background, mask=a)
    # Save the result
    cv2.imwrite(output_path, combined)

def segment_to_points(segment):
    """Convert a segment to a list of points."""
    if isinstance(segment, Line):
        return [segment.start, segment.end]
    elif isinstance(segment, CubicBezier):
        return [segment.start, segment.control1, segment.control2, segment.end]
    elif isinstance(segment, QuadraticBezier):
        return [segment.start, segment.control, segment.end]
    else:
        return []

def smooth_path(path, window_length, polyorder):
	points = np.concatenate([seg.point(np.linspace(0, 1, num=100)) for seg in path])
	x = np.array([p.real for p in points])
	y = np.array([p.imag for p in points])
	# Ensure window_length is an odd number and less than the total number of points
	if window_length % 2 == 0:
		window_length += 1
	if window_length >= len(x):
		window_length = len(x) - 1 if (len(x) - 1) % 2 != 0 else len(x) - 2
	x_smooth = savgol_filter(x, window_length, polyorder)
	y_smooth = savgol_filter(y, window_length, polyorder)
	smoothed_points = [complex(x_smooth[i], y_smooth[i]) for i in range(len(x))]
	new_path = Path()
	for i in range(1, len(smoothed_points)):
		new_path.append(Line(smoothed_points[i-1], smoothed_points[i]))
	# Check if the path is closed
	try:
		if path.isclosed():
			new_path.append(Line(smoothed_points[-1], smoothed_points[0]))
	except AssertionError:
		pass
	return new_path


def smooth_svg(svg_file, output_file, window_length=401, polyorder=3):
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()
    # Handle namespaces
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    # Find all path elements
    path_elements = root.findall('.//svg:path', ns)
    # Parse paths using svgpathtools
    paths = [parse_path(path_elem.get('d')) for path_elem in path_elements]
    # Smooth each path
    smoothed_paths = [smooth_path(path, window_length, polyorder) for path in paths]
    # Update the path data in the SVG
    for path_element, smoothed_path in zip(path_elements, smoothed_paths):
        path_element.set('d', smoothed_path.d())
    # Write the new SVG file
    tree.write(output_file, xml_declaration=True, encoding='utf-8', method='xml')


def add_white_margin(image, margin_size=25):
    # Read the original image
    original_image = image
    # Get the dimensions of the original image
    height, width = original_image.shape
    # Create a black mask for the margin
    margin_mask = np.zeros_like(original_image)
    # Set the outer margin to white
    margin_mask[:margin_size, :] = 255  # Top margin
    margin_mask[-margin_size:, :] = 255  # Bottom margin
    margin_mask[:, :margin_size] = 255  # Left margin
    margin_mask[:, -margin_size:] = 255  # Right margin
    # Overwrite the original image with the white margin
    original_image = np.where(margin_mask == 0, original_image, margin_mask)
    # Return the result
    return original_image

def rotate_points_df(points, pivot, angle):
	# Works with Pandas Dataframe
	# Convert angle to radians
	angle_rad = math.radians(angle)
	# Translate coordinates to the origin
	points['x_translated'] = points['x'] - pivot[0]
	points['y_translated'] = points['y'] - pivot[1]
	# Perform rotation using rotation matrix
	points['x_rotated'] = points['x_translated'] * math.cos(angle_rad) - points['y_translated'] * math.sin(angle_rad)
	points['y_rotated'] = points['x_translated'] * math.sin(angle_rad) + points['y_translated'] * math.cos(angle_rad)
	# Translate the rotated coordinates back
	points['x_rotated'] = points['x_rotated'] + pivot[0]
	points['y_rotated'] = points['y_rotated'] + pivot[1]
	# Create the resulting DataFrame
	result_points = points[['x_rotated', 'y_rotated']].rename(columns={'x_rotated': 'x', 'y_rotated': 'y'})    
	return result_points

# 
def rotate_points(points, pivot, angle):
	# Function accepts list of coordinates (points = [coor1, coor2, coor3])
    # Convert angle to radians
    angle_rad = math.radians(angle)
    # Translate coordinates to the origin
    translated_points = [(point[0] - pivot[0], point[1] - pivot[1]) for point in points]
    # Perform rotation using rotation matrix
    rotated_points = [
        (
            point[0] * math.cos(angle_rad) - point[1] * math.sin(angle_rad),
            point[0] * math.sin(angle_rad) + point[1] * math.cos(angle_rad)
        )
        for point in translated_points
    ]
    # Translate the rotated coordinates back
    rotated_points = [(point[0] + pivot[0], point[1] + pivot[1]) for point in rotated_points]
    return rotated_points


def elongate_digit_df(points, factor):
    # Make a copy of the original points
    oldpoints = points.copy()
    # Calculate the reference point
    ref_x = oldpoints.iloc[0]['x']
    ref_y = oldpoints.iloc[0]['y']
    # Elongate points by the given factor
    for i in range(1, len(points)):
        delta_x = oldpoints.iloc[i]['x'] - ref_x
        delta_y = oldpoints.iloc[i]['y'] - ref_y
        points.iat[i, points.columns.get_loc('x')] = ref_x + delta_x * factor
        points.iat[i, points.columns.get_loc('y')] = ref_y + delta_y * factor
    return points

def proj_keepaspect(points, factor):
    # Make a copy of the original points
    oldpoints = points.copy()
    # Calculate the reference point
    ref_y = oldpoints.iloc[0]['y']
    # Elongate points by the given factor
    for i in range(1, len(points)):
        delta_y = oldpoints.iloc[i]['y'] - ref_y
        points.iat[i, points.columns.get_loc('y')] = ref_y + delta_y * factor
    return points

def change_aspect_ratio_df(points, factor):
    new_points = points.copy()
    new_points['x'] = new_points['x'] * factor
    return new_points

def translate_df(points, amount):
    new_points = points.copy()
    new_points['x'] = new_points['x'] + amount
    return new_points

def ri(start, end):
	if isinstance(start, int) and isinstance(end, int):
		return random.randint(start, end)
	else:
		return random.uniform(start, end)

def reverse_rows(df):
    return df.iloc[::-1].reset_index(drop=True) 
