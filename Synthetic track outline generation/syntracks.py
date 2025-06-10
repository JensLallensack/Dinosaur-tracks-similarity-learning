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
import statistics
import os
import shutil
import copy
from svgpathtools import svg2paths, Path, Line, CubicBezier, QuadraticBezier, wsvg, parse_path
import svgpathtools as sp
from scipy.signal import savgol_filter
import xml.etree.ElementTree as ET
from pandas_ods_reader import read_ods
from syntrackslib import rand, rand_t_df, interpad, threshold, png_to_svg, svg_to_png, add_noise_to_svg, invert_image, fill_holes, white_to_alpha, alpha_to_white, segment_to_points, smooth_path, smooth_svg, add_white_margin, rotate_points_df, elongate_digit_df, change_aspect_ratio_df, translate_df, ri, reverse_rows, proj_keepaspect, rand_t_individual, interpad_radius, rand_t, isnumber, whentodraw, unlist

import warnings
warnings.filterwarnings("ignore")

#Defaults
when_to_draw_default = 0
when_to_draw_EHF_default = 0.5
when_to_draw_RHF_default = 0.5
draw_at_random_default = 1
draw_at_random_RHF_default = 0.3
radius_noise_default = 0.1
radius_noise_RHF_default = 0.2
interpad_size_noise_default = 0.1
interpad_draw_at_random_default = 0.9

drawing_size = 1200 #canvas size


def syntracks(filename,n=5,rangeportion=100,magnitude=1.1,include_standard=True):
    dat = read_ods(filename,skiprows=6)
    def remove_spaces(value):
        if isinstance(value, str):
            return value.replace(" ", "")
        return value
    dat.applymap(remove_spaces)
    coords = dat[["x","y"]]
    coords = coords.dropna(how='all')
    settings = dat[["settings","settings_value"]]
    settings = settings.dropna(how='all')
    params = dat[["parameter","on/off","relative_magnitude","magnitude_noise"]]
    params = params.dropna(how='all')
    params = params.set_index('parameter', drop=True)
    if include_standard == False:
        params.loc["standard","on/off"] = 0
    pads = dat[["digit","pad_class","radius","when_to_draw","do_not_draw_when","draw_at_random","radius_noise","position_noise","interpad_shape","interpad_size","interpad_size_noise","interpad_draw_at_random","reuse","additional_interpad_1","additional_interpad_1_draw_at_random","additional_interpad_1_when_to_draw","additional_interpad_1_radius","additional_interpad_2","additional_interpad_2_draw_at_random","additional_interpad_2_when_to_draw","additional_interpad_2_radius"]]
    pads = pads.dropna(how='all')
    def remove_trailing_nan(df, column_name):
        last_valid_index = df[column_name].last_valid_index()
        if last_valid_index is not None:
            return df.loc[:last_valid_index]
        return df
    pads = remove_trailing_nan(pads, 'pad_class')
    #get settings
    circle_radius_scalefactor = settings.loc[settings['settings'] == "circle_radius_scalefactor", 'settings_value'].values
    interpad_size_default = settings.loc[settings['settings'] == "interpad_size_default", 'settings_value'].values
    circle_radius_scalefactor = circle_radius_scalefactor[0]
    variation_circle_radius_scalefactor = settings.loc[settings['settings'] == "variation_circle_radius_scalefactor", 'settings_value'].values
    variation_circle_radius_scalefactor = variation_circle_radius_scalefactor[0]
    position_noise_default = settings.loc[settings['settings'] == "position_noise_default", 'settings_value'].values
    interpad_radius_asymmetry = settings.loc[settings['settings'] == "interpad_radius_asymmetry", 'settings_value'].values
    interpad_radius_asymmetry_noise = settings.loc[settings['settings'] == "interpad_radius_asymmetry_noise", 'settings_value'].values
    interpad_radius_asymmetry_claw = settings.loc[settings['settings'] == "interpad_radius_asymmetry_claw", 'settings_value'].values
    interpad_radius_asymmetry_claw_noise = settings.loc[settings['settings'] == "interpad_radius_asymmetry_claw_noise", 'settings_value'].values
    interpad_radius_asymmetry_PEP = settings.loc[settings['settings'] == "interpad_radius_asymmetry_PEP", 'settings_value'].values
    interpad_radius_asymmetry_PEP_noise = settings.loc[settings['settings'] == "interpad_radius_asymmetry_PEP_noise", 'settings_value'].values
    interpad_position_noise = settings.loc[settings['settings'] == "interpad_position_noise", 'settings_value'].values
    #replace empty fields with defaults
    def replace_non_numeric(value,default):
        try:
            # Check if the value can be converted to a number
            float(value)
            return value  # Keep the value if it's a number
        except (ValueError, TypeError):
            return default  # Replace with default if not a number
    def replace_by_pad_class(pads,parameter,pad_class,default):
        for b in range(len(pads)):
            if pads.at[b,"pad_class"] == pad_class:
                pads.at[b,parameter] = replace_non_numeric(pads.at[b,parameter],default)
        return pads
    #pad class: EHF
    pads = replace_by_pad_class(pads,"when_to_draw","EHF",when_to_draw_EHF_default)
    pads = replace_by_pad_class(pads,"when_to_draw","RHF",when_to_draw_RHF_default)
    pads = replace_by_pad_class(pads,"draw_at_random","RHF",draw_at_random_RHF_default)
    pads = replace_by_pad_class(pads,"radius_noise","RHF",radius_noise_RHF_default)
    pads["when_to_draw"] = pads["when_to_draw"].apply(lambda x: replace_non_numeric(x, when_to_draw_default))
    pads["draw_at_random"] = pads["draw_at_random"].apply(lambda x: replace_non_numeric(x, draw_at_random_default))
    pads["radius_noise"] = pads["radius_noise"].apply(lambda x: replace_non_numeric(x, radius_noise_default))
    pads["position_noise"] = pads["position_noise"].apply(lambda x: replace_non_numeric(x, position_noise_default))
    pads["interpad_size"] = pads["interpad_size"].apply(lambda x: replace_non_numeric(x, interpad_size_default))
    pads["interpad_size_noise"] = pads["interpad_size_noise"].apply(lambda x: replace_non_numeric(x, interpad_size_noise_default))
    pads["interpad_draw_at_random"] = pads["interpad_draw_at_random"].apply(lambda x: replace_non_numeric(x, interpad_draw_at_random_default))
    def normalize(coords, target=500):
        max_diff_x = max(coords["x"])-min(coords["x"])
        max_diff_y = max(coords["y"])-min(coords["y"])
        maxdim = max(max_diff_x,max_diff_y)
        coords = (coords/maxdim)*target
        meanx = statistics.mean(coords["x"])
        meany = statistics.mean(coords["y"])
        coords["x"] = coords["x"]-meanx
        coords["y"] = coords["y"]-meany
        return coords
    coords = normalize(coords)
    #Range (only separated outlines, or only full outlines)?
    if(rangeportion > 0):
        ranger = range(0,round(rangeportion/100)*n)
    else:
        ranger = range(rangeportion,100)
    #get digits
    def find_ranges(mask):
        # Find the indices where the mask is True
        true_indices = mask.index[mask].tolist()
        if not true_indices:
            return None  # or (None, None) if you prefer
        # Get the first and last occurrence
        first_index = true_indices[0]
        last_index = true_indices[-1]+1
        return (first_index, last_index)
    mask = pads['digit'] == 'digit II'
    digit2 = find_ranges(mask)
    mask = pads['digit'] == 'digit III'
    digit3 = find_ranges(mask)
    mask = pads['digit'] == 'digit IV'
    digit4 = find_ranges(mask)
    mask = pads['digit'] == 'digit I'
    digit1 = find_ranges(mask)
    mask = pads['digit'] == 'heel'
    heel = find_ranges(mask)
    mask = pads['pad_class'] == 'MPP'
    mpp = find_ranges(mask)
    def createtrack(parameter):
        if params.at[parameter, 'on/off'] == 1:
            print("creating ",parameter)
            for i in ranger:
                track = copy.deepcopy(coords)
                pads_c = copy.deepcopy(pads)
                smallest_circle_radius = 45 * circle_radius_scalefactor
                variation_circle_radius = 15 * variation_circle_radius_scalefactor
                stepsize = variation_circle_radius/n
                pad_radius = i * stepsize + smallest_circle_radius
                #Alter coords according to parameter
                if parameter == "int_ang":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit2[0]+1:digit2[1]-1, ['x', 'y']] = rotate_points_df(points=track.iloc[digit2[0]+1:digit2[1]].copy(),pivot=track.iloc[digit2[0]][['x', 'y']],angle=ri(-relative_magnitude-magnitude_noise,-relative_magnitude+magnitude_noise))
                    track.loc[digit4[0]+1:digit4[1]-1, ['x', 'y']] = rotate_points_df(points=track.iloc[digit4[0]+1:digit4[1]].copy(),pivot=track.iloc[digit4[0]][['x', 'y']],angle=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "int_ang_dII":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit2[0]+1:digit2[1]-1, ['x', 'y']] = rotate_points_df(points=track.iloc[digit2[0]+1:digit2[1]].copy(),pivot=track.iloc[digit2[0]][['x', 'y']],angle=ri(-relative_magnitude-magnitude_noise,-relative_magnitude+magnitude_noise))
                if parameter == "int_ang_dIV":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit4[0]+1:digit4[1]-1, ['x', 'y']] = rotate_points_df(points=track.iloc[digit4[0]+1:digit4[1]].copy(),pivot=track.iloc[digit4[0]][['x', 'y']],angle=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "length_dII":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit2[0]:digit2[1]-1, ['x', 'y']] = elongate_digit_df(points=track.loc[digit2[0]:digit2[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "length_dIII":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit3[0]:digit3[1]-1, ['x', 'y']] = elongate_digit_df(points=track.loc[digit3[0]:digit3[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "length_dIV":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit4[0]:digit4[1]-1, ['x', 'y']] = elongate_digit_df(points=track.loc[digit4[0]:digit4[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "length_dIII_proximal":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    slic = track.loc[digit3[0]:digit3[1]-1, ['x', 'y']]
                    d3reversed = reverse_rows(slic)
                    shortened = elongate_digit_df(points=d3reversed,factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                    slic2 = reverse_rows(shortened)
                    slic2.index = slic.index
                    track.loc[digit3[0]:digit3[1]-1, ['x', 'y']] = slic2
                if parameter == "move_dII":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit2[0]:digit2[1]-1, ['x', 'y']] = translate_df(points=track.loc[digit2[0]:digit2[1]-1, ['x', 'y']],amount=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "move_dIV":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit4[0]:digit4[1]-1, ['x', 'y']] = translate_df(points=track.loc[digit4[0]:digit4[1]-1, ['x', 'y']],amount=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "move_mpp":
                    relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit4[0]:digit4[1]-2, ['x', 'y']] = rotate_points_df(points=track.iloc[digit4[0]:digit4[1]-1].copy(),pivot=track.iloc[digit4[1]-1][['x', 'y']],angle=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "aspect_ratio":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track = change_aspect_ratio_df(track,factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "d3proj_keepaspect":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    diff = relative_magnitude - 1 #invert (so that magnitude of "1.1" increases projection, and "0.9" decreases it.
                    relative_magnitude = (1 - diff)
                    track.loc[digit2[0]:digit2[1]-1, ['x', 'y']] = proj_keepaspect(points=track.loc[digit2[0]:digit2[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                    track.loc[digit4[0]:digit4[1]-1, ['x', 'y']] = proj_keepaspect(points=track.loc[digit4[0]:digit4[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                if parameter == "digit_length_asymmetry":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    track.loc[digit4[0]:digit4[1]-1, ['x', 'y']] = elongate_digit_df(points=track.loc[digit4[0]:digit4[1]-1, ['x', 'y']],factor=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                    diff = relative_magnitude - 1
                    inverted_magnitude = 1 - diff
                    track.loc[digit2[0]:digit2[1]-1, ['x', 'y']] = elongate_digit_df(points=track.loc[digit2[0]:digit2[1]-1, ['x', 'y']],factor=ri(inverted_magnitude-magnitude_noise,inverted_magnitude+magnitude_noise))
                if parameter == "size_mpp":
                    relative_magnitude = ((params.loc[parameter, 'relative_magnitude']-1)*magnitude)+1
                    magnitude_noise = params.loc[parameter, 'magnitude_noise']
                    mask = pads_c['pad_class'].str.contains('MPP', na=False)
                    row = pads_c.index[mask][0]
                    with_noise = ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise)
                    pads_c.at[row,"radius"] = pads_c.at[row,"radius"] * with_noise
                    #Add position noise
                if parameter == "rounded_heel":
                    if(pads["pad_class"].eq("MPP").any()):
                        mask = pads_c['pad_class'] == 'ADD'
                        indexes = mask.index[mask].tolist()
                        pads_c.at[indexes[0],"draw_at_random"] = 1
                        pads_c.at[indexes[1],"draw_at_random"] = 1
                if parameter == "rotate_dI":
                    if(pads["digit"].eq("digit I").any()):
                        relative_magnitude = params.loc[parameter, 'relative_magnitude']*magnitude
                        magnitude_noise = params.loc[parameter, 'magnitude_noise']
                        track.loc[digit1[0]:digit1[1]-1, ['x', 'y']] = rotate_points_df(points=track.iloc[digit1[0]:digit1[1]].copy(),pivot=track.iloc[heel[0]][['x', 'y']],angle=ri(relative_magnitude-magnitude_noise,relative_magnitude+magnitude_noise))
                track = rand_t_individual(track, pads_c["position_noise"])
                track_ordered = copy.deepcopy(track)
                # Calculate the "major_interpads" vector.
                # # Initialize empty lists
                listlen = len(pads_c)
                major_interpads = [None] * listlen
                interpad_shape = [None] * listlen
                interpad_shape_def = [None] * listlen
                interpad_size_noise = [None] * listlen
                interpad_draw_at_random = [None] * listlen
                #set defaults for interpad shape
                for q in range(listlen-1):
                    if isnumber(pads_c.at[q,"interpad_shape"]):
                        interpad_shape_def[q] = pads_c.at[q,"interpad_shape"]
                    else:
                        if(pads_c.at[q,"pad_class"] in ["MP", "MPP", "EHF", "RHF", "ADD"]) and (pads_c.at[q+1,"pad_class"] not in "CM"):
                            interpad_shape_def[q] = 1
                        else:
                            interpad_shape_def[q] = 2
                #fix last entry
                if isnumber(pads_c.at[len(pads_c)-1,"interpad_shape"]):
                    interpad_shape_def[len(pads_c)-1] = pads_c.at[len(pads_c)-1,"interpad_shape"]
                else:
                    interpad_shape_def[len(pads_c)-1] = 1
                pads_c["interpad_shape"] = interpad_shape_def
                #get interpad lists
                for w in range(listlen-1):
                    if(pads_c.at[w,"pad_class"] in ["MP", "PEP", "MPP", "EHF", "RHF", "AEP", "ADD"]) and (pads_c.at[w,"digit"] == pads_c.at[w+1,"digit"]) and (pads_c.at[w,"interpad_shape"] != 0):
                        major_interpads[w] = w
                        interpad_size_noise[w] = pads_c.at[w,"interpad_size_noise"]
                        interpad_draw_at_random[w] = pads_c.at[w,"interpad_draw_at_random"]
                        interpad_shape[w] = interpad_shape_def[w]
                #draw pads
                d = draw.Drawing(drawing_size, drawing_size, origin='center')
                def drawcircles(pad,pad_radius,fill='white'):
                    d.append(draw.Circle(pad[0],pad[1],pad_radius,fill='white'))
                previous_pad = tuple([None, None])
                for u in range(listlen):
                    drawit = whentodraw(pads_c.at[u,"when_to_draw"],i,n)
                    drawit3 = True
                    if pads_c.at[u,"do_not_draw_when"] == parameter:
                        drawit3 = False
                    #"reuse" feature
                    if drawit == False:
                        if isnumber(pads_c.at[u,"reuse"]):
                            if random.random() < pads_c.at[u,"reuse"]:
                                drawit = True
                    drawit2 = True
                    if drawit == True and drawit3 == True:
                        draw_at_random = pads_c.at[u,"draw_at_random"]
                        # if draw_at_random < 100:
                        if random.random() > draw_at_random:
                            drawit2 = False
                        if drawit2 == True:
                            coord_to_draw = track.loc[u]
                            drawcircles(track.loc[u],pad_radius*pads_c.at[u,"radius"])
                            #Draw Interpads
                            def draw_all_interpads(index_pad1,index_pad2,shape=None,interpad_size=None,interpad_size_noise=None):
                                if isnumber(shape) == False:
                                    shape = interpad_shape_def[index_pad1]
                                if isnumber(interpad_size) == False:
                                    interpad_size = pads_c.at[index_pad1,"interpad_size"]
                                if isnumber(interpad_size_noise) == False:
                                    interpad_size_noise = pads_c.at[index_pad1,"interpad_size_noise"]
                                ipn = interpad_position_noise[0]
                                if shape == 1:
                                    asym = interpad_radius_asymmetry
                                    noise = interpad_radius_asymmetry_noise
                                if shape == 2:
                                    asym = interpad_radius_asymmetry_claw
                                    noise = interpad_radius_asymmetry_claw_noise
                                if shape == 3:
                                    asym = 0.5
                                    noise =  interpad_radius_asymmetry_noise
                                if pads_c.at[index_pad1,"pad_class"] == "PEP":
                                    asym = interpad_radius_asymmetry_PEP
                                    noise = interpad_radius_asymmetry_PEP_noise
                                asym = unlist(asym)
                                noise = unlist(noise)
                                pad1_coords = track.loc[index_pad1]
                                pad2_coords = track.loc[index_pad2]
                                pad1_radius = pads_c.at[index_pad1,"radius"]*pad_radius
                                pad2_radius = pads_c.at[index_pad2,"radius"]*pad_radius
                                #Draw major interpad
                                ###Coords of najor interpad
                                major_interpad_coords = rand_t(interpad(pad1_coords,pad2_coords),ipn)
                                ###Radius of major interpad
                                if shape == 1:
                                    maxpad = max(pad1_radius,pad2_radius)
                                    relinpadsize = np.random.uniform(interpad_size - interpad_size_noise, interpad_size + interpad_size_noise)
                                    major_interpad_size = maxpad * relinpadsize
                                else:
                                    asym_noisy = np.random.uniform(asym, asym + noise)
                                    major_interpad_size = ((pad1_radius)+(pad2_radius))*asym_noisy
                                major_interpad_size = unlist(major_interpad_size)
                                ###Draw the major interpad
                                drawcircles(major_interpad_coords,major_interpad_size)
                                #Secondary interpads
                                secondary_1_radius = interpad_radius(pad1_radius,major_interpad_size,asym,noise)
                                secondary_2_radius = interpad_radius(major_interpad_size,pad2_radius,asym,noise)
                                secondary_1_coords = rand_t(interpad(pad1_coords,major_interpad_coords),ipn)
                                secondary_2_coords = rand_t(interpad(major_interpad_coords,pad2_coords),ipn)
                                drawcircles(secondary_1_coords,secondary_1_radius)
                                drawcircles(secondary_2_coords,secondary_2_radius)
                                #Tertiary interpads
                                tertiary_1_radius = interpad_radius(pad1_radius,secondary_1_radius,asym,noise)
                                tertiary_2_radius = interpad_radius(secondary_1_radius,major_interpad_size,asym,noise)
                                tertiary_3_radius = interpad_radius(major_interpad_size,secondary_2_radius,asym,noise)
                                tertiary_4_radius = interpad_radius(secondary_2_radius,pad2_radius,asym,noise)
                                tertiary_1_coords = rand_t(interpad(pad1_coords,secondary_1_coords),ipn)
                                tertiary_2_coords = rand_t(interpad(secondary_1_coords,major_interpad_coords),ipn)
                                tertiary_3_coords = rand_t(interpad(major_interpad_coords,secondary_2_coords),ipn)
                                tertiary_4_coords = rand_t(interpad(secondary_2_coords,pad2_coords),ipn)
                                drawcircles(tertiary_1_coords,tertiary_1_radius)
                                drawcircles(tertiary_2_coords,tertiary_2_radius)
                                drawcircles(tertiary_3_coords,tertiary_3_radius)
                                drawcircles(tertiary_4_coords,tertiary_4_radius)
                            #draw at random?
                            drawinterpads = True
                            drawrandom = pads_c.at[u,"interpad_draw_at_random"]
                            if drawrandom < 1:
                                if random.random() > drawrandom:
                                    drawinterpads = False
                            if(u > 0) and (major_interpads[u-1] != None) and (previous_pad[0] != None) and (drawinterpads == True):
                                if pads_c.at[u-1,"pad_class"] == "AEP":
                                    draw_all_interpads(u-2,u)
                                else:
                                    draw_all_interpads(u-1,u)
                            previous_pad = tuple(track.loc[u])
                            #Draw additional interpads
                            addpad_1 = pads_c.at[u,"additional_interpad_1"]
                            if isnumber(pads_c.at[u,"additional_interpad_1_draw_at_random"]):
                                addpad_1_random = pads_c.at[u,"additional_interpad_1_draw_at_random"]
                            else:
                                addpad_1_random = 1
                            addpad_1_radius = pads_c.at[u,"additional_interpad_1_radius"]
                            if isnumber(pads_c.at[u,"additional_interpad_1_when_to_draw"]):
                                addpad_1_whentodraw = pads_c.at[u,"additional_interpad_1_when_to_draw"]
                            else:
                                addpad_1_whentodraw = 0
                            addpad_2 = pads_c.at[u,"additional_interpad_2"]
                            if isnumber(pads_c.at[u,"additional_interpad_2_draw_at_random"]):
                                addpad_2_random = pads_c.at[u,"additional_interpad_2_draw_at_random"]
                            else:
                                addpad_2_random = 1
                            addpad_2_radius = pads_c.at[u,"additional_interpad_2_radius"]
                            if isnumber(pads_c.at[u,"additional_interpad_2_when_to_draw"]):
                                addpad_2_whentodraw = pads_c.at[u,"additional_interpad_2_when_to_draw"]
                            if isnumber(addpad_1) and random.random() < addpad_1_random and whentodraw(addpad_1_whentodraw,i,n):
                                draw_all_interpads(u,int(addpad_1),interpad_size=addpad_1_radius)
                            if isnumber(addpad_2) and random.random() < addpad_2_random and whentodraw(addpad_2_whentodraw,i,n):
                                draw_all_interpads(u,int(addpad_2),interpad_size=addpad_2_radius)
                    if(drawit == False) or (drawit2 == False):
                        previous_pad = tuple([None, None])
                d.save_png('image_orig.png')
                invert_image("image_orig.png","image.png")
                fill_holes("image.png","image.png")
                white_to_alpha("image.png","image.png")
                png_to_svg()
                add_noise_to_svg("image.svg","image.svg")
                smooth_svg("image.svg","image_smooth.svg")
                svg_to_png("image_smooth.svg","image.png")
                alpha_to_white("image.png","imageW.png")
                invert_image("imageW.png","image.png")
                #Blur
                image = cv2.imread('image.png', cv2.IMREAD_UNCHANGED)
                def kernel_odd(center, radius):
                    rand_num = random.randint(center - radius, center + radius)
                    return rand_num if rand_num % 2 != 0 else rand_num + 1 if rand_num < center + radius else rand_num - 1
                kernelsize = kernel_odd(5,2)
                stdev = random.randint(7 - 3, 7 + 3)
                blurred_image = cv2.GaussianBlur(image, (kernelsize, kernelsize), stdev)
                blurred_image = threshold(blurred_image)
                blurred_image = cv2.GaussianBlur(blurred_image, (3, 3), 3)
                blurred_image = add_white_margin(blurred_image)
                #Write file
                if(i == 0):
                    shutil.rmtree(parameter, ignore_errors=True)
                    foldername_blanc = os.path.splitext(filename)[0]
                    mag = str(magnitude).replace('.', '-')
                    foldername = f"{foldername_blanc}_{parameter}_{mag}"
                    if os.path.isdir(foldername) == False:
                        os.makedirs(foldername)
                imagename = f"{parameter}_{i}.png"
                cv2.imwrite(os.path.join(foldername, imagename),blurred_image)
    createtrack("standard")
    createtrack("int_ang")
    createtrack("int_ang_dII")
    createtrack("int_ang_dIV")
    createtrack("length_dII")
    createtrack("length_dIII")
    createtrack("length_dIV")
    createtrack("move_dII")
    createtrack("move_dIV")
    createtrack("move_mpp")
    createtrack("length_dIII_proximal")
    createtrack("aspect_ratio")
    createtrack("d3proj_keepaspect")
    createtrack("digit_length_asymmetry")
    createtrack("size_mpp")
    createtrack("rounded_heel")
    createtrack("rotate_dI")
