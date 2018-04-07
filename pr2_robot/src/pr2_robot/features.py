import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from sensor_stick.pcl_helper import *

def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

def nhist(xs, *args, **kwargs):
    h = [np.histogram(x, *args, **kwargs)[0] for x in xs]
    h = np.concatenate(h).astype(np.float32)
    h /= np.sum(h)#, axis=-1, keepdims=True)
    return h

#import cv2

def compute_color_histograms(cloud, using_hsv=False, bins=16):

    # Compute histograms for the clusters
    point_colors_list = []

    # Step through each point in the point cloud
    for point in pc2.read_points(cloud, skip_nans=True):
        rgb_list = float_to_rgb(point[3])
        if using_hsv:
            point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
        else:
            point_colors_list.append(rgb_list)

    # Populate lists with color values
    channel_1_vals = []
    channel_2_vals = []
    channel_3_vals = []

    for color in point_colors_list:
        channel_1_vals.append(color[0])
        channel_2_vals.append(color[1])
        channel_3_vals.append(color[2])

    res = nhist([channel_1_vals, channel_2_vals, channel_3_vals], bins=bins, range=[0, 256])

    #rmax = np.argmax(res[:bins]) / float(bins) * 256
    #gmax = np.argmax(res[bins:2*bins]) / float(bins) * 256
    #bmax = np.argmax(res[2*bins:]) / float(bins) * 256
    #print('rgb: ({},{},{})'.format(rmax, gmax, bmax))
    #o = np.ones((128,128), dtype=np.float32)
    #m = np.stack([o*bmax, o*gmax, o*rmax], axis=-1)
    #m /= 256.0
    #cv2.imshow('color', m)
    #cv2.waitKey(10)

    return res

def compute_normal_histograms(normal_cloud, bins=16):
    norm_x_vals = []
    norm_y_vals = []
    norm_z_vals = []

    for norm_component in pc2.read_points(normal_cloud,
                                          field_names = ('normal_x', 'normal_y', 'normal_z'),
                                          skip_nans=True):
        norm_x_vals.append(norm_component[0])
        norm_y_vals.append(norm_component[1])
        norm_z_vals.append(norm_component[2])

    # TODO: Compute histograms of normal values (just like with color)

    # TODO: Concatenate and normalize the histograms

    # Generate random features for demo mode.  
    # Replace normed_features with your feature vector
    return nhist([norm_x_vals, norm_y_vals, norm_z_vals], bins=bins, range=[-1,1])

def cloud2feature(cloud, normals, hsv=False, bins=16):
    chists = compute_color_histograms(cloud, using_hsv=hsv, bins=bins)
    nhists = compute_normal_histograms(normals, bins=bins)
    feature = np.concatenate((chists, nhists))
    return feature
