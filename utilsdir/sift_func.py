from os import name
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import itertools as it
import matplotlib.pyplot as plt
import math
import copy
from numba import jit
import warnings
warnings.filterwarnings("ignore")
from utilsdir.utils import *
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging


import timeit



@jit
def localizeExtremum(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
   
    image_shape = dog_images_in_octave[0].shape
    extremum_is_outside_image = False
    for attempt_index in range(num_attempts_until_convergence):
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
       
        hessian = Hessian_fun(pixel_cube)
        gradient = gradient_fun(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    val=abs(functionValueAtUpdatedExtremum)
    if val * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = val
            return [keypoint, image_index]
    return None
    

@jit
def computeKeypointsWithOrientations1(keypoint, octave_index, gaussian_image):
   
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape
    # num_bins=3
    radius_factor=3
    number_bins=36
    peak_ratio=0.8
    scale_factor=1.5
    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  
    
    radius = int(round(radius_factor * scale))
    smooth_histogram = zeros(number_bins)
    weight_factor = -0.5 / (scale ** 2) 
    raw_histogram = zeros(number_bins)
    total=(number_bins/360.0)
    half_radius=radius/2.0
    for i in range(-radius, radius + 1):
        region_y = (round(keypoint.pt[1] / np.float32(2 ** octave_index)))
        region_y=int(region_y)
        region_y += i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = (round(keypoint.pt[0] / np.float32(2 ** octave_index)))
                region_x=int(region_x)
                region_x += j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    x1=gaussian_image[region_y, region_x + 1]
                    x2=gaussian_image[region_y, region_x - 1]
                    dx=(x1-x2)
                    # dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    y1=gaussian_image[region_y - 1, region_x]
                    y2=gaussian_image[region_y + 1, region_x]
                    dy = (y1 - y2)
                    gradient_magnitude = sqrt((x1-x2) * (x1-x2) + (y1 - y2) * (y1 - y2))
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i * 2 + j * 2))  # constant in front of exponential can be dropped because we will find peaks later
                    # total=(number_bins/360.0)
                    histogram_index = round(gradient_orientation * number_bins/360.0)
                    addition_value=weight * gradient_magnitude
                    raw_histogram[int(histogram_index) % number_bins] += addition_value

    for n in range(number_bins):
        # smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % number_bins]))/16.0 
        smooth_histogram[n] = (raw_histogram[n - 2] + raw_histogram[(n + 2) % number_bins]) / 16.
        smooth_histogram[n] += (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % number_bins]))/16.0 
    # orientation_max = max(smooth_histogram)
    rl1=roll(smooth_histogram, 1)
    rl2=roll(smooth_histogram, -1)
    op = where(logical_and(smooth_histogram > rl1, smooth_histogram > rl2))[0]
    for peak_index in op:
        # peak_value = smooth_histogram[peak_index]
        if smooth_histogram[peak_index] >= peak_ratio * max(smooth_histogram):
            peak_value=smooth_histogram[peak_index]
            previous_index=peak_index-1
            next_index=peak_index+1
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(previous_index) % number_bins]
            right_value = smooth_histogram[(next_index) % number_bins]
            numer=(peak_index + 0.5 * (left_value - right_value))
            denomi=(left_value - 2 * peak_value + right_value)
            # interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            interpolated_peak_index=(numer/denomi)%number_bins
            orientation = 360. - interpolated_peak_index * 360.
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            # new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave))
    return keypoints_with_orientations

@jit
def Difference_of_Gaussian(images):
  out = []
  for i in range(1,len(images)):
    out.append(images[i]-images[i-1])
  return out

@jit
def get_extrema_features(sigma,octave,Gaussian_images,keypoints,allkeypoints,s,k,k_size,next_img):
  count=0
  imgs = []
  for i in range(s+3):
    g = gaussian_kernel(k_size,sigma)
    out = convolve2D(next_img,g,(k_size-1)//2)
    imgs.append(out)
    sigma= sigma*k
  Gaussian_images.append(imgs)
  out = Difference_of_Gaussian(imgs)
  for cnt in range(1,len(out)-1):
    for i in range(1,out[cnt].shape[0]-1):
      for j in range(1,out[cnt].shape[1]-1):
        curr = out[cnt][i][j]
        no_maxima = False
        no_minima = False
        for i_ in range(-1,2):
          if no_maxima:
            break
          for j_ in range(-1,2):
            if(out[cnt+1][i+i_][j+j_] > curr or out[cnt-1][i+i_][j+j_] > curr or out[cnt][i+i_][j+j_] > curr):
              no_maxima = True
              break
        for i_ in range(-1,2):
          if no_minima:
            break
          for j_ in range(-1,2):
            if(out[cnt+1][i+i_][j+j_] < curr or out[cnt-1][i+i_][j+j_] < curr or out[cnt][i+i_][j+j_] < curr):
              no_minima = True
              break
        extrema = (not no_maxima) ^ (not no_minima)  
        if extrema :
          local_result=localizeExtremum(i,j,cnt,octave-1,2,out,sigma,0.04,0,10,5)
          if local_result is not None:
            keypoints.append({
                'keypoint':local_result[0],
                'octave_no':octave,
                'octave_index':cnt,
            })
            keypoints_with_orientation = computeKeypointsWithOrientations(local_result[0],octave,imgs[local_result[1]])
  
            for kp in keypoints_with_orientation:
              allkeypoints.append({
                  'keypoint':kp,
                  'octave_no':octave,
                  'octave_index':cnt,
                  })
  
  return imgs[s],out

def scale_space_extrema(octave,sigma,next_img,Gaussian_images,keypoints,allkeypoints,s,k,k_size):
  curr_img,out = get_extrema_features(int(math.pow(2,octave-1))*sigma,octave,Gaussian_images,keypoints,allkeypoints,s,k,k_size,next_img)
  next_img = np.empty((next_img.shape[0]//2,next_img.shape[1]//2))
  next_img = curr_img[::2,::2]
  return next_img

@jit
def generateDescriptors(keypoints, gaussian_images, window_width=8, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    descriptors = []

    for cnt,kp in enumerate(keypoints):
        print(f"Calculated descriptor vector for Keypoints :{cnt}/{len(keypoints)}")
        octave = kp['octave_no']
        layer = kp['octave_index']
        kp = kp['keypoint'] 
        scale = 1
        row_bin_list = []
        magnitude_list = []
        gaussian_image = gaussian_images[octave][layer]
        num_rows, num_cols = gaussian_image.shape
        
        col_bin_list = []
        point = round(scale * array(kp.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - kp.angle
        cos_angle = cos(deg2rad(360.0-kp.angle))
        sin_angle = sin(deg2rad(360.0-kp.angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        orientation_bin_list = []
        histogram_tensor = zeros((window_width + 2, window_width + 2, num_bins))   
        hist_width = scale_multiplier * 0.5 * scale * kp.size
        # half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5)), sqrt(num_rows * 2 + num_cols * 2)))     # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            row_rot=row*cos_angle
            col_rot=-row*sin_angle
            for col in range(-half_width, half_width + 1):
                # row_rot = col * sin_angle + row * cos_angle
                # col_rot = col * cos_angle - row * sin_angle
                row_rot1=row_rot+col*sin_angle
                col_rot1=col_rot+col*cos_angle
                val=0.5 * window_width - 0.5
                row_bin = (row_rot1 / hist_width) + val
                col_bin = (col_rot1 / hist_width) + val
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(round(point[1] + row))
                    window_col = int(round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        x1=gaussian_image[window_row, window_col + 1]
                        x2=gaussian_image[window_row, window_col - 1]
                        y1=gaussian_image[window_row - 1, window_col]
                        y2=gaussian_image[window_row + 1, window_col]
                        # dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        # dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2))
                        gradient_orientation = rad2deg(arctan2(y1-y2, x1-x2)) % 360
                        weight = exp(weight_multiplier * ((row_rot1 / hist_width) * 2 + (col_rot1 / hist_width) * 2))
                        magnitude_list.append(weight * gradient_magnitude)
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = floor([row_bin, col_bin, orientation_bin]).astype(int)
            # row_fraction, col_fraction, orientation_fraction = -(-row_bin + row_bin_floor), col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            row_fraction=-(-row_bin + row_bin_floor)
            col_fraction=-(-col_bin + col_bin_floor)
            orientation_fraction=-(-orientation_bin + orientation_bin_floor)
            
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 - c11
            c01 = c0 * col_fraction
            c00 = c0 - c01
            c111 = orientation_fraction* magnitude * row_fraction*col_fraction
            c110 = magnitude * row_fraction*col_fraction - c111
            c101 = c10 * orientation_fraction
            c100 = c10 - c101
            c011 = c01 * orientation_fraction
            c010 = c01- c011
            c001 = c00 * orientation_fraction
            c000 = c00 -c001

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        histogram_tensor = histogram_tensor[1:-1,1:-1,:]
        descriptor_mat = np.empty((2,2,8))
        for i_ in range(2):
          for j_ in range(2):
            descriptor_mat[i_][j_] = np.sum(histogram_tensor[2*i_:2*i_+4][2*j_:2*j_+4],axis = (0,1))
        
        descriptor_vector = descriptor_mat.flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)
    return array(descriptors, dtype='float32')

def GetDescriptors(imgname):
    org_img = cv2.imread(imgname)
    # print(org_img,imgname)
    # plt.imshow(org_img)
    org_img = resize(org_img, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    next_img = copy.deepcopy(img)
    s = 2
    k_size=21
    k = math.pow(2,1/s)
    sigma = 1.61

    featurex,featurey = [],[]
    keypoints = []
    Gaussian_images = []
    allkeypoints = []
    for i in range(2):
        next_img = scale_space_extrema(i+1,sigma,next_img,Gaussian_images,keypoints,allkeypoints,s,k,k_size)

    print("calculated keypoints")
    print(f"Calculated {len(keypoints)} Keypoints ")
    descriptors = generateDescriptors(allkeypoints,Gaussian_images,window_width=8)

    return allkeypoints,descriptors




if __name__== "__main__":
    start = timeit.default_timer()
    descriptors = GetDescriptors('../images/box.png')
    print(descriptors.shape)
    stop = timeit.default_timer()
    print('Time: ', stop - start)  