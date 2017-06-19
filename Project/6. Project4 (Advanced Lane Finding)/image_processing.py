import numpy as np
import cv2
import pickle
import glob
from tqdm import tqdm
from tracker import tracker

dist_pickle = pickle.load( open( "camera_cal/calibration_ pickle.p", "rb" ) )
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


# Apply Sobel x or y, then takes an absolute value and applies a threshold.
# A function that takes an image, gradient orientation, and threshold min / max values.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


# Applies Sobel x and y, then computes the magnitude of the gradient
# and applies a threshold.
# A function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Applies Sobel x and y, then computes the direction of the gradient
# and applies a threshold.
# A function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Appies Colour threshold
def colour_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hls[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output



def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    #output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output

# Apply undistortion on the image
def undistort_image(img_name):
    # read an image
    img = cv2.imread(fname)

    #undistort the image
    img = cv2.undistort(img, mtx, dist, None, mtx)
    return img


# Apply a process on image and generate binary pixel of interests
def preprocessed_image(img):
    preprocessed_img = np.zeros_like(img[:,:,0])
    grad_x = abs_sobel_thresh(img, orient='x', thresh_min=12, thresh_max=255)
    grad_y = abs_sobel_thresh(img, orient='y', thresh_min=25, thresh_max=255)
    c_binary = colour_threshold(img, sthresh=(150, 255), vthresh=(50, 255))
    
    mag_binary = mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
    #preprocessed_img[( (grad_x == 1) & (grad_y == 1) | (c_binary == 1) )] = 255
    preprocessed_img[( (grad_x == 1) & (grad_y == 1) ) | (c_binary==1) | ((mag_binary==1) & (dir_binary==1))] = 255
    return preprocessed_img
    
# Work on defining perspective transformation area
# Perform the transform
def transform_to_bird_eye_view(img, processed_img):
    # Work on defining perspective transformation area
    img_size = (img.shape[1], img.shape[0])
    bot_width = 0.76 # percent of bottom trapezoid height
    mid_width = 0.08 # percent of middle trapezoid height
    height_pct = 0.62 # percent for trapezoid height
    bottom_trim = 0.935 # percent from top to bottom to avoid car hood
    src = np.float32([[img.shape[1]*(0.5-mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(0.5+mid_width/2), img.shape[0]*height_pct], 
                    [img.shape[1]*(0.5+bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(0.5-bot_width/2),  img.shape[0]*bottom_trim]])
    offset = img_size[0]*0.25
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], [img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    # perform the transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(preprocessed_img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, Minv, img_size

if __name__ == '__main__':
    images = glob.glob('./test_images/test*.jpg')

    for idx, fname in tqdm(enumerate(images)):
        
        #undistort the image
        img = undistort_image(fname)

        # process image and generate binary pixel of interests
        preprocessed_img = preprocessed_image(img)
        result = preprocessed_img
        write_name = './test_images/1. preprocessed_img' + str(idx+1) + '.jpg'
        cv2.imwrite(write_name, result)

        # work on defining perspective transformation area
        # perform the transform
        warped, Minv, img_size = transform_to_bird_eye_view(img, preprocessed_img)
        result = warped
        write_name = './test_images/2. warped' + str(idx+1) + '.jpg'
        cv2.imwrite(write_name, result)

        window_width = 25
        window_height = 80
        # set the overall class to do all the tracking
        curve_centres = tracker(my_window_width=window_width, my_window_height=window_height, my_margin=25, my_ym = 10/720, my_xm = 4/384 , my_smooth_factor = 15)

        window_centroids = curve_centres.find_window_centroids(warped)

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Points used to find the left and right lanes
        right_x = []
        left_x = []

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            # add centre values found in frame to the list of lane points per left, right
            left_x.append(window_centroids[level][0])
            right_x.append(window_centroids[level][1])

            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            '''
            # add centre value found in frame to the list of lane points per left, right
            # left_x.append(window_centroids[level][0])
            # right_x.append(window_centroids[level][1])
            '''
            #Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | (l_mask == 1)] = 255
            r_points[(r_points == 255) | (r_mask == 1)] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero colour channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 colour channels
        
        result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overaly the original road image with window results
        write_name = './test_images/3. warpage' + str(idx+1) + '.jpg'
        cv2.imwrite(write_name, result)

        # fit the lane boundaries to the left, right centre positions found
        y_vals = range(0, warped.shape[0])

        res_y_vals = np.arange(warped.shape[0]-(window_height/2), 0, -window_height)

        left_fit = np.polyfit(res_y_vals, left_x, 2)
        left_fit_x = left_fit[0]*y_vals*y_vals + left_fit[1]*y_vals + left_fit[2]
        left_fit_x = np.array(left_fit_x, np.int32)

        right_fit = np.polyfit(res_y_vals, right_x, 2)
        right_fit_x = right_fit[0]*y_vals*y_vals + right_fit[1]*y_vals + right_fit[2]
        right_fit_x = np.array(right_fit_x, np.int32)

        left_lane = np.array(list(zip(np.concatenate((left_fit_x-window_width/2, left_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fit_x-window_width/2, right_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
        middle_marker = np.array(list(zip(np.concatenate((right_fit_x-window_width/2, right_fit_x[::-1]+window_width/2), axis=0), np.concatenate((y_vals, y_vals[::-1]), axis=0))), np.int32)
        
        road = np.zeros_like(img)
        road_bkg = np.zeros_like(img)
        cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
        cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
        cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])
        
        road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
        result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
        write_name = './test_images/4. road_warped' + str(idx+1) + '.jpg'
        cv2.imwrite(write_name, result)

        ym_per_pix = curve_centres.ym_per_pix # metres per pixel in y dimension
        xm_per_pix = curve_centres.xm_per_pix # metres per pixel in x dimension

        #curve_fit_cr = np.polyfit(np.array(res_y_vals, np.float32)*ym_per_pix, np.array(left_x, np.float32)*xm_per_pix, 2)
        #print(curve_fit_cr)
        #print('a:', (1 + (2*curve_fit_cr[0]*y_vals[-1]*ym_per_pix + curve_fit_cr[1]**2)**1.5), np.absolute(2*curve_fit_cr[0]))
        #print('b:', np.float64(1 + (2*curve_fit_cr[0]*y_vals[-1]*ym_per_pix + curve_fit_cr[1]**2)**1.5))
        #curverad = ((1 + (2*curve_fit_cr[0]*y_vals[-1]*ym_per_pix + curve_fit_cr[1]**2)**1.5)/np.absolute(2*curve_fit_cr[0]))
        #print('c:', curverad)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(np.array(res_y_vals, np.float32)*ym_per_pix, np.array(left_x, np.float32)*xm_per_pix, 2)
        right_fit_cr = np.polyfit(np.array(res_y_vals, np.float32)*ym_per_pix, np.array(right_x, np.float32)*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_vals[-1]*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_vals[-1]*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        curverad = int(left_curverad + right_curverad)/2

        # Calculate the offset of the car on the road
        camera_centre = (left_fit_x[-1] + right_fit_x[-1])/2
        centre_diff = (camera_centre - warped.shape[1]/2)*xm_per_pix
        side_pos = 'left'
        if centre_diff <= 0:
            side_pos = 'right'
            
        # Draw the text showing curvature, offset, and speed
        cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is ' + str(abs(round(centre_diff, 3))) + 'm ' + side_pos + ' of centre', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        write_name = './test_images/5. tracked' + str(idx+1) + '.jpg'
        cv2.imwrite(write_name, result)