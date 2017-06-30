import numpy as np
import cv2

class tracker():

    def __init__(self, my_window_width, my_window_height, my_margin, my_ym = 1, my_xm = 1, my_smooth_factor = 15):
        # list that stores all the past (left, right) centre set values used for smoothing the output
        self.recent_centres = []

        # the window pixel width of the centre values, used to count pixels inside centre windows to determine curve vlalues
        self.window_width = my_window_width

        # the window pixel height of the centre values, used to count pixels inside centre windows to determine curve vlalues
        # breaks the image into vertical levels
        self.window_height = my_window_height

        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = my_margin

        self.ym_per_pix = my_ym # metres per pixel in vertical axis

        self.xm_per_pix = my_xm # metres per pixel in horizontal axis

        self.smooth_factor = my_smooth_factor

    # the main tracking function for finding and storing lane segment positions
    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = [] # store the (left, right) window centroid positions per level
        window = np.ones(window_width) # create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum qurter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :int(warped.shape[1]/2)], axis=0)
        l_centre = np.argmax(np.convolve(window, l_sum)) - window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):, int(warped.shape[1]/2):], axis=0)
        r_centre = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_centre, r_centre))

        # Go through each layer looking for max pixel locations
        for level in range(1, int(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not centre of window
            offset = window_width/2
            l_min_index = int(max(l_centre+offset-margin,0))
            l_max_index = int(min(l_centre+offset+margin, warped.shape[1]))
            #print(l_min_index, l_max_index)
            #print('asdfaer:', np.sum(conv_signal))
            l_centre = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right centre as a reference
            r_min_index = int(max(r_centre+offset-margin,0))
            r_max_index = int(min(r_centre+offset+margin, warped.shape[1]))
            r_centre = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_centre, r_centre))

        self.recent_centres.append(window_centroids)
        # Return averaged values of the line centres, helps to keep the markers from jumping around too much
        return np.average(self.recent_centres[-self.smooth_factor:], axis=0)