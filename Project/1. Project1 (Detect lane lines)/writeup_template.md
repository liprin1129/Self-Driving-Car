#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. First, I converted the images to grayscale,

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/gray.jpg width="700">

then I did Gaussian smoothing to filter out any noises.

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/blur_gray.jpg width="700">

Next, I ran Canny Edge Detector to the smoothed image in order to detect edges,

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/edges.jpg width="700">

then I imposed a quadrilateral mask to eliminate periphery of lane lines. 

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/masked_edges.jpg width="700">

For the next step, Hough Transform was implemented on the edge detected image to identify lane lines.

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/lines_edges.jpg width="700">

Finally, I drew the lane lines on the original image.

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/lines_on_original_image.jpg width="700">

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function by calculating slopes of the line segments. The equation of slope is (y2-y1)/(x2-x1), and the lines will have positive slope value are assumed as on right, otherwise left lines will have negative slope values. Then, I averaged the position of each line, and extrapolate to the top and bottom of the lane on the image, using the equation for bottom position, `avg_x - (avg_y - imshape[0])/slope`, and for top position, `avg_x + (330 - avg_y)/slope`. Finally, I casted extrapolated top and bottom values which are float values into int values, and then drew the lines on the original image.

However, only using the 'draw a single line' method as above did not show the robust result as the examples, because some slopes changes dramatically between 0 to infinity and shows many inappropriate lines. 

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/extended_lines.jpg width="700">

So, I fixed the minimum and maximum slope ranges; right lanes' slopes are over 0.5 and the lanes of left lanes are under -0.5, because  the camera is mounted in a fixed position on the car and lane lines will have similar and general slope values. Then I averaged the top and bottom positions of each of the left and right lane segments, and changed the thickness of the lines. By doing that, I can only draw one extend line from bottom to top for each left and right lane on the original image.

<img src=https://github.com/liprin1129/Self_Driving_Car/blob/master/Project/Project1/CarND-LaneLines-P1/submission/image/extended_line.jpg width="700">

###2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when there are lone cracks on road between left and right lanes. I eliminated objects only outside of the quadrilateral mask. So, this system may recognise the crack as a lane line and it will average those lines. As a result, breath of the lanes will be smaller than correct lines.

###3. Suggest possible improvements to your pipeline

A possible improvement would be to make an another quadrilateral mask inside lanes, and it will eliminate any cracks or objects on road between left and right lanes.