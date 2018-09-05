# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/HOG.png
[image4]: ./output_images/pipeline_test.png
[video1]: ./output_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. HOG features extraction

The code for this step is contained in the second cell of the IPython notebook.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Final choice of HOG parameters

I tried various combinations of parameters aiming at maximizing the accuracy of the classifier on the test set.
For example, increasing the orientations parameter from 9 to 11 produced a drop in the classifier accuracy from 0.965 to 0.947. Similarly, decreasing this parameter from 9 to 7 produced a drop in the classifier accuracy from 0.965 to 0.941. Hence I settled for an orientations parameter of 9.
Using the same approach (varying one parameter at a time), I found the best combination was the following :

`color_space = 'HLS'`
`orient = 9  # HOG orientations`
`pix_per_cell = 8 # HOG pixels per cell`
`cell_per_block = 2 # HOG cells per block`
`hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"`
`spatial_size = (16, 16) # Spatial binning dimensions`
`hist_bins = 16    # Number of histogram bins`
`spatial_feat = True # Spatial features on or off`
`hist_feat = True # Histogram features on or off`
`hog_feat = True # HOG features on or off`

#### 3. Classifier training

I trained a linear SVM using 80% of the available dataset. The remaining 20% was reserved for testing.
Note that the usage of ALL hog channels dramatically incresead the size of the feature vector, slowing down the training time. I decided to keep ALL channels because it also increased the accuracy to more than 98.9% and the training is done only once. There is however a downside to this : the extraction of HOG features from the frames in the later stages of the project will be slower.

### Sliding Window Search

#### 1. Windows and scales

I decided to search 3 window positions at 3 scales all over the image.
The code for this is located in the 9th code cell of the IPython notebook :
`   # Search window for far vehicles
    ystart = 380
    ystop = 540
    scale = 1.2
    out_img, box_list1 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    
    # Seach window for vehicles at intermediary distance
    ystart = 400
    ystop = 600
    scale = 1.4
    out_img, box_list2 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        
    # Search window for close vehicles
    ystart = 400
    ystop = 650
    scale = 1.7
    out_img, box_list3 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
`
 
The 3 windows above all have some overlap with the others. Each window is supposed to be more accurate in finding the vehicles depending on their proximity to our car. Indeed, the first window with a scale of 1.2 searches for cars in the top of the road while the second the second window  (scale of 1.4) searches for cars in the intermediate range and finally the 3rd window searches for close vehicles. As an example, the scale of 1.7 was chosen for this 3rd window as the close cars are expected to measure around 110 pixels (110p / 1.7 = 64.7, which is more or less the size of our patches).
Note that the more windows we use, the more detections happen including false positives. This means the filtering of the heatmap has to be stricter.

#### 2. Pipeline and classifier performance

Ultimately I searched on 3 scales using HSV all channels HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
![alt text][image4]
To optimize the performance of the classifier, I have tried several search windows with different scales. Using 5 scales gave me a better result in identifying cars but considerably slowed down the search. Hence, I decided to keep only 3 scales.

---

### Video Implementation

#### 1. Video output
Here's a [link to my video result](./output_video.mp4)


#### 2. Filter false positives

In order to reduce the false positives, I used 2 techniques : heatmaps and historic detections checks.

The heatmaps filtering is implemented in the 9th cell of the IPython notebook (the related functions are in the 6th cell).
Using a threshold of 4, I am able to eliminate a significant number of false positives. The drawback of increasing this number is that real positives (cars) start to be missed with a threshold greater than 4.
Using `scipy.ndimage.measurements.label()`, I identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

The second technique to reduce false positives is to assess if a box corresponds to a real detection by checking if this box was detected in the previous frames. This is implemented in the 8th code cell of the notebook by defining the class `Detection()`. This class stores the coordinates of the boxes found in the last 10 frames. For each new frame, I check if the newly detected boxes are in the vicinity of the previously detected boxes. This vicinity is determined by the distance of the centroids of the boxes. If it is less than the parameter `margin` (40 pixels), the box is considered to be in the vicinity of the compared box.
We look at the 10 previous frames and if boxes are found in the vicinity of the current box in at least 5 of the 10 frames, then the box is considered consistent and will get drawn in the final image.


---

### Discussion


The most difficult part in the project was trying to reduce false positives without affecting the detection of the vehicles. The cycle of tuning parameters, debugging and fixing parts of the video, generating and checking the full video was tedious and time consuming, especially that fixing a part of the video often broke another one.
Eventually, the final parameters seem to produce a good result where false positives appear briefly at 2 positions only.
If I were going to pursue this project further, one of the improvements that could be made is to make the pipeline more robust by averaging the heatmaps over the past frames.

