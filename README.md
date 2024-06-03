# Coursera: OpenCV and NumPy
My progress through the guided, introductory projects provided by Coursera using OpenCV and NumPy

## Project 1 - [Analyzing Video with OpenCV and Numpy](https://www.coursera.org/projects/analyzing-video-opencv-numpy)
### Task 1: Define a Generator for Reading a Video
```python
def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        retrieved, frame = video.read() # Obtain video frame and whether or not successfully retrieved
        if retrieved:
            yield frame # Return frame (come back to function later)
        else:
            break
    video.release() # Release resources used for video
    yield None # Return None
```
This process was simple enough. The function first opens a given video file using the cv2 library. While this video is opened, frames are read and retrieved during every iteration until the end of the video is reached. When a file is retrieved, denoted by the boolean value "retrieved", the frame is taken elsewhere, and the state of the function is saved. I don't use the "yield" keyword often in Python, but my guess is that yield is used in this instance to return to the function and continue grabbing frames of the video from where it left off.

### Task 2.1: Retrieve and Display Video Frames
```python
for f in get_frames(VFILE):
    if f is None: # End of video
        break
    cv2.imshow('frame',f) # Show frame
    if cv2.waitKey(10) == 27: # Manually stop loop by pressing ESC (27) - (10ms delay)
        break
cv2.destroyAllWindows() # Release resources
```
Interestingly, this for-loop uses the previous function as the driving condition for the loop. I assume the loop would have run indefinitely if not for the condition "if f is None then exit the loop". The rest of the loop plays out each frame on a separate window. The loop is also programmed to stop if the ESC key is pressed on the keyboard, otherwise, the video will play for the full duration before closing.

### Task 2.2: Define a function to obtain a single frame
```python
def get_frame(filename,index):
    count = 0
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        retrieved, frame = video.read() # Obtain video frame and whether or not successfully retrieved
        if retrieved:
            if count == index: # Is true when the desired frame # is found
                return frame # Return frame (come back to function later)
            count += 1 # otherwise, iterate and continue
        else:
            break
    video.release() # Release resources used for video
    return None # Return None
```

Pretty straightforward. The function is almost identical to that of Task 1. The only difference is that it's searching for one singular frame.

### Task 2.3: Examining Pixels
```python
frame = get_frame(VFILE, 80)
print('shape', frame.shape)
print('pixel at (0,0)', frame[0,0,:])
print('pixel at (150,75)', frame[150,75,:])
```
```shell
shape (480, 640, 3)
pixel at (0,0) [47 19  0]
pixel at (150,75) [150 127  86]
```

This snippet of code is pretty interesting. By using the previous function, the frame can be stored in a NumPy array. This array has data such as the shape of the video, aka the dimensions, and the color values of each pixel on the frame. I would assume that the color values were stored in RGB format, but it seems that OpenCV actually uses BGR instead.

### Task 2.4 & 3.1: Displaying and Fixing an image
```python
plt.imshow(frame) # Display image
```
![image](images/RGB-Image.png)

(2.4) As mentioned earlier, OpenCV uses BGR formatting instead of RGB formatting. This causes the original frame to be plotted with tints of red in MATLAB.

```python
fix_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Change color convention
print('pixel at (0,0)', fix_frame[0,0,:]) # Was [47 19 0]
plt.imshow(fix_frame) # Display image
```
```shell
pixel at (0,0) [ 0 19 47]
```
![image](images/BGR-Image.png)

(3.1) The color convention can be changed to RGB using the shown function from the OpenCV library (cv2). This also reverses the previous color value array found at pixel (0,0) to change the format from BGR to RGB.

### Task 3.2 & 3.3: Cropping and Displaying
```python
plt.imshow(fix_frame[240:480, 320:640]) # Crops video frame
```
![image](images/Cropped-Image.png)

(3.2) This line of code simply crops the frame by only displaying the pixels in the designated slice of the array. Simple enough.
```python
darker = 0.5 * fix_frame # Reduces all color values by 50%
darker = darker.astype(np.uint8)
plt.imshow(darker)
```
![image](images/Darker-Image.png)

(3.3) To change the image brightness, the original frame is just multiplied by a percentage constant. However, the variable needs to be converted back to the uint8 data type before displaying. 

### Task 4.1: Displaying a Circle
```python
frame - get_frame(VFILE, 200)
cv2.circle(frame,
          center = (200,200),
          radius = 50,
          color = (0,0,255),
          thickness = 10
          ) # Places circle on frame
fixed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Converts from BGR to RGB format
plt.imshow(fixed_frame) # Displays frame
```
![image](images/Circle-on-Frame.png)

This snippet of code looks long, but really it's mostly just the basic attributes of the circle. Then the frame is displayed as normal.

### Task 4.2: Video Processing - Frame Counter
```python
counter = 0
for frame in get_frames(VFILE):
    if frame is None:
        break
    cv2.putText(frame,
               text= "Frame " + str(counter),
               org = (100,100),
               fontFace = cv2.FONT_HERSHEY_SIMPLEX,
               fontScale = 1,
               color = (0,255, 0),
               thickness = 1
               ) # Places text on frame as "Frame 'counter'"
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27: # Manual Break on ESC key press
        break
    counter += 1
cv2.destroyAllWindows()
```
![image](images/Frame-Counter.png)

Again, this block of code looks lengthy at first, but it's really straightforward the more I look at it. Using the get_frames function I wrote, every frame is processed and a Frame counter is added near the top right of the video.

### Task 5.1: Generating Video File
```python
fourcc = cv2.VideoWriter_fourcc('M','P','4','V') # MP4V Format Output
video_out = cv2.VideoWriter("new.mp4",fourcc,20,(640,480)) # (filename,fourcc,frames per sec, resolution)

counter = 0
for frame in get_frames(VFILE):
    if frame is None:
        break
    cv2.putText(frame,
               text= "Frame " + str(counter),
               org = (100,100),
               fontFace = cv2.FONT_HERSHEY_SIMPLEX,
               fontScale = 1,
               color = (0,255, 0),
               thickness = 1
               ) # Places text on frame as "Frame 'counter'"
    video_out.write(frame) # Write frame to output
    counter += 1
video_out.release() # Release resources used to write video
```

This block of code is just a recreation of the previous one. However, I can now save video files in an MP4 format by writing each frame into an output file. This will work great for any future video processing projects.

### Task 5.2: Frame count
```python
video = cv2.VideoCapture(VFILE)
count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video.release()
print('Frame Count',count)
```
```shell
Frame Count 3597
```

Lucky for me, there is a function that returns the total frame count of the video. However, I could imagine that the frame count could also be obtained via the counter in the get_frames() loop.

### Task 6: Creating a Collage
```python
skip_frames = count // 15  # Number of frames to skip before image capture

frames = []
counter = 0
for f in get_frames(VFILE):
    if counter % skip_frames == 0: # is true at intervals of skip_frames
        frames.append(f) # Add frame to list
    counter += 1

row1 = np.concatenate(frames[0:5], axis = 1)
row2 = np.concatenate(frames[5:10], axis = 1)
row3 = np.concatenate(frames[10:15], axis = 1)
collage = np.concatenate((row1,row2,row3), axis = 0)
collage = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
plt.imshow(collage)
```
![image](images/Collage.png)

The block of codes takes the video and turns it into a collage of images from the video. The count is initially divided by 15 to split the collection of frames into 15 images. This is understandable as the code will take a total of 15 frames over the course of the entire video. Using modulus, these frame previews are evenly distributed throughout the video. Fortunately, NumPy makes the collage part easy by combining 5 frames for each row of the collage. Then, as usual, the collage can be displayed as a full image.

## Personal Project - RGB to Black & White Video (Grayscale)
Using what I learned from the guided project, I wanted to try to write a script that could take a colored video and turn it into a grayscale video.
```python
# Output video to disk
fourcc = cv2.VideoWriter_fourcc('M','P','4','V') # MP4V Format Output
video_out = cv2.VideoWriter("output.mp4",fourcc,FPS,RESOLUTION) # (filename,fourcc,frames per sec, resolution)

for f in get_frames(VFILE):
    if f is None:
        break # End of Video
        
    # Convert frame
    f = np.dot(f, np.array([0.114,0.5870,0.2989]))
    f = f.astype(np.uint8) # Converting units back to uint8
    
    # Expand to 3-Channel Grayscale
    f = np.repeat(np.expand_dims(f, axis=2), 3, axis=2)
    
    
    video_out.write(f) # Write frame to output
    
video_out.release() # Release resources
```

I accomplished this by using luminance weights to change the color of each frame. I then saved all of these frames and produced a grayscale version of the original video.

![image](images/Mario.gif) ![image](images/Mario-output.gif)

## Project 2 - [Recognizing Shapes in Images with OpenCV](https://www.coursera.org/projects/recognizing-shapes-images-opencv)

### Task 1: Load an image from file
The first task was to view an image file and read its contents.
```python
def view_image(image):
    cv2.imshow('view',image) # Displays image
    cv2.waitKey(0) # Wait until a key is pressed
    cv2.destroyAllWindows() # Close image
```

The script is straightforward and looks familiar to the one used while displaying frames.

```python
image = cv2.imread(f) 
view_image(image)

print(image.shape) # (Rows, Columns,BGR-Value)
print(image[0,0,:]) # BGR Value
```
```shell
(640, 427, 3)
[22 24  4]
```
![image](images/subway.jpg)

Running the function of course displays the image. Interestingly but not shockingly, the image can also be stored similarly to frames. Data can also be read from images in the same matter. I imagine that manipulating the image contents will work the same way.

```python
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to gray-scale
print(gray_image.shape)
print(gray_image[0,0])
view_image(gray_image)
```
```shell
(640, 427)
18
```
![image](images/grayscale-subway.png)

The image can be converted to gray-scale using the functions provided by the OpenCV. While working on the small personal project I previously mentioned, I had suspected that there was most likely a function that did what I was trying to do. However, the purpose of the project was to try and do most of it on my own rather than relying on the built-in functions provided. It was still enjoyable to do and a learning experience for me.

### Task 2: Compute image gradients
```python
# Takes gradient on the horizontal axis
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0) # Finds the rate of change (derivative) of an image
abs_sobelx = np.absolute(sobelx) # Obtain Magnitude of derivatives
view_image(abs_sobelx / np.max(abs_sobelx)) # Normalize image and display
```
![image](images/x-gradient.png)

After converting the images to a gray-scale, the next step of the script is to compute the gradients of the image. From my understanding, the derivative of the image is taken to highlight the areas of greatest change in an image. After normalization, the edges of the image can be seen clearly. I can already see how this preps things for edge-detection.

```python
# Takes gradient on the horizontal axis
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0) # Finds the rate of change (derivative) of an image
abs_sobelx = np.absolute(sobelx) # Obtain Magnitude of derivatives
view_image(abs_sobelx / np.max(abs_sobelx)) # Normalize image and display
```
![image](images/y-gradient.png)

In a similar fashion, the y-gradient of the image is also taken using the Sobel() function from the OpenCV library. Since the derivative is being taken on the vertical y-axis, edges that are parallel to the y-axis don't appear due to the rate of change being 0 (Black in color value). It's vice versa for the x-gradient. I assume that if we can combine the two, we can produce an image with all of the edges highlighted.

```python
# Combining both gradients
magnitude = np.sqrt(sobelx**2 + sobely ** 2)
view_image(magnitude / np.max(magnitude)) # Normalize image and display
```
![image](images/magnitude.png)

Sure enough, finding the magnitude of the image produces the combined result of both images.

### Task 3: Detect edges in an image
```python
# Pixels below 50 brightness are no longer considered edge pixels
# Pixels above 150 brightness are certainly edge pixels
edges = cv2.Canny(gray_image, 50, 150) # Edge-detection threshold
view_image(edges)
```
![image](images/canny.png)

Using the Canny() function provided by the OpenCV library, the x and y gradients can be used to further define the edges of an image. Values representing the brightness level for each pixel can be entered to filter out details in the image. Normally, the image will have every edge highlighted, but this is too much unnecessary data to process. So filtering out these "softer" edges can simplify the image more for use.

### Task 4: Recognize lines in an image
### Task 5: Recognize circles in an image

