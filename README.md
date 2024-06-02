# Coursera-OpenCV-NumPy
My progress through guided, introductory projects using OpenCV and NumPy

## Project 1 - Analyzing Video with OpenCV and Numpy
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

