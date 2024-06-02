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

### Task 2: Retrieve and Display Video Frames
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


