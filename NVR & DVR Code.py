import cv2 as cv ## the main package we're using in this stuff
import cvui ## A (very) simple UI lib built on top of OpenCV drawing primitives.


##--------->VIP<---------
## we're using HIKVISION DVR 8 Channels DS-7208HQI-K1(/P) :)
## ===> Default IP adress : 192.168.1.64


## the username and password of the DVR we can find it at the time of login
## but now i'll keep it blank
##---->NOT COMPLETED<---
rtsp_username = "admin"
rtsp_password = "admin000"
ip = "169.254.108.159" ## here need to assign the ip address
rtsp_port = "554"
##---->NOT COMPLETED<---

## Line stream got from CCTV camera
def Stream(channel:int):
    ## we should cast it to string
    channel = str(channel)
    ## channel means the Camera number 
    ## the traditional style of the RTSP 
    ## so that we need to make sure to correctlty enter the username and password
    rtsp = "rtsp://" + rtsp_username + ":" + rtsp_password + f"@{ip}:{rtsp_port}/ISAPI/Streaming/channels/101/video"
    ## the rest of this function is to open the video by a traditional lines of code used many times by OpenCV
    cap = cv.VideoCapture()
    cap.open(rtsp)
    cap.set(3, 640)  # ID number for width is 3
    cap.set(4, 480)  # ID number for height is 480
    cap.set(10, 100)  # ID number for brightness is 10qq
    return cap

# Create a function 'nothing' for creating trackbar as it's mandatory
def nothing(x):
    pass

## declaring the function
video = Stream(2) ## we're using camera 1
cvui.init('screen')

## to make it easier to maipulate we are declaring the width and the height here outside
width = 800
height = 400
## declaring the camera number as well to make it more simple :)
cam_no = 1

while cv.waitKey(1) & 0xFF == 27: ## if you pressed esc the stream will close
    ## firstly read the video
    ret, cap = video.read()
    ## we'll open the video on 800x400 quality
    dim = (800, 400) ## that are the dimensions of the video that we choose
    Full_frame = cv.resize(cap, dim, interpolation=cv.INTER_AREA) ## applying the choosen dimensions
    cv.namedWindow('screen', cv.WINDOW_NORMAL) ## used to create a window with a suitable name and size to display images and videos on the screen
    cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN) ## GUI
    ## Creating trackbars for color change
    cv.createTrackbar('color_track', 'screen', 0, 255, nothing)
    ## Get current positions of trackbar
    color = cv.getTrackbarPos('color_track', 'image')

    ## the command cvui.button is used to place the button at a required place 
    ## and the command cvui.mouse is used to detect the mouse click
    if (cvui.button(Full_frame, width - 100, height - 40, "Next") and cvui.mouse(cvui.CLICK)):
        ## Buttons will return true if they were clicked, which makes
        ## handling clicks a breeze.
        print("Next Button Pressed")
        cvui.init('screen')
        ##  based on the button we either increase or decrease cam_no count 
        ## and then delete the existing cam and create a new cam 
        cam_no = cam_no+1
        if (cam_no >4):
            cam_no = 1
        del cam
        cam = Stream(str(cam_no))
    ## intercgangable loops between the next press and the previous one
    if (cvui.button(Full_frame, width - 200, height - 40, "Previous") and cvui.mouse(cvui.CLICK)):
        print("Previous Button Pressed")
        cvui.init('screen')
        cam_no = cam_no - 1
        if (cam_no<1):
            cam_no=4
        del cam
        cam = Stream(str(cam_no))
    ## now showing the stream and that's it :)
    cv.imshow('screen', Full_frame)
    cv.waitKey(0)