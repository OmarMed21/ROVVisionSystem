##______________________________________*****************WATCH OUT NIGGER******************____________________________________________
## The main Purpose of creating this Library is to facilate the usage of frequent Algorithms and Models to be all in a single place and 
## you just need to import the Class and the Function that contains the Model you wanna use and that's all , you don't even need to    
## import the images or read'em or even create the cv.waitkey() ,it's all here don't worry :)
## Made and Created by : Omar Medhat
## Date of Creation : 31.08.2022
##_____________________________________________________________________________________________________________________________________

from glob import glob
import cv2 as cv 
import numpy as np
import os
import imutils
import math

class THE_MAIN_FUNCTIONS:
    '''
    This class contains all the Function that we'll need to process the Images 
    Args:
    -----
            imgs: list of images entered by the user in the form of strings 
            WantToResize: if you feel like your image is so large so you need to resize it before we apply anything, it'll normally resized to the half of it's size
            ResizeThirdSize: option 1 to resize the image to the third of it's size
            ResizeFourthSize: option 2 to resize the image to the fourth of it's size
            Model: you should firstly write down the model you wanna use
            Matcher: the Matcher or the appropiate matcher ,it's orignally FLANN : Fast Library for Approximate Nearest Neighbors
    '''
    def __init__(self, imgs = list(), Model='SIFT', Matcher='FLANN',  WantToResize = False, ResizeThirdSize=False, ResizeFourthSize=False):
        self.imgs = imgs
        self.lst = []
        self.Matcher = Matcher
        self.Model = Model

        ## Setting SIFT MODEL
        if self.Model.upper() == 'SIFT':
            self.Model = cv.SIFT_create()
            if self.Matcher.upper() == 'DS' or self.Matcher.upper() == 'DESCRIPTOR MATCHER':
                self.Matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_FLANNBASED)
            elif self.Matcher.upper() == 'BF' or self.Matcher.upper() == 'BRUTE FORCE':
                self.Matcher = cv.BFMatcher(cv.NORM_L1, True)
            else:
                self.Matcher = cv.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks = 150))
                
        else:
            ## Setting ORB MODEL
            self.Model = cv.ORB_create()
            if self.Matcher.upper() == 'DS' or self.Matcher.upper() == 'DESCRIPTOR_MATCHER':
                self.Matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_FLANNBASED)
            elif self.Matcher.upper() == 'BF' or self.Matcher.upper() == 'BRUTE FORCE':
                self.Matcher = cv.BFMatcher(cv.NORM_HAMMING, True)
            else:
                self.Matcher = cv.FlannBasedMatcher(dict(algorithm = 6, table_number=6, key_size=12, multi_probe_level = 1), dict(checks = 150))

        self.WantToResize = WantToResize
        self.ResizeThirdSize = ResizeThirdSize
        self.ResizeFourthSize = ResizeFourthSize

        ## List of images and they're already read at Cv2
        for i in self.imgs:
            self.lst.append(cv.imread(i))
        self.__checkResizeDemand()

    def __checkResizeDemand(self):
            if self.WantToResize == True:
                for i in range(len(self.lst)):
                    if self.ResizeThirdSize == True:
                        self.lst[i] = cv.resize(self.lst[i], (int(self.lst[i].shape[1]//3),int(self.lst[i].shape[0]//3)), interpolation=cv.INTER_AREA)
                    elif self.ResizeFourthSize == True:
                        self.lst[i] = cv.resize(self.lst[i], (int(self.lst[i].shape[1]//4),int(self.lst[i].shape[0]//4)), interpolation=cv.INTER_AREA)
                    else:
                        self.lst[i] = cv.resize(self.lst[i], (int(self.lst[i].shape[1]//2),int(self.lst[i].shape[0]//2)), interpolation=cv.INTER_AREA)

    def GoodMatchNormal(self, ds1, ds2):
        matcher = self.Matcher
        matches = matcher.match(ds1, ds2)
        matches = sorted(matches, key = lambda x:x.distance)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        return good
    
    def GoodMatchesFLANN(self, ds1, ds2):
        matches = self.Matcher.knnMatch(ds1,ds2,k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in matches:
            if m1.distance < 0.7 * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        return good_matches

    def DisplayFeatureDetectionAndMatching(self, model_name, flags=0, matchesMask=None):
        '''
        This Function works only for both ORB & SIFT models
        You should choose the model to apply on the image you want instead of typing it and calling many functions for each step
        this Function do all of that directly and returns the image within the applied Model on.

        NOTE: we made assume that you have already imported OpenCV as cv not cv2 but you can modify it.

        To save the most amount of time we'll import the Image here also , so you don't need anything at you ,but calling this function

        Args:
        ----
            model_name: WARNING! You should only choose either SIFT OR ORB
            flags: if you're willing to add flags [ACTUALLY I HAVE NO IDEA WHAT DOES FLAGS MEAN AT ALL :)]
            matchesMask: wether you're interessted in applying mask or not
        Returns:
        -------
            It returns the No. of matching Points and The images after detecting the matched points
        '''
        img1 = self.lst[0]
        img2 = self.lst[1]

        try:
            model_used = self.Model
            matcher = self.Matcher
        except NameError:
            print('DUDE CHECK THE DOC STRINGS PLZZZZZ!!!!!')
            return self.displayImageSpecific.__doc__

        finally:
            while True :
                self.kps1, ds1 = model_used.detectAndCompute(img1, None) ## applying the detection and compution methods together first image 
                self.kps2, ds2 = model_used.detectAndCompute(img2, None) ## applying the detection and compution methods together  second image

                if isinstance(matcher, cv.FlannBasedMatcher): ##checking if we're using FLANN or not as it'll differs totally
                    matches = self.GoodMatchesFLANN(ds1=ds1, ds2=ds2)
                    match_images = cv.drawMatchesKnn(img1, self.kps1, img2, self.kps2, matches, None, flags=2)
                else:
                    matches = self.GoodMatchNormal(ds1=ds1, ds2=ds2)
                    match_images = cv.drawMatches(img1, self.kps1, img2, self.kps2, matches, np.array([]), matchesMask=matchesMask)
                ## Now display our output [No. of Matches + Name of the Model + the Images]
                print(f"You've successfully used the function correctly\nYou've used {model_name.upper()} Model\nWe've Found {len(matches)} Matches between the Two Images")
                cv.imshow('The Output Image after applying the Model', match_images)
                if cv.waitKey(0) & 0xff == 27:
                    cv.destroyAllWindows() ## press escape button

    def __create_mask(self,img1,img2,version):
        '''
        Here You wanna create the mask that'll be used later in blending and represent the Final Stitched Image
        img1, img2 : the Two Images that'll be processed but you won't neither enter any arguments here nor call this method
        '''
        img1 = self.lst[0]
        img2 = self.lst[1]

       # Creating the Mask Now
        h1 = img1.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]

        ##Now setting the dimensions of the Panorama
        h_p = h1
        w_p = w1 + w2

        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------
        ## This bunch of code is used to manage the Frame size between Images
        ## and after that represent it in binary form [blank]
        offset = int(250 / 2)
        barrier = img1.shape[1] - int(250/ 2)
        mask = np.zeros((h_p, w_p))
        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------

        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (h_p, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (h_p, 1))
            mask[:, barrier + offset:] = 1
        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------

        return cv.merge([mask, mask, mask])

    def Stitching(self, Op_Name='Panorama.jpg'):
        '''Steps to apply stitching between photos:
            1. Apply a feature detector which you can get common features between images [SIFT]
            2. match the common features together
            3. Find the Homography of the best Matches
            4. Computing the Homography [Source Image Matrix] = [H Matrix](9 UNKOWNS) = [Destination Image Matrix]
            5. We need 4 Matches as Min number of Matching Points
            6. We neet to differntiate between valid and invalid matches [inliers/outliers] ---> RANSAC [RAndom SAmple Cenesus]
            7. Apply Wrap Transform usingestimated Homography Matrix '''
        img1 = self.lst[0]
        img2 = self.lst[1]
        h1 = img1.shape[0]
        w1 = img1.shape[1]
        w2 = img2.shape[1]

        matcher = self.Matcher
        model_used = self.Model

        kps1, ds1 = model_used.detectAndCompute(img1, None) ## applying the detection and compution methods together first image 
        kps2, ds2 = model_used.detectAndCompute(img2, None) ## applying the detection and compution methods together  second image

        if isinstance(matcher, cv.FlannBasedMatcher): ##checking if we're using FLANN or not as it'll differs totally
            matches = self.Matcher.knnMatch(ds1,ds2,k=2)
            good_points = []
            good_matches=[]
            for m1, m2 in matches:
                if m1.distance < 0.7 * m2.distance:
                    good_points.append((m1.trainIdx, m1.queryIdx))
                    good_matches.append([m1])
        else:
            matcher = self.Matcher
            matches = matcher.match(ds1, ds2)
            matches = sorted(matches, key = lambda x:x.distance)
            good_points = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_points.append(m)
                             
        image1_kp = np.float32([kps1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kps2[i].pt for (i, _) in good_points])

        H, status = cv.findHomography(image2_kp, image1_kp, cv.RANSAC, 5.2) ## find the Homography and apply RANSAC
        ##Now setting the dimensions of the Panorama
        h_p = h1
        w_p = w1 + w2 


        panorama1 = np.zeros((h_p, w_p, 3))
        mask1 = self.__create_mask(img1,img2,version='left_image')
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1

        mask2 = self.__create_mask(img1,img2,version='right_image')
        panorama2 = cv.warpPerspective(img2, H, (w_p, h_p))*mask2

        result=panorama1+panorama2
        
        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        ## --------->> I DON'T UNDERSTAND THIS LINES TBH <<----------------

        cv.imwrite(Op_Name, final_result) ## I've used imwrite() instead of imshow() as there's something wrong while showing 
        ## the only difference is that image will be saved on local as Panorama.jpg

    def MultiplePanoramaStitching(self, folder_name = 'panorama_frames'):
        """
        HARD CODED WAY: four try:except loops to perform TREE stitching 
        """
        folder_loc = os.listdir(folder_name)
        ##Debuging
        print(f"We've found {len(folder_loc)} images")
        ## read each image in the folder by specifing it's location
        imgs_lst = []
        for img in folder_loc:
            imgs_lst.append(img)

        try:
            for i in range(10):
                if i <= len(imgs_lst):
	                model= THE_MAIN_FUNCTIONS(imgs=[f"panorama_frames/{imgs_lst[i]}",\
                        f"panorama_frames/{imgs_lst[i+1]}"], Model='SIFT', Matcher='flann').Stitching(f'P{i+1}.png')
                    
        except IndexError: 
            pass
    
        try:
            for j in range(1, 100):
                if j <= (len(imgs_lst) -2):
                    model= THE_MAIN_FUNCTIONS(imgs=[f"P{j}.png",\
                        f"P{j+1}.png"], Model='SIFT', Matcher='flann').Stitching(f'PP{j}.png')
        except AttributeError:
            pass

        try:
            for x in range(1, 100):
                if x <= (len(imgs_lst) -3):
                    model= THE_MAIN_FUNCTIONS(imgs=[f"PP{x}.png",\
                        f"PP{x+1}.png"], Model='SIFT', Matcher='flann').Stitching(f'PPP{x}.png')
        except AttributeError:
            pass

        try:
            for k in range(1, 100):
                if k <= (len(imgs_lst) -4):
                    model= THE_MAIN_FUNCTIONS(imgs=[f"PPP{k}.png",\
                        f"PPP{k+1}.png"], Model='SIFT', Matcher='flann').Stitching('result1.png')
        except AttributeError:
            pass

        print('Panorama Is Successfully Created, Congrats!!')

class READ_VIDEO_FROM_CABLE:
    """
    Params:
    ------
        user_name/password : the user name and password used to define the ethernet cable
        dvr : the IP of used DVR and you need to already know it before you dig .
        camera : choose which camera you'r interessted to show 
    """
    def __init__(self, user_name='admin', password='admin00', dvr='169.254.108.159', camera= 1): 
        self.user = user_name
        self.password = password
        self.ip = dvr
        self.cam = camera

    def ShowVideo(self):
        """
        Returns: 
        -------
            Firstly it will be implemented automatically each time you call this Class to check weather the Video is read or not
            A Video Stream on the Co-Pilot screen through Ethernet Cable connected to the ROV using RTSP(Real Time Streaming Protocol) on his Screen to perform the desired Tasks on.
        """
        reader = cv.VideoCapture(f"rtsp://{self.user}:{self.password}@{self.ip}/ISAPI/Streaming/channels/101/video")

        if (reader.isOpened()):
            print('Congrats, The Video is Successfully accessed !!')
        else:
            print("Oops..Dude We've Failed to Read the Video")
 
        while True:
            _,frame = reader.read()
            cv.imshow('Live Stream', frame)
            ## press Esc button to end the stream
            if cv.waitKey(1) & 0xff == 27:
                break
        reader.release()
        cv.destroyAllWindows()
        
class VIDEO_EXPORT:
      def EXECUTION(self, vid_name='Video.avi', frames=20, link=None):
        """
        This Method is generally used to save the Video captured by the ROV through many Steps explained before already

        Params:
        ------
            vid_name: the name you wanna the video to be saved as .
            frames: the frame rate

        Returns:
        -------
            The Captured Video with a choosen frame rate and name in the form of avi video file 
        """
        reader = cv.VideoCapture(link)

        if (reader.isOpened()):
            print('Congrats, The Video is Successfully accessed !!')
        else:
            print("Oops..Dude We've Failed to Read the Video")

        ##get the height and the width of our video
        width= int(reader.get(cv.CAP_PROP_FRAME_WIDTH))
        height= int(reader.get(cv.CAP_PROP_FRAME_HEIGHT))

        result = cv.VideoWriter(vid_name,\
                        cv.VideoWriter_fourcc(*'DIVX'),\
                         frames, (width,height))

        while True:
            ret,frame = reader.read()
            if ret == True:
                result.write(frame)
                ## press Esc button to end the stream
                if cv.waitKey(1) & 0xff == 27:
                    break
            else:
                break

        reader.release()
        result.release()
        cv.destroyAllWindows()
        print('The Video is Successfully Saved')

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

class VIDEO_OPERATIONS:
    """
    This Class is used to extract specific frames from the Video captured by the ROV under Water to be able to perform our Operations on'em.
    """
    def __init__(self, vid_path, save_dir='save'):
        self.dir = save_dir
        self.vid_path = vid_path

    def __create_dir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            print(f"ERROR: creating directory with name {path}")

    def EXTRACT_FRAMES(self, gap=20):
        save_path = self.dir
        self.__create_dir(save_path)
        video_paths = glob(f"{self.vid_path}/*")

        for path in video_paths:
            cap = cv.VideoCapture(path)
            idx = 0

            while True:
                ret, frame = cap.read()

                if ret == False:
                    cap.release()
                    break

                if idx == 0:
                    cv.imwrite(f"{save_path}/{idx}.png", frame)
                else:
                    if idx % gap == 0:
                        cv.imwrite(f"{save_path}/{idx}.png", frame)
                idx += 1

        print(f'{len(os.listdir(save_path))} Frames have been Successfully Extracted from the Video ')

    def PanoramaOfFrames(self):
        imgs_lst = []
        for img in os.listdir(self.dir):
            imgs_lst.append(img)
        model = THE_MAIN_FUNCTIONS(imgs_lst).MultiplePanoramaStitching(f"{self.dir}/")
        return model

    def MotionDetection(self):
        cap = cv.VideoCapture(self.vid_path)

        MIN_AREA = 120 ## that's actually the best minimum area after many tries
        first_frame = None
        ## create our object detector which idea is based on the absoulte difference of current frame and the first one
        Detector = cv.createBackgroundSubtractorMOG2(100, 40)

        ## Set our Frame Counter
        Frame_Counter = 0
        ## Create our Tracker
        Tracker = EuclideanDistTracker()
        while True:
            ## First reading the video and then we get two outputs , our interest rn is in the output(frame)
            __, frame = cap.read()
            ## Frame Counter
            Frame_Counter += 1
            ## now converting the Frames to Balck n White and then apply gaussian blur to get rid of the Noise and smooth the Images
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (21,21), 3) ## be aware that the kernel size should always be odd integers
            ## we have choosen 21x21 region to help to smooth out high frequency noise that could throw our motion detection algorithm off.

            ## now we need to subtract the first frame and subsequent new frames from the background
            ## let's firstly check the the first frame is not None 
            ## if it's None it'll be directly take the values og gray variable :)
            if first_frame is None:
                first_frame = gray
                continue

            ## compute the absolute difference between the current frame and first frame, it's mathimatical called delta
            delta = Detector.apply(frame)
            ## apply thresholding to classify pixel values to 0 and 1 values [B n W]
            _ , thresh = cv.threshold(delta, 254, 255, cv.THRESH_BINARY) ##If the delta is less than 40, we discard the pixel and set it to black (i.e. background).
            ## If the delta is greater than , 40we’ll set it to white (i.e. foreground)

            ## now it's dialation turn , which is used to removing noise and isolation of individual elements and joining disparate elements in an image
            #dilate = cv.dilate(thresh, None, iterations=4)

            ## contour detection now
            contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            ## cv.RETR_EXTERNAL :  it returns only extreme outer flags , cv.CHAIN_APPROX_NONE :It removes all redundant points and compresses the contour, thereby saving memory
            cnts = imutils.grab_contours(contours) ## the function of imutils.grab_contours, returning counters (contours) in cnts , without distinguishing between opencv2 or opencv3
            ## record our detections in a list
            detections = []
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv.contourArea(c) > MIN_AREA:
                    x, y, w, h = cv.boundingRect(c)
                    detections.append([x, y, w, h])
                    cx = int((x + x + w) /2)
                    cy = int((y + y + h) /2)
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv.circle(frame, (cx,cy), 2, (255,0,0), -1)
                    print('FRAME ', Frame_Counter, " ", x, y, w, h)
            ## Object Tracking
            IDs = Tracker.update(detections)
            for ID in IDs:
                x, y, w, h, id = ID
                cv.putText(frame, str(id), (x, y-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            cv.imshow("Screen", frame)
            if cv.waitKey(1) & 0xff == 27:
                break

        # When everything done, release the capture
        cap.release()
        # Destroy the all windows now
        cv.destroyAllWindows()
        
class CALIBRATION:
    """
    We're going to calibrate our camera using a real world example [Chess Board] that's already exists in OpenCV  
    The Class will be specialized in calibrating Normal Camera and Stereo Type Camera
    """
    def __init__(self, Size_WnH=(24,17), FrameSize=(640,480), criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    ## Now you need to understand , we have two type of points , the first one is Objective Points which exist in the Real World and found in 3D , the second type is Image Points which are 2D points
    ## we need to prepare both of them before we dig deep into the code
        self.desired_size = Size_WnH ## size of your board or object in advance in width and height
        self.frames = FrameSize
        self.criteria = criteria

    def __create_dir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except OSError:
            print(f"ERROR: creating directory with name {path}")

    def NormalCameraCalibration_FindCorners(self, directory='CalibratedImages'):  
        self.__create_dir(directory)  
        ## now we're preparing obj points like [(0,0,0), (1,0,0),..etc]
        objp = np.zeros((self.desired_size[0] * self.desired_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.desired_size[0], 0:self.desired_size[1]].T.reshape(-1,2) ## special type of numpy array that creates a 2d array with similar values

        objPoints = [] #3D points in real world space
        imgPoints = [] #2D points in Image plane

        images = glob('*.png') ## returns list of files or folders that matches the path specified in the pathname argument.
        ## our images will be stored in that variable

        for i in range(len(images)):
            img = cv.imread(images[i])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ##Now find the corners
            ret, corners = cv.findChessboardCorners(gray, self.desired_size, None)
            
            ## checking if it've found corners so it should add them to the list of imgPoints and objPoints
            if ret == True:
                objPoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria) ##The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as shown on the figure below.
                imgPoints.append(corners)
            
                cv.drawChessboardCorners(img, self.desired_size, corners2, ret)
                cv.imshow('Result', img)
                cv.waitKey(0)

            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None) ##  We can use the function, cv.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors
            h,  w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

            # undistort
            dst = cv.undistort(img, mtx, dist, None, newcameramtx)
            #crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
                
            cv.imwrite(f'{directory}/CalibratedImage{i+1}.png', dst)
        cv.destroyAllWindows()