import os ## here it's used to create and open folder files
import glob ## used to manipulate the data through folders easily
import cv2 as cv
import numpy as np

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

    def StereoCameraCalibration(self, directory='CalibratedImages'):
        """
        The process of calibrating Stereo camera is much likely to the Normal camera but instead of using one view in the 2D plane ,
        we're going to split 'em into left and right 
        """  
        self.__create_dir(directory)  
        ## now we're preparing obj points like [(0,0,0), (1,0,0),..etc]
        objp = np.zeros((self.desired_size[0] * self.desired_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.desired_size[0], 0:self.desired_size[1]].T.reshape(-1,2) ## special type of numpy array that creates a 2d array with similar values
        objp = objp * 20 ## that's actually refers to two cm between the individuall fields and can be modified later depends on the Images

        objPoints = [] ## 3D points in real world space
        imgPoints_R = [] ## 2D points in Image plane ==> right
        imgPoints_L = [] ## 2D points in Image plane ==> Left

        ## collecting and locating the left images
        images_Left = glob('left/*.png') ## returns list of files or folders that matches the path specified in the pathname argument.
        ## our images will be stored in that variable
        ## now for right
        images_Right = glob('right/*.png')


        ## we are going to use zip method to iterate through the two folders at the same time
        for left, right in zip(images_Left, images_Right):
            ## to apply the calibration we need to convert the images into gray scale so that it's much easier 
            img_l = cv.imread(left)
            img_r = cv.imread(right)
            gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
            gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)

            ##Now find the corners for both
            ret_l, corners_l = cv.findChessboardCorners(gray_l, self.desired_size, None)
            ret_r, corners_r = cv.findChessboardCorners(gray_r, self.desired_size, None)
            
            ## checking if it've found corners so it should add them to the list of imgPoints and objPoints after refining 'em
            if ret_l and ret_r == True:
                objPoints.append(objp)
                ## we're finding the sub pixels in order to get more accurate results
                corners_l = cv.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), self.criteria) ##The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as shown on the figure below.
                imgPoints_L.append(corners_l)
                ## for the right images that time
                corners_r = cv.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), self.criteria) ##The function iterates to find the sub-pixel accurate location of corners or radial saddle points, as shown on the figure below.
                imgPoints_R.append(corners_r)
            
                cv.drawChessboardCorners(img_l, self.desired_size, corners_l, ret_l)
                cv.imshow('Left Image', img_l)
                cv.drawChessboardCorners(img_r, self.desired_size, corners_l, ret_r)
                cv.imshow('Right Image', img_r)
                cv.waitKey(0)

            cv.destroyAllWindows()

            ## now the spicy part is coming ... we're going to apply calibration now..
            ## the last None we've entered refers to the flags -- if you know the focal length if the used camera you can apply it
            ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv.calibrateCamera(objPoints, imgPoints_L, gray_l.shape[::-1], None, None) ##  We can use the function, cv.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors
            height_l,  width_l, channels_l = img_l.shape[:2]
            ## the new camera matrix based on the free scaling parameter
            newcameramtx_l, roi_l = cv.getOptimalNewCameraMatrix(mtx_l, dist_l, (width_l,height_l), 1, (width_l,height_l))

            ## for the right that time
            ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv.calibrateCamera(objPoints, imgPoints_R, gray_r.shape[::-1], None, None) ##  We can use the function, cv.calibrateCamera() which returns the camera matrix, distortion coefficients, rotation and translation vectors
            height_r,  width_r, channels_r = img_r.shape[:2]
            ## the new camera matrix based on the free scaling parameter
            newcameramtx_r, roi_r = cv.getOptimalNewCameraMatrix(mtx_r, dist_r, (width_r,height_r), 1, (width_r,height_r))

            ## next Stereo Vision Calibration
            ## Stereovision techniques use two cameras to see the same object.
            ## we're going to fix intreinsic camera matrix
            flags = 0
            flags = (cv.CALIB_FIX_PRINCIPAL_POINT | cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_FOCAL_LENGTH | cv.CALIB_FIX_INTRINSIC)

            ##transformation between two cameras and calculate essential and fundamental matrix
            Stereo, NewCamMt_l, dist_L, NewCamMt_r, dist_R, rot, trans, essentailMat, FundamentalMat = cv.stereoCalibrate(objPoints, imgPoints_L, imgPoints_R, newcameramtx_l, dist_l, newcameramtx_r, dist_r, gray_l.shape[::-1], self.criteria, flags)

            ## the final step of performing rectification and undistortion   
            rect_l ,rect_r, ProjMat_l, ProjMat_r, Q, roi_l, roi_r = cv.stereoRectify(NewCamMt_l, dist_L, NewCamMt_r, dist_R)
            ## performing mapping
            ## The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap. The undistorted image looks like origina
            stereoMap_L = cv.initUndistortRectifyMap(NewCamMt_l, dist_L, rect_l, ProjMat_l, gray_l.shape[::-1], cv.CV_16SC2)
            stereoMap_R = cv.initUndistortRectifyMap(NewCamMt_r, dist_R, rect_r, ProjMat_r, gray_r.shape[::-1], cv.CV_16SC2)
                
            print("Saving parameters!")
            cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

            cv_file.write('stereoMapL_x',stereoMap_L[0])
            cv_file.write('stereoMapL_y',stereoMap_L[1])
            cv_file.write('stereoMapR_x',stereoMap_R[0])
            cv_file.write('stereoMapR_y',stereoMap_R[1])

            cv_file.release()