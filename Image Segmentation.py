import cv2 
import numpy as np
import imutils

class CORAL_OPERATIONS:
    def __init__(self, img):
        self.image = img

    def Segmentation_KMeans(self, n_clusters = 3):
        """
        This Function do the segmentation to the entered images just to make it easier to perform the tasks of the coral reef
        """
        ## firstlty declare the entered image
        ## convert to RGB ==> ESSENTIAL
        #image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        ## reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = self.image.reshape((-1, 3))
        ## convert to float with 32 bit we don't need 64 bit
        pixel_values = np.float32(pixel_values)
        ## Well, we going to cheat a little bit here since this is a large number of data points,
        ## so it'll take a lot of time to process,
        ## we are going to stop either when some number of iterations is exceeded (say 100),
        ## or if the clusters move less than some epsilon value (let's pick 0.2 here)
        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 190, 1) ## we can use another numbers as we like
        ## it's supposed to work with only 3 clusters [pink ==> alive, white ==>, gray==> background]
        ## but for somehow it maybe will not work proberly bec of the background is not flat so  we can use 4 or 5 clusters
        _, labels, (centers) = cv2.kmeans(pixel_values, n_clusters, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        ## convert back to 8 bit values
        centers = np.uint8(centers)
        ## flatten the labels array
        labels = labels.flatten()
        ## convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]
        ## reshape back to the original image dimension
        segmented_image = segmented_image.reshape(self.image.shape)
        return segmented_image

    def HSV(self, arrLow, arrHigh):
        """
        We've decided to detect the dead and alive layers by HSV method as it gives pretty good results till now on images above water 
        """
        ## firstly we need to convert the entered image 
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        ## you need to enter the range of colors you're seeking for after many tests
        COLOR_MIN_ARR = np.array(arrLow)
        COLOR_MAX_ARR = np.array(arrHigh)

        ## apply that as a mask
        mask = cv2.inRange(hsv, COLOR_MIN_ARR, COLOR_MAX_ARR)

        ## after these last steps you need to apply erosion and delation
        ##erroded = cv2.erode(mask, (3, 3), iterations=5)
        dilated = cv2.dilate(mask, (13, 13), iterations=15)
        ## traditional step to get the result ,, i mean to only display the targeted color
        result = cv2.bitwise_and(self.image, self.image, mask=dilated)
        return result, dilated

    def DrawContours(bool_image, color, img):
        """
        This Function actually draws boxs on detected layers like dead layers and so on
        """
        ## dilation applies morphological filter to images
        ## it's essential when you're trying to use findContours()
        #bool_image = cv2.dilate(bool_image, (3, 3), iterations=15)
        #bool_image = cv2.erode(bool_image, (1, 1), iterations=15)
        #bool_image = cv2.Canny(bool_image, 0, 0)
        ## apply cv2.findContours
        contours = cv2.findContours(bool_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)[0]
        ## not essential but it's usefull to sort the detected contours
        contours = tuple(sorted(contours, key=len, reverse=False))
        for cnt in contours:
            ## traditional outputs x-> x value on x axis , y-> y value and height and width
            x0, y0, w0, h0 = cv2.boundingRect(cnt)
            area0 = w0 * h0 
            ## just make sure that the area of something we need to detect
            if (area0 >= 500) and ((w0 / h0) > 0.1):
                cv2.rectangle(img, (x0,y0), ((w0+x0), (h0+y0)), color, 2)

    def OutlierRemover(bool_image):
        blacked = np.zeros_like(bool_image)

        dilatedimg = cv2.dilate(bool_image, (50, 50), iterations=40)
        erodedimg = cv2.erode(dilatedimg, (50, 50), iterations=50)

        erodedimg = cv2.erode(erodedimg, (3, 3), iterations=30)
        # dilatedimg = cv2.dilate(erodedimg, (3, 3), iterations=10)

        contours = cv2.findContours(erodedimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = tuple(sorted(contours, key=len, reverse=True))
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            area = w * h
            if (area >= 500) and ((w / h) > 0.1):
                # print(f"{area} {x, y ,w ,h}")
                y = y - 60
                h = h - 30
                x = x - 5
                w = w + 10
                blacked[y:y + h, x:x + w] = 255
        dilatedimg = cv2.dilate(blacked, (5, 5), iterations=20)
        blacked = cv2.erode(dilatedimg, (5, 5), iterations=40)

        return blacked


class FAST_OPERATIONS:
    def __init__(self):
        self.numer_of_saved_images = 0

    def stitch(self, images, ratio=0.75, thresh=20.0, show_matches=False, save_image=True):
        """
        This function is used cleary to wrap up all the functions decalred below and use 'em in the right way to get the stitched image
        Params:
        -------
            query , train normal colored images
            ratio > lowe's ratio default 0.75
            thresh > default 4.0
            show_matches ? > default = false

            1- detectAndDescribe
            2- match_keypoints
            3- wrap Perspective
            4- draw matches

        Returns:
        --------
            ==> result
            ==> matched image
        """
        ## declare the self method to save_image parameter
        self.save_image = save_image
        ## needed for drawMatches
        (imageb, imagea) = images
        ## make sure all the images got the same size
        imageb = imutils.resize(imageb, height=700)
        imagea = imutils.resize(imagea, height=700)

        ## Now time to dig on dirty
        ## firstly Histogram Equalization
        imageB = self.HEqualize(imageb)
        imageA = self.HEqualize(imagea)

        ## step two :)
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        ## get the Homography matrix
        M = self.match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio, thresh)
        if M is None:
            return None
        ## we need only H from all of these
        (matches, H, status) = M
        ## as i said it's used for Prespictive Transform of images
        result = cv2.warpPerspective(imagea, H, (imageb.shape[1], imageb.shape[0]))
        ## apply border modifier
        if show_matches:
            visual = self.drawMatches(imageb, imagea, kpsA, kpsB, matches, status)
            self.Image_Saver(result)
            result = self.Border_Modifyer(result)
            return result, visual

        self.Image_Saver(result)
        result = self.Border_Modifyer(result)

        return result

    def match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, thresh):
        """
        Params:
        ------
            kpsA, kpsB: it's supposed that you have already get the key points before applying this function so after that you need to give 'em  as a parameter
            featuresA, featuresA : discriptors from detectAndDescribe
            ratio : lowe's ratio

            1- match between two features .Brute matcher  .knn match
            2- take best matches by applying lowe ratio
            3- finds homography

        Returns:
        -------
            ==> returns none if thers is no homography
            ==> matches
            ==> homography matrix
            ==> status of homography test
        """

        ## it's better to use DescriptorMatcher instead of FLANN in this situation
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        ## it's mandatory to apply knnMatch
        rawmatches = matcher.knnMatch(featuresA, featuresB, 2)

        ## the following loop is traditional to get the best matches
        matches = []
        for m in rawmatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            ## calculate and save the Homography matrix needed for applying Prespictive Transform
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, thresh)

            return matches, H, status

        return None

    def detectAndDescribe(self, image):
        """
        Params:
        ------
            image : the entered image

        Returns:
        -------

            ==> key points as float 32 points (x, y)
            ==> features as struct
        """
        (kps, features) = cv2.SIFT_create().detectAndCompute(image,
                                                             None)  ## extract the keypoints and discriptors from the SIFT Model
        kps = np.float32([kp.pt for kp in kps])

        return kps, features

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        """
        Params:
        ------
            imageA, ImageB : the Target images you wanna get the matches of 'em
            kpsA, kpsB : it's supposed that you have already get the key points before applying this function so after that you need to give 'em  as a parameter
            matches: from the same function that get the kps :)

        Returns:
        --------
             matched images
       """
        ## No need to discuss this function bec it's so simple and basic
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]

        visual = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        visual[0:hA, 0:wA] = imageA
        visual[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):

            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(visual, ptA, ptB, (0, 255, 0), 1)

        return visual

    def HEqualize(self, image):
        ## this is a function that applies Histogram equalization
        self.image = image
        grayed = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ## It is a method that improves the contrast in an image, in order to stretch out the intensity range
        equ = cv2.equalizeHist(grayed)
        colored = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)

        return colored

    def Border_Modifyer(self, image):
        ## firstly you need to call the image you've entered
        self.image = image

        ## set the borders (extra padding to your image).
        stitched_img = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (
        0, 0, 0))  ## BORDER_CONSTANT: Pad the image with a constant value (i.e. black or 0
        ## convert the colors to gray make it easier to apply the methods
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        ## then get the threshold or to be more clear the Binary thresh where split the image into black and white pixels and the target is the black one
        thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
        ## get the contours
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)
        ## create a blank by numpy with the dimensions of the threshed image
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        ## draw a box
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        minRectangle = mask.copy()
        sub = mask.copy()

        ## redo the same steps
        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)
        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        areaOI = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(areaOI)
        stitched_img = stitched_img[y:y + h, x:x + w]

        return stitched_img

    def Image_Saver(self, image):
        ## If we want to save the image you can easily choose save_image =True
        if self.save_image:
            cv2.imwrite(f"ProcessedOutput{self.numer_of_saved_images}.jpg", image)
            print(f'Stiyched Image Saves as ProcessedOutput{self.numer_of_saved_images}.jpg')
            self.numer_of_saved_images += 1

