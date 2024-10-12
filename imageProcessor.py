from imageType import ImageType
import numpy as np
import cv2
from PyQt5.QtGui import QImage
from ultralytics import YOLO

class ImageProcessor():
    def __init__(self):
        self.images = []
        self.panorama = None
        self.temp_c = None # for lambda
        self.temp_d = None # for lambda
        self.canny_image = None
        self.dog_image = None
        self.human_detect_image = None

    def loadImage(self, fname):
        image = cv2.imread(fname)
        # PyQt use RGB 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.images.append(rgb_image)
        # self.displayImage()
    
    def getImage(self,  image_type: ImageType):
        if image_type == ImageType.PANORAMA:
            img = self.panorama
            return self.displayImage(img)  # Return QImage ready for display
        if image_type == ImageType.CANNY_RESULT:
            img = self.canny_image
            return self.displayImage(img)
        if image_type == ImageType.DoG_RESULT:
            img = self.dog_image
            return self.displayImage(img)
        if image_type == ImageType.TEMP_C:
            img = self.temp_c
            return self.displayImage(img)
        if image_type == ImageType.TEMP_D:
            img = self.temp_d
            return self.displayImage(img)
        if image_type == ImageType.HUMAN_DETECTION:
            img = self.human_detect_image
            return self.displayImage(img)
        

     # To show image in QLable we have to change 
    def displayImage(self, image):
        qformat = QImage.Format_Indexed8

        if len(image.shape) == 3:
            if(image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0],image.strides[0], qformat)

        # image.shape[0] is the number of pixels in the Y direction.
        # image.shape[1] is the number of pixels in the X direction.
        # image.shape[2] stores the number of channels representing each pixel.

        # img = img.rgbSwapped() # efficiently converts an RGB image to a BGR image..
        return img
        

    def stitching(self):
        imgs = self.images
        stitcher = cv2.Stitcher_create()
        status, result = stitcher.stitch(imgs)
        if status == 0:
            self.panorama = result
            return 1, "🎉Your Panorama is ready!!!"
        if status == 1:
            return 0, "It seems we need more images to create a complete panorama.🫣\nPlease add more images and try again! 😊"
        if status == 2:
            return 0, "Oops! We couldn't create panorama image☹️. \nMake sure the images overlap enough and try again"
        if status == 3:
            return 0, "We need more accurate images and information to proceed.👾\nPlease check your input and try again."
        

    def apply_canny(self, threshold):

        gray_img = cv2.cvtColor(self.panorama, cv2.COLOR_RGB2GRAY)

        if threshold is None:
            # If ther's no passed threshold value (if we called function for the default view) > calculate it from median
            median = np.median(gray_img)
            lower_threshold =  max(median - median * 0.3 , 0)
            upper_threshold =  min(lower_threshold * 3 ,255)
        else:
            # User choose lower threshould by slider and we calculate upper one based on OpenCV documentation
            lower_threshold = threshold
            upper_threshold = threshold * 3

        print(f'Upper Threshold = {upper_threshold} \nLower Threshold = {lower_threshold}')

        # Apply automatic Canny edge detection using the computed thresholds
        canny = cv2.Canny(gray_img, lower_threshold, upper_threshold)
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

        if threshold is None:
            self.canny_image = canny
        self.temp_c = canny

        return lower_threshold, f' 🟣 Lower Threshold = {int(lower_threshold)}  🟣 Upper Threshold = {int(upper_threshold)}'
    
    def apply_DoG(self):
        gray_img = cv2.cvtColor(self.panorama, cv2.COLOR_RGB2GRAY)
        # Convolve the gray resized image with two gaussians (one with a scale of 1 and another with a scale of 3)
        gaussian_1 = cv2.GaussianBlur(gray_img, (31,31), 3)
        gaussian_2 = cv2.GaussianBlur(gray_img, (31,31), 5)

        # find the difference of gaussians and save it >>>>>>>.astype(np.float32) 
        DoG = gaussian_1- gaussian_2
        self.dog_image = DoG
        
    def update_DoG(self, size, shape, operation_type):
        # Determine the shape for the structuring element
        if shape == "MORPH_RECT":
            morph_shape = cv2.MORPH_RECT
        elif shape == "MORPH_ELLIPSE":
            morph_shape = cv2.MORPH_ELLIPSE
        elif shape == "MORPH_CROSS":
            morph_shape = cv2.MORPH_CROSS
        mph_mask = cv2.getStructuringElement(morph_shape, (size, size))
        if operation_type == "Closing":
            self.temp_d = cv2.morphologyEx(self.dog_image, cv2.MORPH_CLOSE, mph_mask)
            # Convert to RGB to show 
        else:
            self.temp_d = cv2.morphologyEx(self.dog_image, cv2.MORPH_OPEN, mph_mask)
        self.temp_d = cv2.cvtColor(self.temp_d, cv2.COLOR_GRAY2RGB)

        return size, shape
        

        
         

       
