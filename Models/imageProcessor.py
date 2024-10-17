from Models.imageType import ImageType
from Models.commonFunctionality import CommonFunctionality
import numpy as np
import cv2
from PyQt5.QtGui import QImage
# from customize_yolo import custom_YOLO
from ultralytics import YOLO
import matplotlib.pyplot as plt


class ImageProcessor():
    def __init__(self):
        self.images = []
        self.panorama = None
        self.temp_c = None # To view live editing on the same image
        self.temp_d = None # To view live editing on the same image
        self.canny_image = None
        self.dog_image = None
        self.human_detect_image = None
        self.common_func = CommonFunctionality()


    def loadImage(self, fname):
        # Read Image with OpenCV
        image = cv2.imread(fname)
        # PyQt use RGB 
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Add image to images list, which gonna use for stitching process
        self.images.append(rgb_image)
    
    def getImage(self,  image_type: ImageType):
        """
        - Responsible for retrieving various images in a format suitable for display on the interface. 
        - enum to avoid typographical errors when accessing the appropriate image from memory, 
          which is referenced as a property in the object.
        """
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
        """
        Convert an image represented as a NumPy array into a QImage format suitable for display in a PyQt application.
        qformat refers to the specific format that determines how pixel data is interpreted and stored in a QImage object in PyQt.
        Different image formats define how color information is stored in each pixel,
        including the number of color channels and their arrangement.
        """
        # Set default qformat - single channel (grayscale)
        qformat = QImage.Format_Indexed8  

        if len(image.shape) == 3:
            if(image.shape[2]) == 4:
                # This qformat used for images with an alpha channel (transparency).
                qformat = QImage.Format_RGBA8888 
            else:
                # This qformat is suitable for standard RGB images.
                qformat = QImage.Format_RGB888 
        """
        construct QImage object, which can be used for display in the PyQt user interface.
        image.shape[1] and image.shape[0] are used for the image's width and height,
        image.strides[0] provides the number of bytes used for each row of the image data.
        """

        img = QImage(image, image.shape[1], image.shape[0],image.strides[0], qformat)
        return img
        

    def stitching(self):
        """
        Stitches a list of images together to create a panorama.

        This method uses OpenCV's Stitcher to combine the images stored in
        self.images into a single panoramic image. It returns the status of 
        the stitching process along with a corresponding message.
        Returns:
        tuple: A tuple containing:
            - int: Status code (1 if successful, 0 otherwise).
              reduce casses in controller
            - str: Message describing the outcome of the stitching process.
                  - "🎉Your Panorama is ready!!!" if successful.
                  - "It seems we need more images to create a complete panorama.🫣
                    Please add more images and try again! 😊" if more images are needed.
                  - "Oops! We couldn't create panorama image☹️. 
                    Make sure the images overlap enough and try again" if the stitching fails.
                  - "We need more accurate images and information to proceed.👾
                    Please check your input and try again." if the input is not suitable.
    
        """
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
        """
        Applies Canny edge detection to the panorama image.

        This method converts the panorama to grayscale and applies the Canny
        edge detection algorithm using the specified threshold values. If no 
        threshold is provided, it computes the lower and upper thresholds 
        based on the median pixel value of the image.

        Parameters:
            threshold (float): The lower threshold value for Canny edge detection. 
                            If None, thresholds will be calculated automatically.

        Returns:
            tuple: A tuple containing:
                - float: The lower threshold value used for Canny detection.
                - str: A message indicating the calculated threshold values.
                    Format: "🟣 Lower Threshold = {lower} 🟣 Upper Threshold = {upper}".
        """
        # Convert the panoramic image to grayscale for edge detection
        gray_img = cv2.cvtColor(self.panorama, cv2.COLOR_RGB2GRAY)

        if threshold is None:
            # If ther's no passed threshold value (if we called function for the default view) > calculate it from median
            median = np.median(gray_img)
            # Set the lower threshold to 70% below the median, ensuring it's not negative
            lower_threshold =  max(median - median *  0.6, 0)
            # Set the upper threshold to 140% above the median, ensuring it does not exceed 255
            upper_threshold =  min(median + median * 1.4, 255)
        else:
            # If the user has selected a lower threshold via a slider, calculate the upper threshold
            # based on OpenCV's recommendation (three times the lower threshold)
            lower_threshold = threshold
            upper_threshold = threshold * 3

        print(f'Upper Threshold = {upper_threshold} \nLower Threshold = {lower_threshold}')

        # Apply automatic Canny edge detection using the computed thresholds
        canny = cv2.Canny(gray_img, lower_threshold, upper_threshold)

        # Convert the binary edge image back to RGB format for display
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)

        # If no threshold was provided, store the result in canny_image for future use
        if threshold is None:
            self.canny_image = canny

        # Store the current Canny result in a temporary variable for processing and for initial result show
        # this is important for live editing
        self.temp_c = canny

        return lower_threshold, f' 🟣 Lower Threshold = {int(lower_threshold)}  🟣 Upper Threshold = {int(upper_threshold)}'
    
    def apply_DoG(self):
        """
        Applies the Difference of Gaussian (DoG) method to the panorama image.

        This method uses a predefined method from the CommonFunctionality 
        class to apply DoG to the panorama and store the resulting image.

        """
        self.dog_image = self.common_func.apply_DoG(self.panorama)
        
    def update_DoG(self, size, shape, operation_type):
        """
        Updates the Difference of Gaussian (DoG) result with morphological operations.

        This method applies either morphological closing or opening on the 
        DoG image using a specified structuring element shape and size.

        Parameters:
            size (int): The size of the structuring element.
            shape (str): The shape of the structuring element. Can be one of 
                        "MORPH_RECT", "MORPH_ELLIPSE", or "MORPH_CROSS".
            operation_type (str): The type of morphological operation to perform. 
                                Can be either "Closing" or "Opening".

        Returns:
            tuple: A tuple containing:
                - int: The size of the structuring element used.
                - str: The shape of the structuring element used.
        """
        # Determine the shape for the structuring element
        if shape == "MORPH_RECT":
            morph_shape = cv2.MORPH_RECT
        elif shape == "MORPH_ELLIPSE":
            morph_shape = cv2.MORPH_ELLIPSE
        elif shape == "MORPH_CROSS":
            morph_shape = cv2.MORPH_CROSS

        # Create structuring element for morphological operations
        mph_mask = cv2.getStructuringElement(morph_shape, (size, size))

        # Apply morphological operation
        if operation_type == "Closing":
            self.temp_d = cv2.morphologyEx(self.dog_image, cv2.MORPH_CLOSE, mph_mask)
        else:
            self.temp_d = cv2.morphologyEx(self.dog_image, cv2.MORPH_OPEN, mph_mask)
        
        # Convert to RGB to show 
        self.temp_d = cv2.cvtColor(self.temp_d, cv2.COLOR_GRAY2RGB)

        return size, shape
    

    def detect_humans(self, conf):
        """
        Detects humans in the panorama image using the YOLO model.

        This method applies the YOLO detection algorithm to the panorama 
        image and counts the number of detected humans. It returns a message 
        indicating the result of the detection.

        Parameters:
            conf (float): The confidence threshold for detecting humans.

        Returns:
            str: A message indicating the detection result:
                - If humans are found, returns the number of detected individuals.
                - If no humans are found, encourages the user to select more images.
        """

        image = self.panorama.copy()
        h_count, image = self.common_func.customize_YOLO(image, conf)

        self.human_detect_image = image
        
        if h_count:
            if h_count == 1:
                return "👥 Great news! I found 1 person in your picture!🎉"
            else:
                return f"👥 Great news! I found {h_count} people in your picture! 🎉 with confidence {conf}"
        return f"Oops! It looks like we couldn’t find anyone in your picture. with confidence {conf} \nBut don’t worry—feel free to go back and choose a few more fabulous photos!🌟"


        

        
         

       
