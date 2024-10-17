import cv2
import numpy as np
from ultralytics import YOLO

class CommonFunctionality:
    """
    A class that provides common image processing functionalities using YOLO 
    for human detection and a Difference of Gaussian (DoG) method for edge detection.

    This class encapsulates methods to apply the YOLO model for detecting humans
    in images and to apply the Difference of Gaussian technique for edge detection.

    Attributes:
        model (YOLO): An instance of the YOLO model initialized with a specified 
                      weights file ('yolo11n.pt').

    Methods:
        customize_YOLO(frame, cnf=0.5):
            Applies the YOLO model to the given frame to detect humans and draw
            bounding boxes around them.
            
        apply_DoG(image):
            Applies the Difference of Gaussian method to the provided image to 
            detect edges.
    """
    def __init__(self):
        self.model = YOLO('yolo11n.pt')

    """
    YOLO Model Returns: - to understand human detection algo.
    Results(
        orig_img,
        path,
        names,
        boxes=None, <<<
        masks=None,
        probs=None,
        keypoints=None,
        obb=None,
        speed=None)

        Boxes(boxes, orig_shape)
        box ->
        data	   Tensor | ndarray	    The raw tensor containing detection boxes and associated data.
        orig_shape Tuple[int, int]	    The original image dimensions (height, width).
        is_track    bool	Indicates   whether tracking IDs are included in the box data.
        xyxy	    Tensor | ndarray	Boxes in [x1, y1, x2, y2] format.
        conf	    Tensor | ndarray	Confidence scores for each box.
        cls	        Tensor | ndarray	Class labels for each box.
        id	        Tensor | ndarray	Tracking IDs for each box (if available).
        xywh     	Tensor | ndarray	Boxes in [x, y, width, height] format.
        xyxyn	    Tensor | ndarray	Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn	    Tensor | ndarray	Normalized [x, y, width, height] boxes relative to orig_shape.
    """
    def customize_YOLO(self, frame, cnf=0.5):
        """
        Applies the YOLO model to the given frame to detect humans and 
        draw bounding boxes around them.

        Parameters:
            frame (ndarray): The image frame on which to apply human detection.
            cnf (float, optional): The confidence threshold for detecting humans.
                                   Default is 0.5. Only detections with confidence
                                   above this threshold will be considered.

        Returns:
            tuple: A tuple containing:
                - h_count (int): The count of detected humans.
                - frame (ndarray): The modified frame with bounding boxes drawn 
                                   around detected humans.
        """
        results = self.model(source=frame, show=False, conf=cnf, device='cpu')
        h_count = 0
        for result in results:
            # Loop detection boxes
            for box in result.boxes:
                # Get the box's coordinates and important properties.
                x1, y1, x2, y2 = box.xyxy[0] 
                conf = box.conf[0]    
                cls = int(box.cls[0])  
                # drow box if cls == 0 (human) and the confednce is grater than input value
                if cls == 0 and conf >= cnf:   
                    h_count += 1
                    # Drow box using cv2
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return  h_count, frame
    
    def apply_DoG(self, image):
        """
        Applies the Difference of Gaussian (DoG) method to the provided image 
        to detect edges.

        Parameters:
            image (ndarray): The input image on which to apply the DoG.

        Returns:
            ndarray: The resulting image after applying the Difference of 
                     Gaussian, highlighting the edges.
        """
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Convolve the gray resized image with two gaussians (one with a scale of 1 and another with a scale of 3)
        gaussian_1 = cv2.GaussianBlur(gray_img, (19,19), 3)
        gaussian_2 = cv2.GaussianBlur(gray_img, (31,31), 5)

        # find the difference of gaussians and save it
        DoG = gaussian_1 - gaussian_2
        return DoG 
            
