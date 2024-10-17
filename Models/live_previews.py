       
import cv2
import numpy as np
from ultralytics import YOLO
from .commonFunctionality import CommonFunctionality
class live():
    def __init__(self):
        self.common_func = CommonFunctionality()
    
    def canny(self):
        """
        Starts a live preview of Canny edge detection using camera.

        This method captures video frames from the webcam, converts each frame
        to grayscale, and applies Canny edge detection. The original frame is 
        displayed alongside the detected edges in a combined view.

        The live preview continues until the user presses the 'q' key to quit.
        If the video device cannot be opened, an error message is printed.

        Returns:
            None
        """
        print("loading canny live preview.. ")
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Could not open video device")
        while True:
            ret, frame = video.read() # بمسك فريم فريم عشان أعالجه
            if not ret or frame is None:
                print("No frame captured")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # يفضل الرمادي في شغلنا

            edges = cv2.Canny(gray, 50, 150)

            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # رجعت من رمادي عشان بحتاج يكون عندي 3 تشانلز

            combined = np.hstack((frame, edges_3d)) # بحتاج 3 تشانلز عشان اعرف ادمج الفريم الاصلي (3 تشانلز) مع الايدج الي توصلتله
            cv2.imshow('Video, Edges', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        video.release()
        cv2.destroyAllWindows()

    def DoG(self):
        """
        Starts a live preview of the Difference of Gaussian (DoG) edge detection

        This method captures video frames from the webcam, applies the Difference of Gaussian
        algorithm to each frame, and then performs morphological opening on the resulting image.
        The original frame is displayed alongside the processed edges in a combined view.

        The live preview continues until the user presses the 'q' key to quit.
        If the video device cannot be opened, an error message is printed.

        Returns:
            None
        """
        morph_shape = cv2.MORPH_RECT
        size = 5
        mph_mask = cv2.getStructuringElement(morph_shape, (size, size))
        print("loading DoG live preview.. ")
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Could not open video device")
        while True:
            ret, frame = video.read() 
            if not ret or frame is None:
                print("No frame captured")
                break

            DoG = self.common_func.apply_DoG(frame)

            edges = cv2.morphologyEx(DoG, cv2.MORPH_OPEN, mph_mask)
            
            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) 
            combined = np.hstack((frame, edges_3d)) 
            cv2.imshow('Video, Edges', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        video.release()
        cv2.destroyAllWindows()

    def detect_humans(self):
        """
        Starts a live preview for human detection using the YOLO algorithm.

        This method captures video frames from the webcam and applies the YOLO algorithm to detect humans 
        in each frame. The processed frames are displayed in a window for real-time visualization.

        The live preview continues until the user presses the 'q' key to quit. 
        If a frame cannot be captured, an error message is printed.

        Returns:
            None
        """
        print("loading YOLO live preview.. ")
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read() 
            if not ret or frame is None:
                print("No frame captured")
                break
            
            _, frame=self.common_func.customize_YOLO(frame)
            cv2.imshow('Human Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # إغلاق الكاميرا
        video.release()
        cv2.destroyAllWindows()

