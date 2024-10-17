from Views.MainWindow import Ui_MainWindow
from Models.imageProcessor import ImageProcessor
from Views.dialogs import Dialog
from Models.imageType import ImageType
from Models.live_previews import live
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtCore import QtConcurrent
from concurrent.futures import ThreadPoolExecutor

import sys
import time


class Controller(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("AI-Enhanced  Image Stitching and Edge Detection")
        self.image_processor = ImageProcessor()
        self.valid_images = False  # Manage navigation to the next pages
        self.live_preview = live()

        """
        1- Why this MVC structure?
            I chose this Model-View-Controller (MVC) structure to effectively separate concerns within the application.
            The ImageProcessor class acts as the Model, handling all image processing tasks. 
            The UI, defined in Ui_MainWindow, serves as the View, presenting information to the user and capturing input. 
            The Controller manages the interaction between the Model and View.

            This design allows for a single set of images to be loaded once through the UI and used across all three pages. 
            By creating an ImageProcessor object within the Controller, I ensure that the panoramic image and any subsequent 
            processing results are easily accessible throughout the application. 
            >>> This simplifies the workflow and ensures efficient access to the stitched panorama without redundant processing or reloading.

        2- Purpose of valid_images:
        The boolean variable valid_images helps organize the app's flow. 
        If the panorama creation process fails, navigating to the next pages would be ineffective.
        This variable acts as a guard, ensuring that users can only proceed when valid images are available, 
        enhancing the overall user experience.
        
        """
        # region UI initialization 
        """
        Set slider default values ​​and appearance and disappearance status
        """
        # When running the application, we want show the sidebar that contains only icons
        self.text_sidebar.setHidden(True)
        self.stackedWidget.setCurrentIndex(0)
        self.warrning_befor_stitch.setHidden(True)
        self.comment_stitch.setHidden(True)
        self.kernel_slider.setValue(5)
        self.kernel_shape_combo.setCurrentIndex(0)
        self.open_close_combo.setCurrentIndex(1)
        self.conf_slider.setValue(50)
        # endregion

        # region switchPages Actions

        # Go to the appropriate page after click sidebar btns
        self.stitch_page_btn1.clicked.connect(self.switch_to_stitchPage)
        self.stitch_page_btn2.clicked.connect(self.switch_to_stitchPage)

        self.edge_page_btn1.clicked.connect(self.switch_to_edgePage)
        self.edge_page_btn2.clicked.connect(self.switch_to_edgePage)

        self.human_page_btn1.clicked.connect(self.switch_to_humanPage)
        self.human_page_btn2.clicked.connect(self.switch_to_humanPage)
        # endregion 
        
        # Upload images
        self.pick_imgs_btn.clicked.connect(self.pick_images)

        # stitch action
        self.stitch_btn.clicked.connect(self.handle_stitch)

        # notes buttons 
        self.canny_note.clicked.connect(self.canny_note_action)
        
        # canny slider
        self.threshold_slider.valueChanged.connect(lambda value: self.handle_canny(value))

        # DoG slider
        self.kernel_slider.valueChanged.connect(self.handle_DoG)
        self.kernel_shape_combo.currentIndexChanged.connect(self.handle_DoG)
        self.open_close_combo.currentIndexChanged.connect(self.handle_DoG)

        # YOLO slider
        self.conf_slider.valueChanged.connect(self.show_humen)

        # region live preview buttons actions
        self.live_canny.clicked.connect(self.run_canny_live)
        self.live_dog.clicked.connect(self.run_DoG_live)
        self.live_human_dtect.clicked.connect(self.run_YOLO_live)

        # endregion 

    # region live
    def live_buttons_state(self, state):
        """
        Enable or disable live processing buttons based on the provided state.

        Parameters:
            state (bool): The state to set for the live processing buttons. 
                          True to enable the buttons, False to disable them.
        """
        self.live_canny.setEnabled(state)
        self.live_dog.setEnabled(state)
        self.live_human_dtect.setEnabled(state)

    def run_canny_live(self):
        """
        Initiates live Canny edge detection.

        Disables the live processing buttons, applies the Canny edge detection,
        and then re-enables the buttons.
        """
        print("clicked")
        self.live_buttons_state(False)
        apply = lambda: self.live_preview.canny()
        apply()
        self.live_buttons_state(True)
    
    def run_DoG_live(self):
        print("clicked")
        self.live_buttons_state(False)
        apply = lambda: self.live_preview.DoG()
        apply()
        self.live_buttons_state(True)

    def run_YOLO_live(self):
        print("clicked")
        self.live_buttons_state(False)
        apply = lambda: self.live_preview.detect_humans()
        apply()
        self.live_buttons_state(True)
    # endregion
        

    # region switch Pages functions

    def switch_pages_warning(self):
        """
        Displays a warning dialog indicating that the user cannot proceed 
        to the next page without selecting images.
        """
        dialog = Dialog("Warning⚠️","👾 Oops! You can't proceed to the next page until you select images. 👾")
        dialog.exec_()  

    # Implement page switching functions
    def switch_to_stitchPage(self):
        """
        Switches the current view to the stitching page.
        """
        self.stackedWidget.setCurrentIndex(0)

    def switch_to_edgePage(self):
        """
        Switches the current view to the edge detection page.

        Checks if images have been uploaded before switching. 
        If not, displays a warning dialog.
        """
        if self.valid_images:
            self.stackedWidget.setCurrentIndex(1)
        else:
            self.switch_pages_warning()
        
    def switch_to_humanPage(self):
        """
        Switches the current view to the human detection page.

        Checks if images have been uploaded before switching. 
        If not, displays a warning dialog.
        """
        if self.valid_images:
            self.stackedWidget.setCurrentIndex(2)
        else:
            self.switch_pages_warning()
    
    # endregion 
    # region regarad to images
    def show_img(self, s_lable, image_type, height=300):
        """
        Displays a processed image in the specified label.

        Parameters:
            s_lable (QLabel): The label where the image will be displayed.
            image_type (ImageType): The type of image to retrieve from the processor.
            height (int, optional): The height to scale the image. Defaults to 300.
        """
        # Get processed image and show it in suitable way
        img = self.image_processor.getImage(image_type)
        scaled_pixmap = QPixmap.fromImage(img).scaled(self.stackedWidget.size().width(), height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        s_lable.setPixmap(scaled_pixmap)
        s_lable.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    
    def pick_images(self):
        # If there's images in image_proccesor we should clear it
        self.image_processor.images.clear()

        # Browse images 
        Inames,_ = QFileDialog.getOpenFileNames(self, 'Choose Images', 'C:/Users/DELL/Desktop/cv/Project/assets/test', 'Images (*.png  *.jpeg  *.jpg  *.xmp )') 
        """
        1. we want to delete the already existing labels in org_imgs_layout
        >> I already add 4 labels, but this is a more general syntax: reversed(range(self.org_imgs_layout.count()))
        2. Based on the number of images we need to determine the minimum width 
           that the image can reserve in the layout.
        3. We have to go through all the images that the user has selected:
         - add them to the imageProcessing object (image_processor)
         - Then create a QLabel, scale the image to reserved area and put the image in it.
        """
        if Inames:
            for i in reversed(range(self.org_imgs_layout.count())):
                widget = self.org_imgs_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater() 

            min_width = self.stackedWidget.size().width() // len(Inames) if  len(Inames) > 0 else self.stackedWidget.size().width()
            for Iname in Inames:
                # load image into image_processor 
                self.image_processor.loadImage(Iname)
                # add new lable for each selected image then add images
                label = QLabel()
                pixmap = QPixmap(Iname)
                resized_pixmap = pixmap.scaled(min_width, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(resized_pixmap)
                label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter) 
                # Add QLabel to the layout
                self.org_imgs_layout.addWidget(label)
    # endregion
    def handle_stitch(self):
        """
        Handles the stitching of images selected by the user. 
        Validates the input images, performs stitching, and updates the UI accordingly.

        Displays appropriate messages for the user based on the success or failure of the stitching process.
        If successful, it allows navigation to the next pages and 
        * initiates additional image processing tasks 
        (Canny edge detection and Difference of Gaussian) in the background.
        * This feature need to implement 

        Raises:
            Exception: If image stitching fails or no images are selected.
        """
        # Hide instruction labels & previos results
        self.warrning_befor_stitch.setHidden(True)
        self.comment_stitch.setHidden(True)
        self.panorama_show.setText("View Panorama")
        self.valid_images = False 

        # Check if there's images to stitch
        if len(self.image_processor.images) == 0 :
            self.warrning_befor_stitch.setText("You have to PICK images first🙄")
            self.warrning_befor_stitch.setHidden(False)
            return
        s = time.time()
        # Try to create panorama image
        state, string_to_show = self.image_processor.stitching()
        """ state: 1 = success, 0 = failed
            - in both cases we return appropriate text to display to the user (string_to_show); to provide sufficient information.
            - Only if the operation is successful 
                - will we display the panorama image
                - We will also change the value of the boolean that allows us to navigate to the next pages
                - We will also run the last operations in the background and put the results in the appropriate place
        """
        # success or failed
        self.comment_stitch.setText(string_to_show)
        self.comment_stitch.setHidden(False)

        # success
        if state:
            # Get image in vail form and show it
            self.show_img(self.panorama_show,ImageType.PANORAMA, 300)

            # change boolean value to swich to next pages
            self.valid_images = True 

            e = time.time()
            print(f"stitching run time = {e-s}")

            # Run other image processes in the background (see the note in func. docstring)
            self.run_background_tasks()
            
            
    def run_background_tasks(self):
        """
        >> IMPORTANT NOTE
        Represents a basic structure for executing background tasks for image processing
        after successful stitching. Currently, this function does not function as intended
        because the UI is updated only once after all operations are completed.

        The intention is to implement this functionality using QThreads in the future
        to allow for real-time updates to the UI while processing tasks in the background.
        
        Tasks include applying Canny edge detection and Difference of Gaussian (DoG) 
        to the stitched panorama image and displaying results.
        Each task's execution time is printed to the console for performance monitoring.
        """
        s = time.time()
        self.handle_canny()
        e = time.time()
        print(f"canny run time = {e-s}")

        s = time.time()
        self.image_processor.apply_DoG()
        self.handle_DoG()
        e = time.time()
        print(f"DoG run time = {e-s}")

        self.show_both()
        
        s = time.time()
        self.show_humen()
        e = time.time()
        print(f"human detection run time = {e-s}")
    

    # region EdgeDetection
    def handle_canny(self, threshold = None):
        """
        Applies Canny edge detection to the panorama image.

        Parameters:
            threshold (float, optional): The lower threshold for Canny edge detection. 
                                        If None, it is calculated based on the image's median.
        
        This method displays the threshold values and the resulting edge-detected image in the UI.
        NOTE that the threshold is not directly taken from slider value changes; instead, a lambda
        function is used to pass threshold just after slider is adjusted.
        This allow to find the threshold initial value based on the image's median 
        Displays the threshold values and the resulting edge-detected image in the UI.
        """
        # Update the Canny processing with the specified thresholds
        lower,threshold_values = self.image_processor.apply_canny(threshold)

        # show threshold values
        self.canny_threshould.setText(threshold_values) 

        # Display the resulting image after applying Canny in the specified result area
        self.show_img(self.canny_result,ImageType.TEMP_C,500)

    def canny_note_action(self):
        """
        Displays a dialog with information about how the Canny edge detection thresholds are set. 
        Provides guidance on user interaction with the slider for selecting the lower threshold.
        """
        dialog = Dialog("🌟 NOTE 🌟","We’ve chosen a scale from 0 to 85 for setting your thresholds.\n💡 Here’s how it works: \nyou’ll select the lower threshold, and we’ll automatically determine the upper threshold for you! \nThis approach is designed to give you the best results, based on OpenCV documentation.\nHappy edge detecting! 🖼️✨")
        dialog.exec_()  

    def handle_DoG(self):
        """""
        Applies the Difference of Gaussian (DoG) operation on the input image.

        Retrieves the kernel size and shape from user input and applies the DoG algorithm,
        then displays the resulting image in the UI.

        Ensures that the image processing results are updated based on user-selected parameters.
        """
        # This dictionary maps user-friendly names of morphological kernel shapes to their corresponding
        # OpenCV constants
        shapes = {"Rectangular Kernel" : "MORPH_RECT",
                  "Elliptical Kernel": "MORPH_ELLIPSE",
                  "Cross-shaped Kernel": "MORPH_CROSS"}
        
        # Get the current value of the kernel size from the slider
        kernel_size = self.kernel_slider.value()

        # Retrieve the selected shape of the kernel from the combo box
        combo_text = self.kernel_shape_combo.currentText()

        # Map the selected shape to the corresponding OpenCV morphological operation type
        shape = shapes[combo_text]

        # Get the selected morphological operation type
        operation_type = self.open_close_combo.currentText()

        # Update the Difference of Gaussian (DoG) processing with the specified kernel size, shape, and operation type
        self.image_processor.update_DoG(kernel_size, shape, operation_type)
        
        # Display the resulting image after applying DoG in the specified result area
        self.show_img(self.dog_result,ImageType.TEMP_D, 500)

    def show_both(self):
        """
        Displays the panorama image, Canny edge detection results, and DoG results 
        in the UI to allow for easy comparison of the images.
        """
        self.show_img(self.panorama_result,ImageType.PANORAMA, 220)
        self.show_img(self.canny_result_2,ImageType.CANNY_RESULT, 220)
        self.show_img(self.dog_result_2,ImageType.TEMP_D, 220)
    # endregion
    
    def show_humen(self):
        """
        Detects humans in the panorama image based on user-defined confidence thresholds.

        Updates the UI with the detection results and displays a loading message while processing.
        """
        self.human_comment.setText("...")

        # Show loading message during detection
        self.human_detec_result.setText("Loading... ⌛")

        # Get confidence value from slider (scaled to [0, 1])
        conf_value = self.conf_slider.value() / 100

        # Perform human detection and get the result comment
        comment = self.image_processor.detect_humans(conf_value)
        self.show_img(self.human_detec_result,ImageType.HUMAN_DETECTION, 500)

        # Update UI with the detection result comment
        self.human_comment.setText(comment)








    










        

                
    """
    need:
    1- upload images > chick if they valid > create image_processer object with images, change valid_images value
     - resize images and implement 3 processes
    2- implement condition before switch to 2nd & 3rd screen to prevent accece it without images 
    """"""
    Displays a warning dialog indicating that the user cannot proceed 
    to the next page without selecting images.
    """

        
         




