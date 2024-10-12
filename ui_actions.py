from WindowUI import Ui_MainWindow
from imageProcessor import ImageProcessor
from warningDialog import WarningDialog
from imageType import ImageType
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

class UIActionManager(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("AI-Enhanced  Image Stitching and Edge Detection")
        self.image_processor = ImageProcessor()
        self.valid_images = False 

        """
        1- why this structure? >>>>>>>>>>
        I chose this structure because we are working with a single set of images, loaded once through the UI and used across all three windows -like a "has-a" relationship.
        By creating an ImageProcessor object within the UI class, I ensure that the panoramic image and any subsequent image processing results are easily accessible throughout the application.
        ==> This approach simplifies the workflow, and ensures efficient access to the stitched panorama in all windows without redundant processing or reloading.

        2- valid_images: this boolean variable helps me organize the app's flow, if the panorama creation process doesn't work, navigating to the next pages won't be of any value.
        
        """

        # When running the application, we want show the sidebar that contains only icons
        self.text_sidebar.setHidden(True)
        self.stackedWidget.setCurrentIndex(0)
        self.warrning_befor_stitch.setHidden(True)
        self.comment_stitch.setHidden(True)
        
        self.kernel_slider.setValue(5)
        self.kernel_shape_combo.setCurrentIndex(0)
        self.open_close_combo.setCurrentIndex(0)
       
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


    
    # region switch Pages functions

    def switch_pages_warning(self):
        dialog = WarningDialog("Warning⚠️","👾 Oops! You can't proceed to the next page until you select images. 👾")
        dialog.exec_()  

    # Implement page switching functions
    def switch_to_stitchPage(self):
        self.stackedWidget.setCurrentIndex(0)

    def switch_to_edgePage(self):
        # Check if images uploaded
        if self.valid_images:
            self.stackedWidget.setCurrentIndex(1)
        else:
            self.switch_pages_warning()
        
    def switch_to_humanPage(self):
        # Check if images uploaded
        if self.valid_images:
            self.stackedWidget.setCurrentIndex(2)
        else:
            self.switch_pages_warning()
    # endregion 

    def show_img(self, s_lable, image_type ):
        # Get processed image and show it in suitable way
        img = self.image_processor.getImage(image_type)
        scaled_pixmap = QPixmap.fromImage(img).scaled(self.stackedWidget.size().width(), 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        s_lable.setPixmap(scaled_pixmap)
        s_lable.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    
    def pick_images(self):

        # If there's images in image_proccesor we should clear it
        self.image_processor.images.clear()

        # Brows images >>>>
        Inames,_ = QFileDialog.getOpenFileNames(self, 'Choose Images', 'C:/Users/DELL/Desktop/cv/Project/assets/test', 'Images (*.png  *.jpeg  *.jpg  *.xmp )') 
        
        if Inames:
            # we want to delete the labels in org_imgs_layout
            ## I already add 4 labels, but this is a more general syntex
            # reversed(range(self.horizontalLayout_3.count()))
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
                resized_pixmap = pixmap.scaled(min_width, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(resized_pixmap)
                label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter) 

                # Add QLabel to the layout
                self.org_imgs_layout.addWidget(label)
    
    def handle_stitch(self):

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
            self.show_img(self.panorama_show,ImageType.PANORAMA)

            # change boolean value to swich to next pages
            self.valid_images = True 

            # Run other processes in background
            self.handle_canny()
            self.image_processor.apply_DoG()
            self.handle_DoG()
            self.show_both()

        
    
    def handle_canny(self, threshold = None):
        lower,threshold_values = self.image_processor.apply_canny(threshold)
        # show threshold values
        self.canny_threshould.setText(threshold_values) 
        self.show_img(self.canny_result,ImageType.TEMP_C)

    def canny_note_action(self):
        dialog = WarningDialog("🌟 NOTE 🌟","We’ve chosen a scale from 0 to 85 for setting your thresholds.\n💡 Here’s how it works: \nyou’ll select the lower threshold, and we’ll automatically determine the upper threshold for you! \nThis approach is designed to give you the best results, based on OpenCV documentation.\nHappy edge detecting! 🖼️✨")
        dialog.exec_()  

    def handle_DoG(self):
        shapes = {"Rectangular Kernel" : "MORPH_RECT",
                  "Elliptical Kernel": "MORPH_ELLIPSE",
                  "Cross-shaped Kernel": "MORPH_CROSS"}
        kernel_size = self.kernel_slider.value()
        combo_text = self.kernel_shape_combo.currentText()
        shape = shapes[combo_text]
        operation_type = self.open_close_combo.currentText()

        self.image_processor.update_DoG(kernel_size, shape, operation_type)
        self.show_img(self.dog_result,ImageType.TEMP_D)

    def show_both(self):
        self.show_img(self.panorama_result,ImageType.PANORAMA)
        self.show_img(self.canny_result_2,ImageType.CANNY_RESULT)
        self.show_img(self.dog_result_2,ImageType.TEMP_D)








    










        

                
    """
    need:
    1- upload images > chick if they valid > create image_processer object with images, change valid_images value
     - resize images and implement 3 processes
    2- implement condition before switch to 2nd & 3rd screen to prevent accece it without images 
    """

        
         




