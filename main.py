# # Import the needed libraries
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# class Image_Processor:
#     def __init__(self) -> None:
#         pass
#     # validate images type
#     def img_valedation(images):
#         pass

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys
from ui_actions import UIActionManager

app = QApplication(sys.argv)

window = UIActionManager()

window.show()
sys.exit(app.exec_())

