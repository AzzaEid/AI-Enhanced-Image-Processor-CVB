from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import sys
from Controller import Controller

app = QApplication(sys.argv)

window = Controller()

window.show()
sys.exit(app.exec_())

