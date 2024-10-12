from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import QTimer, Qt

class WarningDialog(QDialog):
    def __init__(self, title, message):
        super().__init__()
        self.setWindowTitle(title)
        
        layout = QVBoxLayout()
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        ok_button = QPushButton("OK✅")
        ok_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        ok_button.clicked.connect(self.accept)  # Connect the button to close the dialog
        layout.addWidget(ok_button)
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)  # Center the button
        
        self.setLayout(layout)

        self.setStyleSheet("""
                           font-size : 22px;
                           padding : 10px;
                           
                           """)
        
        # Set up a timer to close the dialog after 5 seconds
        
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)  
        self.timer.timeout.connect(self.accept)  # To close the window
        self.timer.start(20000)  # 20 sec 

