from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import QTimer, Qt

class Dialog(QDialog):
    """
    A custom dialog for displaying messages to the user.

    This class inherits from QDialog and creates a modal dialog that displays
    a title and a message, along with an 'OK' button to close the dialog. 

    The dialog automatically closes after 20 seconds if the user does not interact
    with it. The message is displayed in a centered QLabel, and the 'OK' button 
    is centered and configured to close the dialog when clicked.

    Parameters:
        title (str): The title of the dialog window.
        message (str): The message to be displayed in the dialog.

    Attributes:
        timer (QTimer): A timer that triggers the dialog to close automatically
                         after a specified duration.
    """
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
        layout.setAlignment(ok_button, Qt.AlignCenter)  # Center the button
        
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

