import sys
from PyQt5.QtWidgets import QApplication
from mainModel import Main_Frame

# execution program
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainFrame = Main_Frame()
    mainFrame.show()
    sys.exit(app.exec_())