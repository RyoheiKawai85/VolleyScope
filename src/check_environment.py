import sys

import cv2
import pandas
from ultralytics import YOLO

print("使用中のPython:")
print(sys.executable)
print()
print("OpenCV:", cv2.__version__)
print("pandas:", pandas.__version__)
print("Ultralyticsを読み込めました")