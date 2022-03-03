import cv2
from cv2 import FONT_HERSHEY_COMPLEX
import imutils
import numpy as np
from imutils import paths
from sqlalchemy import BLANK_SCHEMA
from ObjectDetection import ObjectDetection

class DistanceDetection:
    

    def __init__(self):
        pass

    def focal_length_finder(self, measured_distance, real_width, width_in_rf):
        focal_length = (width_in_rf * measured_distance) / real_width
        return focal_length
      
    def distance_to_camera(self, focalLength, knownWidth, perWidth):
        return (knownWidth * focalLength) / perWidth

    def detect_Distance(self):   
        KNOWN_DISTANCE = 72
        CHAIR_WIDTH = 24
        TABLE_WIDTH = 26 
        newObj = ObjectDetection()

        ref_chair = cv2.imread('ObjDetectionProject/images/chair6ft.jpg')
        ref_table = cv2.imread('ObjDetectionProject/images/table6ft.jpg')


        chair_data = newObj.detectObj(ref_chair)
        print(chair_data)
        chair_width_in_rf = chair_data[0][1]

        table_data = newObj.detectObj(ref_table)
        print(table_data)
        table_width_in_rf = table_data[1][1]

        print(f"Chair in pixels: {chair_width_in_rf} table width in pixels: {table_width_in_rf} ")

        focal_chair = self.focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
        focal_table = self.focal_length_finder(KNOWN_DISTANCE, TABLE_WIDTH, table_width_in_rf)

        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()

            data = newObj.detectObj(frame)
            print(data)
            for d in data:
                if d[0] == 'chair':
                    distance = self.distance_to_camera(focal_chair, CHAIR_WIDTH, d[1])
                    x, y = d[2]
                elif d[0] == 'table':
                    distance = self.distance_to_camera(focal_table, TABLE_WIDTH, d[1])
                    x, y = d[2]
                cv2.putText(frame, f'dist: {round(distance,2)}inch', (x,y), FONT_HERSHEY_COMPLEX, 0.6, BLANK_SCHEMA, 2 )

            cv2.imshow("Image", frame)
            key = cv2.waitKey(34)
            if key==27:
                break
        cap.release()
        cv2.destroyAllWindows()




#57 chair 
#61 dining table 