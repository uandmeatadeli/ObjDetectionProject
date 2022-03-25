import cv2
from ObjectDetection import ObjectDetection

class DistanceDetection:
    

    def __init__(self):
        pass

    def focal_length_finder(self, measured_distance, real_width, width_in_rf):
        focal_length = (width_in_rf * measured_distance) / real_width
        return focal_length
      
    def distance_to_camera(self, focalLength, knownWidth, perWidth):
        distance =(knownWidth * focalLength) / perWidth
        return distance

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
        #cap = cv2.VideoCapture('ObjDetectionProject/images/TestVideo.mp4')
        cap.set(3, 1920)
        cap.set(4, 1080)

        while True:
            ret, frame = cap.read()
            
            data = newObj.detectObj(frame)
            #print(data)

            for d in data:
                if d[0] == 'chair':
                    distance = self.distance_to_camera(focal_chair, CHAIR_WIDTH, d[1])
                    x, y = d[2]
                    cv2.rectangle(frame, (x, y-25), (x+300, y),(0,0,0), -1 )
                    cv2.putText(frame, f'dist: {round(distance,2)}inches', (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2 )
                elif d[0] == 'diningtable':
                    distance = self.distance_to_camera(focal_table, TABLE_WIDTH, d[1])
                    x, y = d[2]
                    cv2.rectangle(frame, (x, y-25), (x+300, y),(0,0,0), -1 )
                    cv2.putText(frame, f'dist: {round(distance,2)}inches', (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2 )

                if distance < 100:
                    cv2.rectangle(frame, (x, y-50), (x+260, y-25),(0,0,0), -1 )
                    cv2.putText(frame, 'Object ahead!!!',(x,y-25), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2 )
    
            cv2.imshow("Image", frame)
            key = cv2.waitKey(34)
            if key==27:
                break
        cap.release()
        cv2.destroyAllWindows()




#57 chair 
#61 dining table 