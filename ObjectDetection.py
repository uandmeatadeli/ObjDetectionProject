import cv2
import numpy as np

class ObjectDetection:
   
    def __init__(self):
        pass

    def detectObj(self):
        # Load Yolo 
        net =cv2.dnn.readNet("ObjDetectionProject/yolov3.weights", "ObjDetectionProject/yolov3.cfg")
        classes = []
        with open("ObjDetectionProject/coco.names", "r") as f:
            classes = f.read().splitlines() 

        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # Loading image 
        #img = cv2.imread("room_ser.jpg")
        cap = cv2.VideoCapture(1)

        while True:
            #cap = cap.set(cv2.CAP_PROP_FPS, 30)
            _, img = cap.read()
            height, width, channels = img.shape

            # Detecting Objects 
            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), True, crop=False)

            #for b in blob:
            #    for n, img_blob in enumerate(b):
            #       cv2.imshow(str(n),img_blob)

            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_names)
            # print(outs)

            # SHowing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection [3] * height)

                        #cv2.circle(img, (center_x, center_y), 10, (0,255,0),2)

                        # Rectangle Coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        #cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            #print(len(boxes))
            #number_objects_detected = len(boxes)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            #print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    #print(label)
                    cv2.rectangle(img,(x,y), (x+w, y+h), color,2)
                    cv2.putText(img, label + " " + str(round(confidences[i],2)), (x, y + 30), font, 3, color, 3)
            cv2.imshow("Image", img)
            key = cv2.waitKey(34)
            if key==27:
                break
        cap.release()
        cv2.destroyAllWindows()


