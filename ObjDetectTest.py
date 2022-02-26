from ObjectDetection import ObjectDetection 
from DistanceDetection import DistanceDetection

class ObjectDetectionTest:
    
    def main():
        newObj = ObjectDetection()
        newObj.detectObj()
        distanceObj = DistanceDetection()
        distanceObj.show_img()
        


    if __name__ == "__main__":
        main() 