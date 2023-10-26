from modules.face_recognition.face_recognition import FaceRecognition
import cv2
FR = FaceRecognition()
img = cv2.imread('./test_folder/Thien.jpg')
# print(img)
detected = FR.detect(img)
img = FR.draw_detect_result(img, detected)