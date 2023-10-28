from modules.face_recognition.face_recognition import FaceRecognition
from modules.face_recognition.utils import Utils
import cv2
FR = FaceRecognition()
img = cv2.imread('./test_folder/Thien.jpg')
# print(img)
detected = FR.detect(img)
features = FR.get_face_features(img, detected)
print(features[0].shape)


