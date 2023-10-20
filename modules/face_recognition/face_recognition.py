import os
import onnxruntime as ort
import random
import numpy as np
import cv2
import time
import torch
import torchvision

import cv2
import numpy as np
# from iresnet import iresnet100
from modules.face_recognition.iresnet import iresnet100
from modules.face_recognition.utils import Utils

class FaceRecognition:
    def __init__(self,
                 detection_weights = './weights/detection_face.onnx',
                 conf_thres=0.7,
                 iou_thres=0.5,
                 img_size=(640,640),
                 classes_txt='./modules/face_recognition/classes.txt',
                 face_feat_extraction_weights='./weights/face_feat_extraction.onnx',
                 database_tensor='./database/face_database'):
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device()=='GPU' else ['CPUExecutionProvider']
        self.detection_weights=detection_weights
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.img_size=img_size
        self.names=classes_txt
        self.recog_session=None
        self.detect_session=None
        self.face_feat_extraction_weights=face_feat_extraction_weights
        self.database_tensor=database_tensor
        self.detect_session=ort.InferenceSession(self.detection_weights, providers=self.providers)
        self.recog_session=ort.InferenceSession(self.face_feat_extraction_weights, providers=self.providers)
        
    def detect(self, img_or):
        image, ratio, dwdh = Utils.letterbox(img_or, auto=False)
        image = image.transpose((2,0,1))
        

        result = None
        return result
        pass