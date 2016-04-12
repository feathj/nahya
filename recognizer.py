import os
import cv2
import numpy as np


class Recognizer:

    def __init__(self, img_dir, casc_path):
        #images, labels = self.load_images(img_dir)
        #self.model = cv2.createEigenFaceRecognizer()
        #self.model.train(np.asarray(images), np.asarray(images))

        self.classifier = cv2.CascadeClassifier(casc_path)

    def load_images(self, img_dir):
        images = []
        labels = []
        for base_dir in os.walk(img_dir):
            for image_path in os.listdir(base_dir):
                try:
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    images.push(img)
                    labels.push(os.path.directory(base_dir))
                except:
                    next
        return images, labels

    def img_from_buffer(self, buff, cv2_img_flag=0):
        buff.seek(0)
        img_array = np.asarray(bytearray(buff.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    def recognize(self, img_buffer):
        image = self.img_from_buffer(img_buffer, cv2.IMREAD_GRAYSCALE)
        [label, confidence] = self.model.predict(np.assarray(image))
        return label, confidence

    def detect(self, img_buffer):
        image = self.img_from_buffer(img_buffer, cv2.IMREAD_GRAYSCALE)
        faces = self.classifier.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(20, 20),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        print faces

        res = []
        for (x, y, w, h) in faces:
            f = {}
            f['x'] = int(x)
            f['y'] = int(y)
            f['width'] = int(w)
            f['height'] = int(h)
            res.append(f)
        return res
