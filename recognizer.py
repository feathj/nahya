import os
import cv2
import numpy as np


class Recognizer:

    def __init__(self, img_dir, casc_path):
        images, labels = self.__load_images(img_dir)
        self.model = cv2.createEigenFaceRecognizer()
        self.model.train(np.asarray(images), np.asarray(labels))

        self.classifier = cv2.CascadeClassifier(casc_path)

    def __load_images(self, img_dir):
        images = []
        labels = []
        for base_dir in os.listdir(img_dir):
            if not os.path.isdir(os.path.join(img_dir, base_dir)):
                next
            for image_path in os.listdir(os.path.join(img_dir, base_dir)):
                full_image_path = os.path.join(img_dir, base_dir, image_path)
                if os.path.isdir(full_image_path):
                    next
                try:
                    img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
                    images.append(img)
                    labels.append(1)
                except Exception as exp:
                    print exp
                    next
        return images, labels

    def __img_from_buffer(self, buff, cv2_img_flag=0):
        buff.seek(0)
        img_array = np.asarray(bytearray(buff.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    def __recognize_face(self, img, face):
        cx = face['x'] + (face['width'] / 2)
        cy = face['y'] + (face['height'] / 2)

        x1 = cx - 100
        x2 = cx + 100

        y1 = cy - 100
        y2 = cy + 100

        debug = [x1, x2, y1, y2]

        cropped = np.ascontiguousarray(img[y1:y2, x1:x2])
        #cv2.imwrite('/app/cropped.jpg', cropped)
        [label, confidence] = self.model.predict(cropped)
        return label, confidence, debug

    def detect_faces(self, img_buffer):
        image = self.__img_from_buffer(img_buffer, cv2.IMREAD_GRAYSCALE)
        faces = self.classifier.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(20, 20),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        res = []
        for (x, y, w, h) in faces:
            f = {}
            f['x'] = int(x)
            f['y'] = int(y)
            f['width'] = int(w)
            f['height'] = int(h)

            label, confidence, debug = self.__recognize_face(image, f)
            print label, confidence, debug
            f['label'] = label
            f['confidence'] = confidence
            f['debug'] = debug
            res.append(f)
        return res
