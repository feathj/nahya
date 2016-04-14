import os
import cv2
import numpy as np


class Recognizer:

    def __init__(self, img_dir, casc_path):
        images, labels = self.__load_images(img_dir)

        self.id_to_label_hash = {}
        labels_processed = {}
        hash_id = 0
        hashed_labels = []
        for label in labels:
            if label not in labels_processed:
                hash_id += 1
                labels_processed[label] = hash_id
                self.id_to_label_hash[hash_id] = label
            hashed_labels.append(labels_processed[label])

        self.model = cv2.createEigenFaceRecognizer(threshold=7000)
        self.model.train(np.asarray(images), np.asarray(hashed_labels))

        self.classifier = cv2.CascadeClassifier(casc_path)

    def __load_images(self, img_dir):
        images = []
        labels = []
        for base_dir in os.listdir(img_dir):
            if not os.path.isdir(os.path.join(img_dir, base_dir)):
                continue
            for image_path in os.listdir(os.path.join(img_dir, base_dir)):
                full_image_path = os.path.join(img_dir, base_dir, image_path)
                if image_path.startswith('.'):
                    continue
                try:
                    orig = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
                    mirrored = cv2.flip(orig, 0)

                    images.append(orig)
                    labels.append(base_dir)
                    images.append(mirrored)
                    labels.append(base_dir)
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

        max_dim = face['width']
        if face['height'] >= max_dim:
            max_dim = face['height']

        x1 = cx - (max_dim * 0.5)
        x2 = cx + (max_dim * 0.5)

        y1 = cy - (max_dim * 0.5)
        y2 = cy + (max_dim * 0.5)

        cropped = np.ascontiguousarray(img[y1:y2, x1:x2])
        resized = cv2.resize(cropped, (200, 200))

        # cv2.imwrite('/app/resized.jpg', resized)
        [label, confidence] = self.model.predict(resized)
        print self.id_to_label_hash.get(label, ""), confidence
        return self.id_to_label_hash.get(label, ""), confidence

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

            label, confidence = self.__recognize_face(image, f)
            f['label'] = label
            f['confidence'] = confidence
            res.append(f)
        return res
