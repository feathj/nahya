import os
import cv2
import numpy as np


class Recognizer:

    def __init__(self, img_dir, face_casc_path, eye_casc_path):
        self.face_classifier = cv2.CascadeClassifier(face_casc_path)
        self.eye_classifier = cv2.CascadeClassifier(eye_casc_path)

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

        self.model = cv2.createEigenFaceRecognizer(
                num_components=80,
                threshold=3500
        )
        # self.model = cv2.createLBPHFaceRecognizer(threshold=50)
        # self.model = cv2.createFisherFaceRecognizer(num_components=2)
        self.model.train(np.asarray(images), np.asarray(hashed_labels))

    def __eye_angle(self, eyes):
        e1, e2 = eyes[0], eyes[1]
        if e1[0] >= e2[0]:
            e2, e1 = eyes[0], eyes[1]

        e1x, e1y, e1w, e1h = e1[0], e1[1], e1[2], e1[3]
        e2x, e2y, e2w, e2h = e2[0], e2[1], e2[2], e2[3]

        e1c = (e1x + (0.5 * e1w), e1y + (0.5 * e1h))
        e2c = (e2x + (0.5 * e2w), e2y + (0.5 * e2h))

        dx = e2c[0] - e1c[0]
        dy = e1c[1] - e2c[1]

        res = np.degrees(np.arctan2(dy, dx))

        return ((e1x, e1y, e1w, e1h), (e2x, e2y, e2w, e2h), res)

    def __detect_eyes(self, img):
        eyes = self.eye_classifier.detectMultiScale(img)
        if len(eyes) == 2:
            return self.__eye_angle(eyes)
        return None

    def __process_image(self, img):
        img_copy = cv2.resize(img, (200, 200))

        eyes = self.__detect_eyes(img_copy)
        if eyes:
            rot_mat = cv2.getRotationMatrix2D(
                    (100, 100),
                    (eyes[2] * -1.0),
                    1.0
            )
            img_copy = cv2.warpAffine(
                    img_copy,
                    rot_mat,
                    img_copy.shape,
                    flags=cv2.INTER_LINEAR
            )

        circle_mask = np.zeros((200, 200), np.uint8)
        cv2.ellipse(
                circle_mask,
                (100, 100),
                (70, 90),
                0,
                0,
                360,
                (255),
                -1,
                8
        )
        masked = np.full((200, 200), 255, np.uint8)
        np.copyto(
                masked,
                img_copy,
                where=circle_mask.astype(bool, casting='unsafe')
        )

        return masked

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
                    processed = self.__process_image(orig)

                    images.append(processed)
                    labels.append(base_dir)
                except Exception as exp:
                    print exp
                    continue
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
        eyes = self.__detect_eyes(cropped)
        processed = self.__process_image(cropped)
        cv2.imwrite('detect.jpg', processed)

        [label, confidence] = self.model.predict(processed)
        return self.id_to_label_hash.get(label, ""), confidence, eyes

    def detect_faces(self, img_buffer):
        image = self.__img_from_buffer(img_buffer, cv2.IMREAD_GRAYSCALE)
        faces = self.face_classifier.detectMultiScale(
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

            label, confidence, eyes = self.__recognize_face(image, f)
            if label:
                f['recognition'] = {}
                f['recognition']['label'] = label
                f['recognition']['confidence'] = '%.2f' % confidence
            if eyes:
                f['e1'] = [int(i) for i in eyes[0]]
                f['e2'] = [int(i) for i in eyes[1]]
            res.append(f)
        return res
