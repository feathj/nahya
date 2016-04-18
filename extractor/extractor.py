import os
import cv2
import numpy as np
import uuid


base_path = os.path.dirname(os.path.abspath(__file__))
unprocessed_path = os.path.join(base_path, 'unprocessed')
processed_path = os.path.join(base_path, 'processed')

classifier = cv2.CascadeClassifier(
        os.path.join(base_path, 'haarcascade_face.xml')
)

for img_path in os.listdir(unprocessed_path):
    image = cv2.imread(
            os.path.join(unprocessed_path, img_path),
            cv2.IMREAD_GRAYSCALE
    )

    detected_faces = classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(20, 20),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in detected_faces:
        # calc bounding box
        cx = x + (w * 0.5)
        cy = y + (h * 0.5)
        max_dim = w
        if max_dim <= h:
            max_dim = h

        x1 = cx - (max_dim * 0.5)
        x2 = cx + (max_dim * 0.5)

        y1 = cy - (max_dim * 0.5)
        y2 = cy + (max_dim * 0.5)

        # crop and resize
        cropped = np.ascontiguousarray(image[y1:y2, x1:x2])
        resized = cv2.resize(cropped, (200, 200))

        new_filename = "{0}.jpg".format(uuid.uuid1())
        cv2.imwrite(os.path.join(processed_path, new_filename), resized)
        print "Wrote: {0}".format(new_filename)
