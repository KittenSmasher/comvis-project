# %% [markdown]
# Libraries

# %%
import cv2 as cv
import os
import numpy as np
import math

# %%
classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# %% [markdown]
# #### **FACE DETECTION**

# %%
train_path = './images/train/'
train_dir = os.listdir(train_path)

face_list = []
class_list = []

print(train_dir)

# %%
for idx, tdir in enumerate(train_dir):
    filenames = f"{train_path}/{tdir}"
    for filename in os.listdir(filenames):
        path = f"{filenames}/{filename}"
        img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # print(path)

        # use classifier (xml) to detect faces
        faces = classifier.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

        if(len(faces) < 1):
            continue
        else:
            for face_rect in faces:
                x, y, w, h = face_rect
                face_image = img_gray[y:y+h, x:x+w]
                face_list.append(face_image)
                class_list.append(idx)

# %%
print(class_list)
print(face_list)

# %% [markdown]
# #### **FACE RECOGNITION**

# %%
face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(face_list, np.array(class_list))

# %% [markdown]
# #### **TESTING**

# %%
test_path = './images/test/'

for filename in os.listdir(test_path):
    img_path = f"{test_path}/{filename}"

    test_img = cv.imread(img_path)
    img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    faces = classifier.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

    if(len(faces) < 1):
        continue
    else:
        for face_rect in faces:
            x, y, w, h = face_rect
            face_image = img_gray[y:y+h, x:x+w]

            res, conf = face_recognizer.predict(face_image)

            conf = math.floor(conf * 100) / 100 

            cv.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            image_text = f"{train_dir[res]} : {str(conf)}%"

            cv.putText(test_img, image_text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv.imshow("Result", test_img)
            cv.waitKey(0)
            cv.destroyAllWindows()




