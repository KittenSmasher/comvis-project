# libraries
import cv2 as cv
import os
import numpy as np
import math
import random
import shutil
import time

# menu
def main():
    while True:
        print("Football Player Face Recognition")
        print("1. Train and Test Model")
        print("2. Predict")
        print("3. Exit")
        choice = int(input(">> "))
        
        if choice == 1:
            train_and_test()
        
        elif choice == 2:
            predict()
            
        elif choice == 3:
            break

def train_and_test():
    # pre-trained models
    models = cv.CascadeClassifier('./pre-trained-models/haarcascade_frontalface_default.xml')

    # datasets
    dataset_path = './Dataset'

    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    train_ratio = 0.75

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        images = os.listdir(folder_path)
        
        # number of train images
        n = int(len(images) * train_ratio)
        
        # random shuffle
        random.shuffle(images)
        
        # SPLIT DATA
        train_img = images[:n]
        test_img = images[n:]
        
        # create separate folder in train folder
        train_folder = os.path.join(train_path, folder_name)
        os.makedirs(train_folder, exist_ok=True)
        
        # copy train images to train folder
        for img in train_img:
            source_path = os.path.join(folder_path, img)
            dest_path = os.path.join(train_folder, img)
            shutil.copy(source_path, dest_path)
            
        # copy to test images to test folder
        for img in test_img:
            source_path = os.path.join(folder_path, img)
            # rename test images just incase duplicate name
            dest_path = os.path.join(test_path, f'{folder_name}-{img}')
            shutil.copy(source_path, dest_path)
    
    # delete additional folders that will be made
    shutil.rmtree('./Dataset/train/train')
    shutil.rmtree('./Dataset/train/test')
    
    # FACE DETECTION
    
    train_path = './Dataset/train/'
    train_folder = os.listdir(train_path)
    
    faces = []
    labels = []
    
    for idx, folders in enumerate(train_folder):
        filenames = os.path.join(train_path, folders)
        
        for filename in os.listdir(filenames):
            path = os.path.join(filenames, filename)
            img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)

            face = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
            
            if(len(face) < 1):
                continue
            else:
                for rect in face:
                    x, y, w, h = rect
                    face_image = img_gray[y:y+h, x:x+w]
                    faces.append(face_image)
                    labels.append(idx)
    
    # FACE RECOGNITION
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    
    
    print("Training and Testing")
    time.sleep(1)
    print("Training and Testing Finished")
    time.sleep(0.5)
    print("Average Accuracy = {}")




def predict():
    print("this is predict")
    # try:
        
    # except NameError:
    #     print("model not found.")
    #     input("press enter to continue...")
    #     main()

if __name__ == '__main__':
    main()


# face detection