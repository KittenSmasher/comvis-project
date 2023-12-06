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
    
    # TESTING
    test_path = './Dataset/test/'
    
    accuracy_list = []
    
    for filename in os.listdir(test_path):
        img_path = f"{test_path}/{filename}"
        test_img = cv.imread(img_path)
        img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        face = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
        if(len(face) < 1):
            continue
        else:
            for rect in face:
                x, y, w, h = rect
                face_image = img_gray[y:y+h, x:x+w]

                res, conf = recognizer.predict(face_image)
                
                accuracy_list.append(conf)
        
    
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    
    print("Training and Testing")
    time.sleep(1)
    print("Training and Testing Finished")
    time.sleep(0.5)
    print(f"Average Accuracy = {float(average_accuracy)}")
    
    input("press enter to continue...")
    main()


def predict():
    if models != 0:
        predict_path = input("Input absolute path for image to predict >> ")
        predict_img = cv.imread(predict_path)
        img_gray = cv.imread(predict_img, cv.IMREAD_GRAYSCALE)
        
        face = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
        for rect in face:
            x, y, w, h = rect
            face_img = img_gray[y:y+h, x:x+w]
            res, conf = recognizer.predict(face_img)
            
            conf = math.floor(conf*100)/100
            
            cv.rectangle(predict_img, (x, y), (x+w, y+h), (0, 255, 0), 2)           
            image_text = f"{train_folder[res]} : {str(conf)}%"  
            
            cv.putText(predict_img, image_text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv.imshow("Result", predict_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
       print("model not found.")
       input("press enter to continue...")
       main()

if __name__ == '__main__':
    main()


# face detection