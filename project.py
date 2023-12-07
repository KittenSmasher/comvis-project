# libraries
import cv2 as cv
import os
import numpy as np
import math
import random
import time

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# menu
def main():
    clear()
    while True:
        print("Football Player Face Recognition")
        print("1. Train and Test Model")
        print("2. Predict")
        print("3. Exit")
        choice = int(input(">> "))
        
        if choice == 1:
            print("Training and Testing")
            time.sleep(0.2)
            print("calculating accuracy, this may take some time...")
            avg = train_and_test()
            print(f"Average Accuracy = {avg:.2f}%")
            input("Press Enter to Continue...")
            clear()
        
        elif choice == 2:
            predict()
            
        elif choice == 3:
            break

def train_and_test():
    # pre-trained models
    models = cv.CascadeClassifier('./pre-trained-models/haarcascade_frontalface_default.xml')

    # datasets
    dataset_path = './Dataset'
    
  # split dataset
    train_list = []
    test_list = []

    images = []

    train_ratio = 0.75

    # convert label to integer
    label_map = {}

    for i, folder in enumerate(os.listdir(dataset_path)):
        label_map[folder] = i

    for folder in os.listdir(dataset_path):

        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        for img in os.listdir(folder_path):
            images.append(img)

        num_img = len(images)
        num_train = int(num_img * train_ratio)

        random.shuffle(images)

        label = label_map[folder]

        for img in images[:num_train]:
            img_path = os.path.join(folder_path, img)
            train_list.append((img_path, label))

        for img in images [num_train:]:
            img_path = os.path.join(folder_path, img)
            test_list.append(img_path)

    # FACE DETECTION
    faces = []
    labels = []
    width = 100

    for face, label in train_list:
        image = cv.imread(face)

        h, w = image.shape[:2]
        aspect_ratio = w/h
        height = int(width/aspect_ratio)

        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_gray = cv.resize(img_gray, (width, height))

        img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
        img = models.detectMultiScale(img_blur)  

        if(len(img) < 1):
            continue
        else:
            for rect in img:
                x, y, w, h = rect
                face_img = img_gray[y:y+h, x:x+w]
                faces.append(face_img)
                labels.append(label)

    # FACE RECOGNITION
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    correct = 0
    total_predict = 0

    for face in test_list:
        image = cv.imread(face)

        h, w = image.shape[:2]
        aspect_ratio = w/h
        height = int(width/aspect_ratio)

        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_gray = cv.resize(img_gray, (width, height))

        img_blur = cv.GaussianBlur(img_gray, (5,5), 0)
        img = models.detectMultiScale(img_blur)    

        if(len(img) < 1):
            continue
        else:
            for rect in img:
                x, y, w, h = rect
                face_img = img_gray[y:y+h, x:x+w]
                res, conf = recognizer.predict(face_img)

                face_path = os.path.basename(os.path.dirname(face))
                actual_label = label_map[face_path]

                total_predict += 1

                if res == actual_label:
                    correct +=1

    avg_accuracy = correct / total_predict * 100
    
    return avg_accuracy




def predict():
    print('meow')
    # if models != 0:
    #     predict_path = input("Input absolute path for image to predict >> ")
    #     predict_img = cv.imread(predict_path)
    #     img_gray = cv.imread(predict_img, cv.IMREAD_GRAYSCALE)
        
    #     face = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        
    #     for rect in face:
    #         x, y, w, h = rect
    #         face_img = img_gray[y:y+h, x:x+w]
    #         res, conf = recognizer.predict(face_img)
            
    #         conf = math.floor(conf*100)/100
            
    #         cv.rectangle(predict_img, (x, y), (x+w, y+h), (0, 255, 0), 2)           
    #         image_text = f"{train_folder[res]} : {str(conf)}%"  
            
    #         cv.putText(predict_img, image_text, (x, y-10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
    #         cv.imshow("Result", predict_img)
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    # else:
    #    print("model not found.")
    #    input("press enter to continue...")
    #    main()

if __name__ == '__main__':
    main()


# face detection