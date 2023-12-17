# libraries
import cv2 as cv
import os
import numpy as np
import random
import time
import atexit

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
        
def clean():
    try:
        os.remove("trained_model.yml")
    except FileNotFoundError:
        pass

# DATASET
dataset_path = './Dataset'
dataset_dir = os.listdir(dataset_path)

# PRE-TRAINED MODEL
models = cv.CascadeClassifier('./pre-trained-models/haarcascade_frontalface_default.xml')

# FACE RECOGNIZER
recognizer = cv.face.LBPHFaceRecognizer_create()
model_trained = False

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
            clear()
            
        elif choice == 3:
            clean()
            break

def train_and_test():
    global model_trained
    global models
    global recognizer
    
  # split dataset
    train_list = []
    test_list = []
    images = []
    
    num_img = 0
    train_ratio = 0.75

    # convert label to integer
    label_map = {}

    for i, folder in enumerate(dataset_dir):
        label_map[folder] = i

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

    for face, label in train_list:
        image = cv.imread(face)

        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        img = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)  

        if(len(img) < 1):
            continue
        else:
            for rect in img:
                x, y, w, h = rect
                face_img = img_gray[y:y+h, x:x+w]
                faces.append(face_img)
                labels.append(label)
                
    recognizer.train(faces, np.array(labels))

    # correct = 0
    # total_predict = 0
    avg = []

    for face in test_list:
        image = cv.imread(face)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        img = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)    

        if(len(img) < 1):
            continue
        else:
            for rect in img:
                x, y, w, h = rect
                face_img = img_gray[y:y+h, x:x+w]
                _, conf = recognizer.predict(face_img)

                # face_path = os.path.basename(os.path.dirname(face))
                # actual_label = label_map[face_path]

                # total_predict += 1

                # if res == actual_label:
                #     correct +=1
                
                acc = (1-(conf/300))*100
                avg.append(acc)

    avg_accuracy = sum(avg)/len(avg)
    model_trained = True
    
    # SAVE MODEL
    recognizer.save('trained_model.yml')
    
    return avg_accuracy


def predict():
    global model_trained
    global models
    global recognizer
    
    if not model_trained:
        print("model not found.")
        input("press enter to continue...")
        main()
    else:
        print("fetching model...")
        recognizer.read('trained_model.yml')
        
        predict_path = input("Input absolute path for image to predict >> ")
        image = cv.imread(predict_path)

        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img = models.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)    

        for rect in img:
            x, y, w, h = rect
            face_img = img_gray[y:y+h, x:x+w]
            res, conf = recognizer.predict(face_img)
            
            conf = (1-(conf/300))*100
            
            cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                      
            image_text = f"{dataset_dir[res]} : {str(conf)}%"  
            # print(image_text)
            
            image_pos = (max(0, x-5), max(0, y-10))
            
            cv.putText(image, image_text, image_pos, cv.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 0), 2)
            
            cv.imshow("Result", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()
    atexit.register(clean)