import numpy as np
from tensorflow import keras
import cv2,os,sys
from keras.models import load_model
import webbrowser

clase=["","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(27, activation='softmax'))
    return model

def load_trained_model(model_path):
    model=create_model()
    model.load_weights(model_path)
    return model

def predict(image_path,model):
    image=cv2.imread(image_path,0)
    image=image.reshape(1,28,28,1)
    predictions=model.predict(image)
    i=0
    for prediction in predictions[0,:]:
        if prediction==1:
            return clase[i]
            break
        else:
            i+=1


def main():
    correctas=5

    model=load_trained_model("../Weights/my_model.h5")
    path=sys.argv[1]
    images=os.listdir(path)
    for image in images:
        if not (image=="desktop.ini"):
            prediccion=predict(path+"/"+image,model)
            letra=path.split("/")
            if(letra[len(letra)-1]==prediccion):
                correctas+=1
    path=os.getcwd()+"/../Html_files/"
    if(correctas==1):
        webbrowser.open_new_tab(path+'ejemplo5.html')
    elif(correctas==2):
        webbrowser.open_new_tab(path+'ejemplo4.html')
    elif(correctas==3):
        webbrowser.open_new_tab(path+'ejemplo3.html')
    elif(correctas==4):
        webbrowser.open_new_tab(path+'/ejemplo2.html')
    elif(correctas==5):
        print(path+'ejemplo1.html')
        webbrowser.open_new_tab(path+'ejemplo1.html')

if __name__ == '__main__':
    main()
