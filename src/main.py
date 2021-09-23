import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),
                model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),
                model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),
                      len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),
                      len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

def generate_model():
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(filters=32, 
                     kernel_size=(3,3),
                     activation='relu',
                     input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    # Second convoulutional layer
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add((MaxPooling2D(pool_size=2, strides=2)))
    
    
    # Fully connected classifier
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.35))
    model.add(Dense(7, activation='softmax'))
    
    return model

# Ask for the choice of the mode
mode = input("Choose the mode: ")

if mode != "train" and mode != "display":
    print("No such mode")
    exit(1)
else:
    print("Mode selected,", mode)
    
model = generate_model()
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  metrics=['accuracy'])

if mode == "train":
    
    batch_size = 64
    num_epoch = 50

    # Define data generator
    train_dir = '../images/train_alpha'
    val_dir = '../images/test'
    
    train_datagen = ImageDataGenerator(rescale = 1./255)
    val_datagen = ImageDataGenerator(rescale=1./255.0)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')
    
    
    early_stopping = EarlyStopping(monitor='val_accuracy',
                              min_delta=0.001,
                              patience=2,
                              verbose=1,
                              restore_best_weights=True)


    callbacks_list = [early_stopping]

   
    model_info = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=validation_generator.n // batch_size,
            callbacks = callbacks_list)
    
    plot_model_history(model_info)
    model.save_weights('model.h5')
    
elif mode == "display":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
                    4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1)
                , 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame,
                        emotion_dict[maxindex],
                        (x+10, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2, 
                        cv2.LINE_AA)

        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
