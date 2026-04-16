#Normal CNN model made ourselves

import cv2
import glob
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, losses, saving, Input

from Camera import get_input
X = []
y = []
img_size =[480,366]
grade_list = [1,2,3,4,5,6,7,8,9,10]
minus_label =1


for u in grade_list: #goes through all the labels
    text_files_front = glob.glob("Data/Grade " + str(u) + "/front/*.jpg")
    text_files_back = glob.glob("Data/Grade " + str(u) + "/back/*.jpg")

    for z in range(len(text_files_front)): #Goes through all the images
        img_back = cv2.imread(text_files_back[z])
        img_front = cv2.imread(text_files_front[z])

        img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB) #CV2 reads in BGR so we turn it back into RGB
        img_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB)

        img_width = img_size[0] // 2
        img_height = img_size[1]

        img_back = cv2.resize(img_back, (img_width, img_height))
        img_front = cv2.resize(img_front, (img_width, img_height))

        #combine the images horizontally
        combined = cv2.hconcat([img_front, img_back])

        combined = combined / 255.0
        X.append(combined)
        y.append((u - 1))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

inputs = Input(shape=(img_size[1], img_size[0], 3))
x = layers.Conv2D(16, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

outputs = layers.Dense(len(grade_list), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',#loss for multiple classification
                # a way to measure how wrong a classifier is when predicting one correct class out of many. - 0-9
                metrics=['accuracy'])

hist = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size =16,
    epochs=15,
    verbose=2
)

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test,
                                        verbose=2)  # verbose: Controls logging output (0 = silent, 1 = progress bar, 2 = one line per epoch).

#prediction
prediction = model.predict(X_test)
y_pred = np.argmax(model.predict(X_test), axis=1)

#confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=[f" {i}" for i in range(1, 11)]
)
disp.plot()
plt.show()

print(f"Test loss: {test_loss} Test accuracy: {test_acc}")
model.save("Pokemon_Grading_CNN-Model.keras")


