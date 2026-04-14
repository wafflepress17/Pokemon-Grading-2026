#plan b: train on just the back and just the front and then combine prediction for prediction



import cv2
import glob
import numpy as np


from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, losses, saving

from Camera import get_input

X = []
y = []
img_size = [240, 183]
grade_list = [1,2,3,4,5,6,7,8,9,10]
minus_label =1


for u in grade_list: #loads the images goes through dataset
    text_files_front = glob.glob("Data/Grade " + str(u) + "/front/*.jpg")
    text_files_back = glob.glob("Data/Grade " + str(u) + "/back/*.jpg")

    for z in range(len(text_files_front)):
        img_back = cv2.imread(text_files_back[z])
        img_front = cv2.imread(text_files_front[z])

        img_width = img_size[0] // 2
        img_height = img_size[1]

        img_back = cv2.resize(img_back, (img_width, img_height))
        img_front = cv2.resize(img_front, (img_width, img_height))

        combined = cv2.hconcat([img_front, img_back])

        combined = combined / 255.0

        X.append(combined)
        y.append(u - minus_label)

        # data augmentation
        brightness = 0.5 + np.random.rand()
        aug = np.clip(combined * brightness, 0, 1)
        X.append(aug)
        y.append(u - minus_label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(img_size[0], img_size[1], 3))) # 6 channels because two images are combined meaning there are now 6 channels
#augmentation
model.add(layers.RandomRotation(0.1))
model.add(layers.RandomZoom(0.1))
model.add(layers.RandomContrast(0.1))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))  # relu: whether to fire neuron or not
model.add(layers.MaxPooling2D((2, 2)))  # simplifies through 2x2
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # increase filters to look for more complex patterns
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(grade_list), activation='softmax'))  # softmax is for multiple classification

model.summary()

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',#loss for multiple classification
                # a way to measure how wrong a classifier is when predicting one correct class out of many. - 0-9
                metrics=['accuracy'])

hist = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size =16,
    epochs=5,
    verbose=2
)


plt.plot(hist.history['accuracy'], label='accuracy')  # Plot for accuracy and Epoch
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test,
                                        verbose=2)  # verbose: Controls logging output (0 = silent, 1 = progress bar, 2 = one line per epoch).

prediction = model.predict(X_test)
y_pred = np.argmax(model.predict(X_test), axis=1)

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
#
print(f"Test loss: {test_loss} Test accuracy: {test_acc}")
model.save("Pokemon_Grading_CNN-Model.keras")

######################################################
#
#
# while True:
#     get_input() #saves the picture from camera
#     new_model = saving.load_model("Pokemon_Grading_CNN-Model.keras")
#
#     def image_predict(new_model):
#
#         img_test_back = cv2.imread("backcaptured_image.png")
#         img_test_front = cv2.imread("frontcaptured_image.png")
#         img_width = img_size[0] // 2
#         img_height = img_size[1]
#
# ########## delete one half of picture ################
#         h, w = img_test_back.shape[:2]
#         half_w = w // 2
#
#         front_restored = img_test_front[:, :half_w]
#         back_restored = img_test_back[:,:half_w]
#
#         # (Optional) show them
#         cv2.imshow("Front Restored", front_restored)
#         cv2.imshow("Back Restored", back_restored)
#         cv2.waitKey(1)
# #######################################################
#         img_test_back = cv2.resize(img_test_back, (img_width, img_height))
#         img_test_front = cv2.resize(img_test_front, (img_width, img_height))
#
#         combined = cv2.hconcat([img_test_front, img_test_back])
#         cv2.imshow("Image", combined)
#         cv2.waitKey(1)
#
#
#         combined = combined / 255.0
#         combined = combined.reshape(1, img_size[1], img_size[0], 3)
#         pred = np.argmax(new_model.predict(combined), axis=1)
#         print("prediction:" +str (pred[0] + minus_label))
#
#     image_predict(new_model)


#######################################################################################################################3
# GridSearch
# Best score = 0.0000 using {'batch_size': 16, 'epochs': 5, 'model__dropout_rate': 0.0, 'model__optimizer': 'adam'}

import cv2
import glob
import numpy as np
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from scikeras.wrappers import KerasClassifier

X = []
y = []
img_size = [240, 183]
grade_list = [1,2,3,4,5,6,7,8,9,10]
minus_label =1
start=time()


def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))


for u in grade_list:
    text_files_front = glob.glob("Data/Grade " + str(u) + "/front/*.jpg")
    text_files_back = glob.glob("Data/Grade " + str(u) + "/back/*.jpg")

    for z in range(len(text_files_front)):
        img_back = cv2.imread(text_files_back[z])
        img_front = cv2.imread(text_files_front[z])

        img_width = img_size[0] // 2
        img_height = img_size[1]

        img_back = cv2.resize(img_back, (img_width, img_height))
        img_front = cv2.resize(img_front, (img_width, img_height))

        combined = cv2.hconcat([img_front, img_back])

        combined = combined / 255.0

        X.append(combined)
        y.append(u - minus_label)

X = np.array(X)
X = X.astype(np.float32)
y = np.array(y)

X_train, X_test, y_train, y_test = (train_test_split(X,y,test_size=0.2,random_state=42))

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def build_model(optimizer="adam", dropout_rate=0.0):
    model = models.Sequential()
    model.add(layers.Input(shape=(img_size[1], img_size[0], 3)))
    #augmentation
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))
    model.add(layers.RandomContrast(0.1))

    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))

    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(len(grade_list), activation='softmax'))

    if optimizer == "adam":
        opt = Adam()
    elif optimizer == "sgd":
        opt = SGD()
    elif optimizer == "rmsprop":
        opt = RMSprop()

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = KerasClassifier(
    model=build_model,
    verbose=1
)


param_grid = {
    "batch_size": [16, 32],
    "epochs": [5, 10],
    "model__optimizer": ["adam", "sgd", "rmsprop"],
    "model__dropout_rate": [0.0, 0.2]
}

n_cv = 3

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=n_cv,verbose =2)
grid_result = grid.fit(X, y)

# print out results
print('time for grid search = {:.0f} sec'.format(time()-start))
display_cv_results(grid_result)

