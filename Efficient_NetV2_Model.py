# EfficientNetV2-B0 Code

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, saving, Input
#from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetV2B0
#from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

X = []
y = []
img_size =[480,366]
grade_list = [1,2,3,4,5,6,7,8,9,10]
minus_label=1

for u in grade_list:
    text_files_front = glob.glob("Data/Grade " + str(u) + "/front/*.jpg")
    text_files_back = glob.glob("Data/Grade " + str(u) + "/back/*.jpg")

    print(f"Grade {u}: {len(text_files_front)} front, {len(text_files_back)} back")

    for z in range(len(text_files_front)):
        img_back = cv2.imread(text_files_back[z])
        img_front = cv2.imread(text_files_front[z])

        img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
        img_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB)

        img_width = img_size[0] // 2
        img_height = img_size[1]

        img_back = cv2.resize(img_back, (img_width, img_height))
        img_front = cv2.resize(img_front, (img_width, img_height))

        combined = cv2.hconcat([img_front, img_back])

        # combined = preprocess_input(combined)

        X.append(combined)
        y.append(u - minus_label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = (train_test_split(
    X,y,test_size=0.2,random_state=42, stratify=y))

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

def augment_image(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)

    # Add small random rotation via tf.keras.layers
    img = tf.keras.layers.RandomRotation(0.05)(
        tf.expand_dims(img, 0), training=True
    )[0]

    # Random zoom
    img = tf.keras.layers.RandomZoom(0.05)(
        tf.expand_dims(img, 0), training=True
    )[0]

    img = tf.clip_by_value(img, 0, 255)
    return img.numpy()

augmented_X = []
augmented_y = []

for i in range(len(X_tr)):
    # Always keep the original
    augmented_X.append(X_tr[i])
    augmented_y.append(y_tr[i])

    # Add 4 augmented copies
    for _ in range(4):
        augmented_X.append(augment_image(X_tr[i]))
        augmented_y.append(y_tr[i])

X_tr = np.array(augmented_X)
y_tr = np.array(augmented_y)

print(f"Training samples after augmentation: {len(X_tr)}")

base_model = EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(366,480,3),
    pooling=None,
    include_preprocessing=True,
    name="efficientnetv2-b0",
)

base_model.trainable = False

inputs = Input(shape=(366, 480, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',#loss for multiple classification
                # a way to measure how wrong a classifier is when predicting one correct class out of many. - 0-9
                metrics=['accuracy'])

hist = model.fit(
    X_tr, y_tr,
    batch_size=16,
    epochs=20,
    validation_data=(X_val, y_val)
)

# After hist1 completes, unfreeze top layers
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False   # keep early layers frozen

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

hist2 = model.fit(
    X_tr, y_tr,
    batch_size=16,
    epochs=20,
    validation_data=(X_val, y_val)
)

plt.plot(hist.history["accuracy"] + hist2.history["accuracy"],     label="accuracy")
plt.plot(hist.history["val_accuracy"] + hist2.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {test_loss}  Test accuracy: {test_acc}")

# Predictions — argmax instead of > 0.5
y_pred = np.argmax(model.predict(X_test), axis=1)

# Confusion matrix — same as your code
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix,
    display_labels=[f"{i}" for i in range(1,len(conf_matrix)+1)]
)
disp.plot()
plt.show()

model.save("PokemonGrader_EfficientNetV2B0_V2-Model.keras")