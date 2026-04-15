# Import model and test it with camera, press f for front of card and b for back of card

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import saving
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from Camera import get_input

img_size = [480, 366]

efficient_model    = saving.load_model("PokemonGrader_EfficientNetV2B0_V2-Model.keras")
cnn_model = saving.load_model("Pokemon_Grading_CNN-Model.keras")


backbone      = efficient_model.get_layer("efficientnetv2-b0")
en_conv_layer    = backbone.get_layer("top_activation")

# Build within backbone's own graph — avoids the cross-graph Keras 3 error
gradcam_efficient_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[en_conv_layer.output, backbone.output]
)
cnn_conv_layer = cnn_model.get_layer("conv2d_2")
gradcam_cnn_model = tf.keras.Model(
    inputs=cnn_model.input,
    outputs=[cnn_conv_layer.output, cnn_model.output]
)

def make_gradcam_heatmap(img_array, gradcam_model, pred_index=None):
    img_t = tf.cast(img_array, tf.float32)

    # compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, full_preds = gradcam_model(img_t, training=False)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = int(tf.argmax(full_preds[0]))
        class_score = full_preds[:, pred_index]

    #This is the gradient of the output neuron (top predicted or chosen)
    #with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap      = tf.maximum(heatmap, 0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

#Overlays heatmap onto image
def overlay_gradcam(img_rgb, heatmap, alpha=0.4):
    h, w            = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)


def grade_card(model, gradcam_model,efficient):
    img_front = cv2.imread("frontcaptured_image.png")
    img_back  = cv2.imread("backcaptured_image.png")

    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
    img_back  = cv2.cvtColor(img_back,  cv2.COLOR_BGR2RGB)

    h, w = img_front.shape[:2]
    half_w = w // 2

    front_restored = img_front[:, :half_w]
    back_restored = img_back[:, :half_w]

    # (Optional) show them
    # cv2.imshow("Front Restored", front_restored)
    # cv2.imshow("Back Restored", back_restored)

    img_front = cv2.resize(front_restored, (img_size[0] // 2, img_size[1]))
    img_back  = cv2.resize(back_restored,  (img_size[0] // 2, img_size[1]))

    combined     = cv2.hconcat([img_front, img_back])
    #img_array = np.expand_dims(combined, axis=0)
    if efficient:
        #img_array = preprocess_input(combined.astype(np.float32))
        img_array = np.expand_dims(combined, axis=0)
    else:
        combined_pre = combined / 255.0
        img_array = np.expand_dims(combined_pre, axis=0)

    predictions = model.predict(img_array, verbose=0)
    grade = int(np.argmax(predictions[0])) + 1
    confidence = predictions[0][grade - 1] * 100

    #Second choice
    second_idx = int(np.argsort(predictions[0])[-2])

    print(f"Predicted Grade: {grade}/10")
    print(f"Confidence: {confidence:.1f}%")

    heatmap_pred = make_gradcam_heatmap(img_array, gradcam_model, pred_index=grade - 1)
    heatmap_second = make_gradcam_heatmap(img_array, gradcam_model, pred_index=second_idx)

    overlay_pred   = overlay_gradcam(combined, heatmap_pred)
    overlay_second = overlay_gradcam(combined, heatmap_second)

    #Plotting it
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Predicted Grade: {grade}/10  |  Confidence: {confidence:.1f}%",
        fontsize=13, fontweight="bold"
    )

    axes[0].imshow(combined)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_pred)
    axes[1].set_title(f"GradCAM — Grade {grade}")
    axes[1].axis("off")

    axes[2].imshow(overlay_second)
    axes[2].set_title(
        f"GradCAM — Grade {second_idx + 1}\n"
        f"(2nd: {predictions[0][second_idx]:.1%})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    return grade


while True:
    get_input()
    print("EfficientNetV2-B0 Model")
    grade_card(efficient_model, gradcam_efficient_model,True)
    print("Normal CNN Model")
    grade_card(cnn_model, gradcam_cnn_model, False)