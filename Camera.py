from cv2 import imwrite, imshow, waitKey, destroyWindow, VideoCapture
import keyboard



def camerapicture(name):
    # Initialize webcam (0 = default camera)
    cam = VideoCapture(0)

    # Capture one frame
    ret, frame = cam.read()

    if ret: #if image was taken
        imshow("Captured", frame)
        imwrite(str(name)+"captured_image.png", frame)
        print("picture " + str(name) + " is taken")
        waitKey(1)
        destroyWindow("Captured")
    else:
        print("Failed to capture image.")

    cam.release()



def get_input():
    f_pressed = False
    b_pressed = False
    while not f_pressed or not b_pressed:
        if keyboard.is_pressed('f'):
            camerapicture("front")
            f_pressed = True
        if keyboard.is_pressed('b'):
            camerapicture("back")
            b_pressed=True