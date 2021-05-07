from collections import deque
import numpy as np
import cv2


#important paths and some constants
class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt"
                            ).read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'#pre-trained model
#         self.VIDEO_PATH = None
        self.VIDEO_PATH = "test/example1.mp4"
        # SAMPLE_DURATION is maximum deque size
        #self.cap = cv2.VideoCapture(0)
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112


# creating object of Class Parameter
param = Parameters()

# A Double ended queue to store our frames captured and with time
# old frames will pop out of the deque
clickit = deque(maxlen=param.SAMPLE_DURATION)

# load the human activity recognition model
print("loading human activity recognition model...")
net = cv2.dnn.readNet(model=param.ACTION_RESNET)

print("accessing input video ...")
# Take video file as input if given else turn on web-cam
# So, the input should be mp4 file or live web-cam video
vc = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

while True:
    # Loop over and read capture from the given video input
    (accessing, frame) = vc.read()

    # break when no frame is grabbed (or end if the video)
    if not accessing:
        print("no capture read from input video - exit")
        break

    # resize frame and append it to our deque
    frame = cv2.resize(frame, dsize=(600, 430))
    clickit.append(frame)

    # Process further only when the deque is filled
    if len(clickit) < param.SAMPLE_DURATION:
        continue

    # now that our captures array is filled we can
    # construct our image blob
    # We will use SAMPLE_SIZE as height and width for
    # modifying the captured frame
    imageBlob = cv2.dnn.blobFromImages(clickit, 1.0,
                                       (param.SAMPLE_SIZE,
                                        param.SAMPLE_SIZE),
                                       (114.7748, 107.7354, 99.4750),
                                       swapRB=True, crop=True)

    # Manipulate the image blob to make it fit as as input
    # for the pre-trained OpenCV's
    # Human Action Recognition Model
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Forward pass through model to make prediction
    net.setInput(imageBlob)
    outputs = net.forward()
    # Index the maximum probability
    label = param.CLASSES[np.argmax(outputs)]

    # Show the predicted activity
    cv2.rectangle(frame, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_ITALIC == 26,
                0.8, (0, 0, 0), 2)

    # Display it on the screen
    cv2.imshow("Momitoring Human Activity Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    # Press key 'q' to break the loop
    if key == ord("a"):
        break


