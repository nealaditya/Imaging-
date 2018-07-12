import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

cap = cv2.VideoCapture('perfusion.mov')
refPt = []
intensity_arr = []


def get_intensity(image):
    sum = 0
    for x in image:
        sum += x
    print(sum)
    return sum


def get_total_possible(img):
    sum = 0
    for x in img:
        sum += 255
    print(sum)
    return sum


f

def convert_gray(path):
    img = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def create_shadow(path):
    gray = cv2.imread(path, 0)
    return gray


def define_roi(cap):

    def create_vert(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
            refPt.append((x, y))
        if event == cv2.EVENT_LBUTTONUP:
            if len(refPt) > 1:
                cv2.line(frame, refPt[len(refPt) - 2], refPt[len(refPt) - 1], (0, 255, 0), 1)

    cap.set(1, 1)
    ret, frame = cap.read()
    cv2.imshow('image', frame)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', create_vert)

    while 1:
        cv2.imshow('image', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vertices = np.asarray(refPt)
    mask = np.zeros((frame.shape[0], frame.shape[1]))
    cv2.fillConvexPoly(mask, vertices, 1)
    mask = mask.astype(np.bool)
    cap.release
    cv2.destroyAllWindows()
    return mask


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

mask = define_roi(cap)

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame', frame)
        cv2.namedWindow('Frame')
        intensity = get_intensity(frame[mask])
        intensity_arr.append(intensity)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows(Vi)
print(intensity_arr)
