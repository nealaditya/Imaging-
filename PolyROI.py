import cv2
import numpy as np

refPt = []
path2 = "/Users/Neal/Documents/OHSU 2018/Imaging Test/checkboard.png"
path = "face.png"


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


def create_vert(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 1, (0, 255, 0), 1)
        refPt.append((x, y))
    if event == cv2.EVENT_LBUTTONUP:
        if len(refPt) > 1:
            cv2.line(img, refPt[len(refPt) - 2], refPt[len(refPt) - 1], (0, 255, 0), 2)


def create_flat_index(mask):
    flatmask = mask.flatten()
    positions = []
    index = 0
    for x in flatmask:
        if x:
            positions.append(index)
        index += 1
    return positions


def get_flat_intensity(image, index):
    flat_image = image.flatten()
    roi = flat_image[index]
    total = 0
    for x in roi:
        total += x
    print(total)
    return total


def create_shadow(path):
    gray = cv2.imread(path, 0)
    return gray


def convert_gray(path):
    img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


img = convert_gray(path)
cv2.namedWindow('image')
cv2.setMouseCallback('image', create_vert)

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        cv2.line(img, refPt[0], refPt[len(refPt) - 1], (0, 255, 0), 2)
        break

gray = create_shadow(path)
vertices = np.asarray(refPt)
mask = np.zeros((gray.shape[0], gray.shape[1]))

cv2.fillConvexPoly(mask, vertices, 1)
mask = mask.astype(np.bool)
get_intensity(gray[mask])
get_total_possible(gray[mask])

cv2.waitKey(0)
cv2.destroyAllWindows()
