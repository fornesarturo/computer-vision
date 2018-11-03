import numpy as np
import cv2
import random
import time


def bounce_ball_TOP_BOTTOM(velocityY):
    return -1 * velocityY

def bounce_ball_LEFT_RIGHT(velocityX):
    return -1 * velocityX

def ball(frame, binarized, center, vel, global_Measurements, only_one_square, rands, debug=False):

    global_Width = global_Measurements[0]
    global_Height = global_Measurements[1]

    ball_diameter = 20
    ball_radius = ball_diameter/2
    ball_color = (255, 125, 0)
    thickness = -1

    velX = vel[0]
    velY = vel[1]

    score = 0

    

    if(debug):
        binarized = cv2.resize(binarized, (global_Width, global_Height))
        cv2.circle(binarized, tuple(center), ball_diameter, ball_color, thickness)
    else:
        frame = cv2.resize(frame, (global_Width, global_Height))
        cv2.circle(frame, tuple(center), ball_diameter, ball_color, thickness)
    

    if (center[0] - 10) <= 0:
        velX = bounce_ball_TOP_BOTTOM(velX)
    elif (center[0] + 10) >= global_Width:
        velX = bounce_ball_TOP_BOTTOM(velX)
    if (center[1] - 10) <= 0:
        velY = bounce_ball_LEFT_RIGHT(velY)
    elif (center[1] + 10) >= global_Height:
        velY = bounce_ball_LEFT_RIGHT(velY)    

    if (center[1] < global_Height):
        print("Binarized at ", (center[0]-10), " ", center[1], " is: ", binarized[center[1]][center[0]-10])

    if binarized[center[1]][center[0]-10] == 255:
        velX = bounce_ball_TOP_BOTTOM(velX)
    if binarized[center[1]][center[0]+10] == 255:
        velX = bounce_ball_TOP_BOTTOM(velX)
    if binarized[center[1]+10][center[0]] == 255:
        velY = bounce_ball_LEFT_RIGHT(velY)
    if binarized[center[1]-10][center[0]] == 255:
        velY = bounce_ball_LEFT_RIGHT(velY)

    if only_one_square:
        randX = random.randint(1,global_Width-50)
        randY = random.randint(1,global_Height-50)        
        only_one_square = False
    else:
        randX = rands[0]
        randY = rands[1]

    cv2.rectangle(frame, (randX, randY), (randX+35, randY+35), (0,255,0), -1)

    ball_x1 = center[0]-ball_radius
    ball_x2 = center[0]+ball_radius
    ball_y1 = center[1]+ball_radius
    ball_y2 = center[1]-ball_radius

    if (ball_x1 <= (randX+60)) and (ball_x2 >= randX) and (ball_y1 <= (randY+60)) and (ball_y2 >= randY):
        score += 10
        print("New Score: ", score)
        only_one_square = True   

    center[0] += velX
    center[1] += velY

    rands = [randX, randY]

    vel = [velX, velY]

    if(debug):
        frame = binarized

    return frame, center, vel, only_one_square, rands

def getROI(frame, kernel, detector, debug=False):
    green_min = 70
    green_max = 90

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h, s, v = cv2.split(hsv)

    green_mask = cv2.inRange(h, green_min, green_max)

    green_mask = cv2.erode(green_mask, kernel, iterations=2)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    keypoints = detector.detect(green_mask)

    points = []
    for keypoint in keypoints:
        points.append([keypoint.pt[0], keypoint.pt[1]])

    if debug:
        im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts, False)
    
    if debug:
        cv2.drawContours(im_with_keypoints, [hull], 0, (0, 255, 0), 10, 8)
    
    board_mask = np.zeros(frame.shape, np.uint8)
    cv2.fillPoly(board_mask, [hull], (255, 255, 255))

    roi = frame & board_mask

    if debug:
        return roi, im_with_keypoints
    return roi, False

def binarize(frame, kernel):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_th, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    masked = binarized * gray
    negation = cv2.bitwise_not(binarized)
    dilation = cv2.dilate(negation, kernel, iterations=2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    return closing

def getBlobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByArea = True
    params.minArea = 200

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def main(cap, debug=False):
    global_Width = 1280
    global_Height = 720
    global_Measurements = [global_Width, global_Height]
    only_one_square = True
    vel = [30, 30]
    rands = [random.randint(1, global_Width-50), random.randint(1, global_Height-50)]

    center = [global_Width//2, global_Height//2]
    detector = getBlobDetector()

    kernel_blobs = np.ones((9, 9), np.uint8)
    kernel_binarization = np.ones((5, 5), np.uint8)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        roi, debug_im = getROI(frame, kernel_blobs, detector)
        binarized = binarize(roi, kernel_binarization)

        frame, center, vel, only_one_square, rands = ball(frame, binarized, center, vel, global_Measurements, only_one_square, rands, True)

        cv2.imshow('ROI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture("puntos.mp4")
    main(cap)
