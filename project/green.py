import numpy as np
import cv2
import random
import time
import math


has_clicked = False
refPt = []

def bounce_ball_TOP_BOTTOM(velocityY):
    return -1 * velocityY

def bounce_ball_LEFT_RIGHT(velocityX):
    return -1 * velocityX

def click(event, x, y, flags, param):
    global refPt, has_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x, y]
        has_clicked = True

def ball(frame, binarized, center, vel, global_Measurements, only_one_square, rands, score, max_min, ratios, debug=False):
    global_Width = global_Measurements[0]
    global_Height = global_Measurements[1]

    # print("FRAME: ", frame.shape, " BINARIZED: ", binarized.shape)

    if max_min[0] == float('Inf') or max_min[0] == float('-Inf') or max_min[1] == float('Inf') or max_min[2] == float('-Inf'):
        print("Here validation")
        max_min[0] = 0
        max_min[1] = global_Width
        max_min[2] = 0
        max_min[3] = global_Height
    else:
        for i in range(len(max_min)):
            if i < 2:
                max_min[i] = max_min[i] * ratios[0]
            else:
                max_min[i] = max_min[i] * ratios[1]
            max_min[i] = int(math.floor(max_min[i]))

    ball_diameter = 20
    ball_radius = ball_diameter/2
    ball_color = (255, 125, 0)
    thickness = -1

    velX = vel[0]
    velY = vel[1]

    out_of_bounds = False

    if (center[0] - 10) <= 5 or (center[0] + 10) >= global_Width-5:
        velX = bounce_ball_TOP_BOTTOM(velX)
        out_of_bounds = True
    if (center[1] - 10) <= 5 or (center[1] + 10) >= global_Height-5:
        velY = bounce_ball_LEFT_RIGHT(velY)
        out_of_bounds = True

    bounce = False

    x_Circle = ball_radius * math.cos(45 * math.pi / 180)
    y_Circle = ball_radius * math.sin(45 * math.pi / 180)

    x_Circle = int(math.floor(x_Circle))
    y_Circle = int(math.floor(y_Circle))

    # if not out_of_bounds and (
    #         binarized[center[1]][center[0]-10] == 255 or 
    #         binarized[center[1]][center[0]+10] == 255 or
    #         binarized[center[1] + y_Circle][center[0] + x_Circle] == 255 or
    #         binarized[center[1] + y_Circle][center[0] - x_Circle] == 255):
    #     velX = bounce_ball_TOP_BOTTOM(velX)
    #     bounce = True
    # if not out_of_bounds and (
    #         binarized[center[1]+10][center[0]] == 255 or 
    #         binarized[center[1]-10][center[0]] == 255 or
    #         binarized[center[1] - y_Circle][center[0] + x_Circle] == 255 or
    #         binarized[center[1] - y_Circle][center[0] - x_Circle] == 255):
    #     velY = bounce_ball_LEFT_RIGHT(velY)
    #     bounce = True

    
    if not bounce and not out_of_bounds and (
            binarized[center[1]][center[0]-10] == 255 or 
            binarized[center[1]][center[0]+10] == 255):
            # binarized[center[1] + y_Circle][center[0] + x_Circle] == 255 or
            # binarized[center[1] + y_Circle][center[0] - x_Circle] == 255):
        velX = bounce_ball_TOP_BOTTOM(velX)
        bounce = True
    if not bounce and not out_of_bounds and (
            binarized[center[1]+10][center[0]] == 255 or 
            binarized[center[1]-10][center[0]] == 255):
            # binarized[center[1] - y_Circle][center[0] + x_Circle] == 255 or
            # binarized[center[1] - y_Circle][center[0] - x_Circle] == 255):
        velY = bounce_ball_LEFT_RIGHT(velY)
        bounce = True


    if only_one_square:
        try:
            randX = random.randint(max_min[0]+30, max_min[1]-30)
            randY = random.randint(max_min[2]+30, max_min[3]-30)
        except:
            randX = global_Width // 2
            randY = global_Height // 2
        only_one_square = False
    else:
        randX = rands[0]
        randY = rands[1]

    cv2.rectangle(frame, (randX, randY), (randX+35, randY+35), (0,255,0), -1)

    if debug:
        cv2.rectangle(binarized, (max_min[0], max_min[2]), (max_min[0]+10, max_min[2]+10), (0,0,255), -1)
        cv2.rectangle(binarized, (max_min[1], max_min[3]), (max_min[1]-10, max_min[3]-10), (0,0,255), -1)
    
    ball_x1 = center[0]-ball_radius
    ball_x2 = center[0]+ball_radius
    ball_y1 = center[1]+ball_radius
    ball_y2 = center[1]-ball_radius

    if (ball_x1 <= (randX+60)) and (ball_x2 >= randX) and (ball_y1 <= (randY+60)) and (ball_y2 >= randY):
        score += 10
        print("New Score: ", score)
        only_one_square = True   

    if debug:
        cv2.circle(binarized, tuple(center), ball_diameter, ball_color, thickness)
    else:
        cv2.circle(frame, tuple(center), ball_diameter, ball_color, thickness)
    

    center[0] += velX
    center[1] += velY

    rands = [randX, randY]

    vel = [velX, velY]

    if(debug):
        frame = binarized

    return frame, center, vel, only_one_square, rands, score

def resize(image, global_Measurements):
    print("WIDTH: ", global_Measurements[0] , " HEIGHT: ", global_Measurements[1])
    return cv2.resize(image, (global_Measurements[0], global_Measurements[1]))

def getROI(frame, kernel, detector, debug=False):
    green_min = 50
    green_max = 100

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    green_mask = cv2.inRange(h, green_min, green_max)

    green_mask = cv2.erode(green_mask, kernel, iterations=2)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)

    keypoints = detector.detect(green_mask)
    # print(keypoints)

    min_x = float('Inf')
    max_x = -float('Inf')

    min_y = float('Inf')
    max_y = -float('Inf')

    points = []
    for keypoint in keypoints:
        # print(keypoint, keypoint.pt[0], keypoint.pt[1])
        points.append([keypoint.pt[0], keypoint.pt[1]])
        if keypoint.pt[0] < min_x: 
            min_x = keypoint.pt[0]
        if keypoint.pt[0] > max_x:
            max_x = keypoint.pt[0]
        if keypoint.pt[1] < min_y: 
            min_y = keypoint.pt[1]
        if keypoint.pt[1] > max_y:
            max_y = keypoint.pt[1]

    min_max = [min_x, max_x, min_y, max_y]
    # print(points)

    if debug:
        im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pts = np.array(points, np.int32)
    # print("Points: ", pts)
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts, False)

    # print(hull)
    
    if debug:
        if hull is not None and len(hull) > 0:
            cv2.drawContours(im_with_keypoints, [hull], 0, (0, 255, 0), 10, 8)
    
    board_mask = np.zeros(frame.shape, np.uint8)
    if hull is not None and len(hull) > 0:
        cv2.fillPoly(board_mask, [hull], (255, 255, 255))

    roi = frame & board_mask

    if debug:
        return roi, im_with_keypoints, min_max
    return roi, False, min_max

def binarize(frame, kernel):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_th, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    negation = cv2.bitwise_not(binarized)
    dilation = cv2.dilate(negation, kernel, iterations=2)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    return closing

def getBlobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByCircularity = False
    params.minCircularity = 0.6

    params.filterByArea = False
    params.minArea = 200

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False

    detector = cv2.SimpleBlobDetector_create(params)
    
    return detector

def main(cap, debug=False):
    global_Width = 980
    global_Height = 600
    # global_Width = 1280
    # global_Height = 720
    global_Measurements = [global_Width, global_Height]
    only_one_square = True
    vel = [25, 16]
    score = 0
    rands = [random.randint(1, global_Width-50), random.randint(1, global_Height-50)]

    global refPt
    # center = [global_Width//2, global_Height//2]
    detector = getBlobDetector()

    kernel_blobs = np.ones((9, 9), np.uint8)
    kernel_binarization = np.ones((5, 5), np.uint8)

    cv2.namedWindow('ROI')

    max_min = []

    font = cv2.FONT_HERSHEY_SIMPLEX

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        rate_y = global_Height / frame.shape[0]
        rate_x = global_Width / frame.shape[1]

        ratios = [rate_x, rate_y]


        roi, debug_im, max_min = getROI(frame, kernel_blobs, detector, True)
        debug_im = resize(debug_im, global_Measurements)
        cv2.imshow('debug', debug_im)
        binarized = binarize(roi, kernel_binarization)
        frame = resize(frame, global_Measurements)
        binarized = resize(binarized, global_Measurements)
        
        if not len(refPt)==0:
            frame, refPt, vel, only_one_square, rands, score = ball(frame, binarized, refPt, vel, global_Measurements, only_one_square, rands, score, max_min, ratios, False)
            cv2.putText(frame, 'Score: ' + str(score), (5, 35), font, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.setMouseCallback('ROI', click)
        cv2.imshow('ROI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    # cap = cv2.VideoCapture("puntos2.mp4")
    cap = cv2.VideoCapture(0)
    main(cap)
