import numpy as np
import cv2
import random
import time

has_clicked = False
refPt = []

def bounce_ball_TOP_BOTTOM(velocityY):
    return -1 * velocityY

def bounce_ball_LEFT_RIGHT(velocityX):
    return -1 * velocityX

# def click_canvas(event, x, y, flags, param):
#     global ref_points, has_clicked
#     if event == cv2.EVENT_LBUTTONDOWN:
#         ref_points = [x,y]
#         print("Clicked on: ", x, y)
#         has_clicked = True

def click(event, x, y, flags, param):
    global refPt, has_clicked

    if not has_clicked:
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [x, y]
            print("Clicked: ", refPt[0], ", ", refPt[1])
            has_clicked = True

def ball(frame, binarized, center, vel, global_Measurements, only_one_square, rands, score, debug=False):
    global_Width = global_Measurements[0]
    global_Height = global_Measurements[1]

    print("Ball Center: ", center)

    ball_diameter = 20
    ball_radius = ball_diameter/2
    ball_color = (255, 125, 0)
    thickness = -1

    velX = vel[0]
    velY = vel[1]

    out_of_bounds = False

    if (center[0] - 10) <= 0:
        velX = bounce_ball_TOP_BOTTOM(velX)
        out_of_bounds = True
    elif (center[0] + 10) >= global_Width:
        velX = bounce_ball_TOP_BOTTOM(velX)
        out_of_bounds = True
    if (center[1] - 10) <= 0:
        velY = bounce_ball_LEFT_RIGHT(velY)
        out_of_bounds = True
    elif (center[1] + 10) >= global_Height:
        velY = bounce_ball_LEFT_RIGHT(velY)
        out_of_bounds = True

    #  AQUI EL PEDO QUE DICE ES: 
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    # Lo que tengo pensado es que el binarizado en esa posicion  que estamos checando, es: 206, 213, 224... por alguna razon no esta binarizaddo
    # como 0 o 255
    
    print("Binarized: ", binarized[center[1]][center[0]])

    if not out_of_bounds and binarized[center[1]][center[0]-10] == 255:
        velX = bounce_ball_TOP_BOTTOM(velX)
    if not out_of_bounds and binarized[center[1]][center[0]+10] == 255:
        velX = bounce_ball_TOP_BOTTOM(velX)
    if not out_of_bounds and binarized[center[1]+10][center[0]] == 255:
        velY = bounce_ball_LEFT_RIGHT(velY)
    if not out_of_bounds and binarized[center[1]-10][center[0]] == 255:
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
    return cv2.resize(image, (global_Measurements[0], global_Measurements[1]))

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
    # print(keypoints)

    points = []
    for keypoint in keypoints:
        points.append([keypoint.pt[0], keypoint.pt[1]])
    # print(points)

    if debug:
        im_with_keypoints = cv2.drawKeypoints(green_mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pts = np.array(points, np.int32)
    # print("Points: ", pts)
    pts = pts.reshape((-1, 1, 2))
    hull = cv2.convexHull(pts, False)

    # print(hull)
    
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

def mainImage(image, debug=False):
    global_Width = 1280
    global_Height = 720
    global_Measurements = [global_Width, global_Height]
    only_one_square = True
    vel = [30, 30]
    score = 0
    rands = [random.randint(1, global_Width-50), random.randint(1, global_Height-50)]

    center = [global_Width//2, global_Height//2]
    detector = getBlobDetector()

    kernel_blobs = np.ones((9, 9), np.uint8)
    kernel_binarization = np.ones((5, 5), np.uint8)

    cv2.imshow("img", image)
    cv2.namedWindow('ROI')

    frame = image

    while(True):
        roi, debug_im = getROI(frame, kernel_blobs, detector, True)
        cv2.imshow('roi', debug_im)
        binarized = binarize(roi, kernel_binarization)

        game_show, center, vel, only_one_square, rands, score = ball(frame, binarized, center, vel, global_Measurements, only_one_square, rands, score, True)
        # cv2.setMouseCallback('ROI', click_canvas)
        cv2.imshow('ROI', game_show)
        
        time.sleep(0.025)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main(cap, debug=False):
    global_Width = 1280
    global_Height = 720
    global_Measurements = [global_Width, global_Height]
    only_one_square = True
    vel = [30, 30]
    score = 0
    rands = [random.randint(1, global_Width-50), random.randint(1, global_Height-50)]

    global refPt
    # center = [global_Width//2, global_Height//2]
    detector = getBlobDetector()

    kernel_blobs = np.ones((9, 9), np.uint8)
    kernel_binarization = np.ones((5, 5), np.uint8)

    cv2.namedWindow('ROI')

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        roi, debug_im = getROI(frame, kernel_blobs, detector)
        binarized = binarize(roi, kernel_binarization)
        frame = resize(frame, global_Measurements)
        binarized = resize(binarized, global_Measurements)
        
        if not len(refPt)==0:
            frame, refPt, vel, only_one_square, rands, score = ball(frame, binarized, refPt, vel, global_Measurements, only_one_square, rands, score, False)
        cv2.setMouseCallback('ROI', click)
        cv2.imshow('ROI', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture("puntos.mp4")
    img = cv2.imread('puntos_img2.jpg')
    main(cap)
    # mainImage(img)
