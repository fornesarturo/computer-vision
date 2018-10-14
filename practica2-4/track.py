import numpy as np
import cv2

def getAverage (matches, train_kp, keep_n = 5):
    # Keep n of the top matches
    topMatches = matches[:keep_n]
    avg_x = 0
    avg_y = 0
    points = []
    for match in topMatches:
        x = train_kp[match.trainIdx].pt[0]
        y = train_kp[match.trainIdx].pt[1]
        avg_x += x
        avg_y += y
        points.append([x, y])
    avg_x //= keep_n
    avg_y //= keep_n

    avg = (int(avg_x), int(avg_y))
    return avg, points, topMatches


def main(debug=False, image="drive_logo.jpg", polyline=False, circle_color=(0, 255, 0), polyline_color=(0, 255, 255)):
    if debug:
        from matplotlib import pyplot as plt
    cap = cv2.VideoCapture(0)
    modelImg = cv2.imread(image)

    # Initiate ORB (Oriented FAST and Rotated BRIEF) 
    # *free version of SIFT developed by the OpenCV guys
    orb = cv2.ORB_create()

    # find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(modelImg, None)
    if debug:
        drawn1 = cv2.drawKeypoints(modelImg, kp1, None, color=(0, 255, 0), flags=4)
        plt.imshow(drawn1), plt.show()

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    while (cap.isOpened()):
        ret, frame = cap.read()

        resKp, des2 = orb.detectAndCompute(frame, None)

        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        avg, points, topMatches = getAverage(matches, train_kp=resKp, keep_n=5)

        if debug:
            draw_matches = cv2.drawMatches(modelImg, kp1, frame, resKp, topMatches, None, flags=2)
            plt.imshow(draw_matches), plt.show()

        frame = cv2.circle(frame, center=avg, radius=100, color=circle_color, thickness=10)

        if polyline:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, polyline_color, thickness=10)

        cv2.imshow('Detecting...', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    main(image="drive_logo.jpg", polyline=False, debug=False)
