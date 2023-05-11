import cv2 as cv
import numpy as np


def center_xy(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def count(video):
    backgroundSubtractorMOG2 = cv.createBackgroundSubtractorMOG2()
    counted, detected = 0, []

    while True:
        try:
            ret, frame = video.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            blur = cv.GaussianBlur(gray, (5, 5), 5)
            img_sub = backgroundSubtractorMOG2.apply(blur)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
            dilate = cv.dilate(img_sub, np.ones((5, 5)))
            dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
            dilate = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)

            counter_shape, h = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            cv.line(frame, (50, 550), (600, 550), (255, 0, 0), 3)

            for (i, c) in enumerate(counter_shape):
                (x, y, w, h) = cv.boundingRect(c)

                if (w >= 80) and (h >= 80):
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    center = center_xy(x, y, w, h)
                    detected.append(center)

                    for (x, y) in detected:
                        if (550 - 7) < y < (550 + 7) and 50 < x < 600:
                            counted += 1
                            detected.remove((x, y))

            cv.putText(frame, "counter - " + str(counted), (450, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv.imshow('video original', frame)

            if cv.waitKey(10) and cv.waitKey(10) % 256 == 27:
                break

        except BaseException as e:
            print(e.args)
            break

    cv.destroyAllWindows()
    video.release()


video = cv.VideoCapture('video.mp4')
count(video)
