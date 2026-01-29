# Source - https://stackoverflow.com/q
# Posted by Mitch, modified by community. See post 'Timeline' for change history
# Retrieved 2026-01-28, License - CC BY-SA 3.0

import cv2

cv2.NamedWindow("w1", cv.CV_WINDOW_AUTOSIZE)
capture = cv2.CaptureFromCAM(0)

def repeat():

    frame = cv2.QueryFrame(capture)
    cv.ShowImage("w1", frame)


while True:
    repeat()


