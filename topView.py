import cv2
import numpy as np

def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect

def four_point_warp(image, pts, out_w=None, out_h=None):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    maxW = int(max(wA, wB)) if out_w is None else int(out_w)

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxH = int(max(hA, hB)) if out_h is None else int(out_h)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    return warped, M

# Example usage:
# frame = cv2.imread("angled.jpg")

# Replace with your 4 corner points of the table region in the angled image:
# pts = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
# top_view, H = four_point_warp(frame, pts)
# cv2.imwrite("top_view.png", top_view)
