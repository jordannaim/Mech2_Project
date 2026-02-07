import cv2
import numpy as np

# ---------- CONFIG ----------
# Expected 4 black circular markers around the cups (TL, TR, BR, BL after ordering)
OUT_W, OUT_H = 900, 900

DST = np.array([
    [0, 0],               # TL
    [OUT_W - 1, 0],        # TR
    [OUT_W - 1, OUT_H-1],  # BR
    [0, OUT_H - 1]         # BL
], dtype=np.float32)

def order_points(pts):
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # TR
    rect[3] = pts[np.argmax(d)]  # BL
    return rect

def compute_homography_from_black_circles(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # HoughCircles likes blurred input
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]

    # Tune these to your setup (marker size in the image matters most)
    minR = max(8, int(min(h, w) * 0.01))
    maxR = max(minR + 2, int(min(h, w) * 0.10))

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.12),
        param1=120,   # Canny high threshold internal to Hough
        param2=35,    # accumulator threshold (lower = more detections)
        minRadius=minR,
        maxRadius=maxR
    )

    if circles is None:
        return None, None

    circles = np.round(circles[0]).astype(int)  # (N,3) with x,y,r

    # Keep circles that are actually dark at the center (reject bright false positives)
    good = []
    for x, y, r in circles:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if gray[y, x] < 75:  # center pixel dark threshold; tune 60-120
            good.append((x, y, r))

    if len(good) < 4:
        return None, circles

    # Pick 4 that are farthest apart (greedy) to favor corner markers
    # Start from largest radius candidates
    good.sort(key=lambda t: t[2], reverse=True)

    chosen = []
    for cand in good:
        if not chosen:
            chosen.append(cand)
            continue
        # enforce spacing
        ok = True
        for cx, cy, cr in chosen:
            if (cand[0]-cx)**2 + (cand[1]-cy)**2 < (0.15*min(h, w))**2:
                ok = False
                break
        if ok:
            chosen.append(cand)
        if len(chosen) == 4:
            break

    if len(chosen) < 4:
        # fallback: just take first 4
        chosen = good[:4]

    pts = np.array([[x, y] for x, y, r in chosen], dtype=np.float32)
    src = order_points(pts)
    H = cv2.getPerspectiveTransform(src, DST)
    return H, circles

def warp_top_view(frame_bgr, H):
    return cv2.warpPerspective(frame_bgr, H, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    H_last = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw = frame.copy()

        H, circles = compute_homography_from_black_circles(frame)
        if H is not None:
            H_last = H

        # Draw detected circles for debugging
        if circles is not None:
            for x, y, r in circles:
                cv2.circle(raw, (x, y), r, (0, 255, 0), 2)
                cv2.circle(raw, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("RAW", raw)

        if H_last is not None:
            top = warp_top_view(frame, H_last)
            cv2.imshow("TOP_VIEW", top)
        else:
            blank = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for 4 black circles...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("TOP_VIEW", blank)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
