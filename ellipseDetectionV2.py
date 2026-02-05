import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ---- Tunables ----
CANNY_LO, CANNY_HI = 80, 160


MIN_COMP_PIX = 90          # minimum edge pixels in a component to attempt ellipse fit
MAX_COMP_PIX = 4000         # reject giant components (table lines etc.)

MIN_AXIS = 40          # px; reject tiny ellipses
MAX_AXIS = 380              # px; reject huge ellipses

MIN_ASPECT = 0.3           # min(minAxis/maxAxis)
MAX_ASPECT = 1.00

SAMPLE_N = 300              # points sampled along ellipse perimeter for scoring
RING_THICK = 5              # px tolerance for edge hit near perimeter point
SUPPORT_THRESH = 0.6       # required perimeter support ratio

NMS_CENTER_DIST = 18        # px
NMS_AXIS_DIST = 12          # px

def ellipse_points(cx, cy, a, b, ang_deg, n=120):
    # a,b are semi-axes
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    ca, sa = np.cos(np.deg2rad(ang_deg)), np.sin(np.deg2rad(ang_deg))
    x = a*np.cos(t)
    y = b*np.sin(t)
    xr = ca*x - sa*y + cx
    yr = sa*x + ca*y + cy
    return np.stack([xr, yr], axis=1)

def support_score(edges, cx, cy, MA, ma, angle_deg, thick=2, n=120):
    # MA,ma are full axes from cv2.fitEllipse; convert to semi-axes
    a = MA / 2.0
    b = ma / 2.0
    pts = ellipse_points(cx, cy, a, b, angle_deg, n=n)

    h, w = edges.shape
    hits = 0
    for (xf, yf) in pts:
        x = int(round(xf)); y = int(round(yf))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x0 = max(0, x - thick); x1 = min(w, x + thick + 1)
        y0 = max(0, y - thick); y1 = min(h, y + thick + 1)
        if np.any(edges[y0:y1, x0:x1] > 0):
            hits += 1
    return hits / float(n)

def nms_ellipses(ells):
    # ells: list of dict with keys: cx,cy,MA,ma,angle,score
    ells = sorted(ells, key=lambda d: d["score"], reverse=True)
    kept = []
    for e in ells:
        ok = True
        for k in kept:
            dc = np.hypot(e["cx"] - k["cx"], e["cy"] - k["cy"])
            da = abs(e["MA"] - k["MA"]) + abs(e["ma"] - k["ma"])
            if dc < NMS_CENTER_DIST and da < NMS_AXIS_DIST:
                ok = False
                break
        if ok:
            kept.append(e)
    return kept

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #blur = cv2.GaussianBlur(blur, (5, 5), 0)

    edges = cv2.Canny(blur, CANNY_LO, CANNY_HI)

    # Stabilize edges slightly (does not change the look much)
    edges_stable = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, K3, iterations=1)
    edges_stable = cv2.dilate(edges_stable, K3, iterations=3)

    # Connected components on edge pixels (no requirement for closed contours)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((edges_stable > 0).astype(np.uint8), connectivity=8)

    candidates = []
    for lab in range(1, num):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < MIN_COMP_PIX or area > MAX_COMP_PIX:
            continue

        ys, xs = np.where(labels == lab)
        if len(xs) < 5:
            continue

        pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

        # fit ellipse to the component’s edge pixels
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(pts)

        if MA < MIN_AXIS or ma < MIN_AXIS or MA > MAX_AXIS or ma > MAX_AXIS:
            continue

        aspect = min(MA, ma) / max(MA, ma)
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        # score by perimeter support in the original edge map
        sc = support_score(edges_stable, cx, cy, MA, ma, angle, thick=RING_THICK, n=SAMPLE_N)
        if sc < SUPPORT_THRESH:
            continue

        candidates.append({"cx": cx, "cy": cy, "MA": MA, "ma": ma, "angle": angle, "score": sc})

    kept = nms_ellipses(candidates)

    for e in kept:
        cx, cy = e["cx"], e["cy"]
        MA, ma = e["MA"], e["ma"]
        angle = e["angle"]

        cv2.ellipse(out, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, f'{e["score"]:.2f}', (int(cx)+6, int(cy)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    cv2.imshow("edges_canny", edges)
    cv2.imshow("edges_stable", edges_stable)
    cv2.imshow("ellipses", out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
