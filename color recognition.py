import cv2
import numpy as np

MIN_CONTOUR_AREA = 800  

# HSV ranges for each color. Red wraps around the hue circle (0 and 180),
# so it needs two ranges combined.
COLOR_RANGES = {
    "Red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "Green": [
        (np.array([40, 70, 70]), np.array([80, 255, 255])),
    ],
    "Blue": [
        (np.array([100, 120, 70]), np.array([130, 255, 255])),
    ],
}

# BGR draw colors for each label (just for the bounding box/text, not detection)
DRAW_COLORS = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
}

def get_mask(hsv_frame, ranges):
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv_frame, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    # Morphological cleanup: erode away small noise, then dilate to
    # restore/fill the shape of real objects.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_colors():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Detecting red, green, and blue objects. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, ranges in COLOR_RANGES.items():
            mask = get_mask(hsv, ranges)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                draw_color = DRAW_COLORS[color_name]
                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
                cv2.putText(
                    frame, color_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2
                )

        cv2.imshow("Color Detection - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_colors()