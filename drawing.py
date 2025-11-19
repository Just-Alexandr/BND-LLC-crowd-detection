import cv2


def draw_detections(frame, detections):
    """
    Рисует bounding boxes и уверенность
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]

        text = f"{conf:.2f}"
        color = (255, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)

        cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    return frame
