from ultralytics import YOLO


def load_model(weights_path="yolov8x.pt"):
    """
    Загружает модель YOLO
    """
    return YOLO(weights_path)


def detect_people(model, frame, imgsz=640, conf_threshold=0.25):
    """
    Выполняет детекцию людей на кадре
    Возвращает список словарей:
      {'bbox': [x1,y1,x2,y2], 'conf': float, 'cls': int}
    """
    results = model(frame, imgsz=imgsz, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id != 0:
            continue
        if conf < conf_threshold:
            continue

        detections.append({
            "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist(),
            "conf": conf,
            "cls": cls_id
        })
    return detections
