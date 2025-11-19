import cv2
from tqdm import tqdm

from detector import load_model, detect_people
from drawing import draw_detections
from utils import open_video, create_writer, auto_filename


INPUT = "data/crowd.mp4"
OUTPUT = auto_filename("results/output.mp4")
IMGSZ = 3072


def main():
    model = load_model("yolov8x.pt")

    cap = open_video(INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = create_writer(OUTPUT, fps, w, h)

    print(f"Кадров для обработки: {count}")

    for _ in tqdm(range(count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people(model, frame, imgsz=IMGSZ)
        frame = draw_detections(frame, detections)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"saved at {OUTPUT}")


if __name__ == "__main__":
    main()
