import cv2
import os


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {path}")
    return cap


def create_writer(path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def auto_filename(path):
    """
    Возвращает свободное имя файла, если целевое уже существует.
    example:
        input:  results/output.mp4
        output: results/output_1.mp4  (если output.mp4 существует)
    """
    base, ext = os.path.splitext(path)

    i = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{i}{ext}"
        i += 1
    return new_path
