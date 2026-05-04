from ultralytics import YOLO
import cv2
from pathlib import Path
from scripts.visualize import VisualizeResults
import numpy as np
from PIL import Image

def main():
    video_source = Path("./data/video1.mp4")
    output_path = "./"

    detector = YOLO("yolo26n.pt")

    cap = cv2.VideoCapture(video_source)
    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    rotation=0
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")
    
    vis = VisualizeResults()

    fps = 60
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = fps or src_fps

    # Sometimes the video is shown rotated
    out_size = (src_h, src_w) if rotation == 90 or rotation == 270 else (src_w, src_h)

    writer = None
    if output_path:
        if isinstance(output_path, str):
            output_path = Path(output_path)
        
        output_path.mkdir(exist_ok=True, parents=True) if output_path.is_dir() else output_path.parent.mkdir(exist_ok=True, parents=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, out_size)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yolo_result = detector.predict(rgb, verbose = False)[0]
            boxes = yolo_result.boxes.xyxy.cpu().numpy() if yolo_result.boxes is not None and len(yolo_result.boxes) > 0 else None

            if boxes is None:
            # or (boxes[0][2] - boxes[0][0]) < bbox_min_width:
                    annotated = frame
            else:
                image_pil = Image.fromarray(rgb)
                result = {
                    "image": image_pil,
                    "boxes": boxes.tolist(),
                    # "track_ids": track_ids.tolist(),
                    "names": ["person" for i in range(len(boxes))],
                    "scores": [1 for i in range(len(boxes))],
                    # "classes": pred_classes,
                    "yolo_conf": yolo_result.boxes.conf.cpu().numpy().tolist()
                }

                annotated_pil = vis.visualize(result)
                annotated = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow("Pipeline", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()


if __name__=='__main__':
    main()