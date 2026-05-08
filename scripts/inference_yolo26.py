import argparse
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from alphapose.models import builder
from .required_modules.utils.config import update_config
from .required_modules.simple_transform import SimpleTransform
import scripts.required_modules.utils.transforms as t
from time import perf_counter

from dataclasses import dataclass
import numpy as np

from scripts.visualize import draw_joints

# BGR order for OpenCV
BOUNDING_BOX_COLORS = [
    (0, 165, 255),    # Orange
    (255, 0, 128),    # Purple
    (100, 200, 0),    # Emerald Green
    (147, 20, 255),   # Hot Pink
]

@dataclass
class Detection:
    box: np.ndarray          # [x1, y1, x2, y2]
    name: int                # class id
    yolo_conf: float
    keypoints: np.ndarray | None = None   # only for persons
    pose_conf: np.ndarray | None = None   # only for persons

    @property
    def is_person(self) -> bool:
        return self.name == 0

    @property
    def has_pose(self) -> bool:
        return self.keypoints is not None

def main(args):

    source = args.source
    # try casting to int for webcam index
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    pose = VideoInference(
        detector_weights=args.det_weights,
        pose_model_weights=args.checkpoint,
        pose_model_cfg=args.cfg
    )

    # single image
    if isinstance(source, str) and source.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        frame = cv2.imread(source)
        result = pose.process_frame(frame)

        frame = pose.draw_result(frame, result)

        cv2.imshow('Pose', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # video / webcam
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f'Cannot open: {source}'

    elapsed_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = perf_counter()
        result = pose.process_frame(frame)
        frame = pose.draw_result(frame, result)
            
        elapsed_times.append(perf_counter() - start_time)
        cv2.imshow('Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"Average time: {(1000 * np.sum(elapsed_times) / len(elapsed_times)):.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

class VideoInference():
    def __init__(
            self,
            detector_weights = './detector/yolo26/data/yolo26x.pt',
            pose_model_cfg = './configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
            pose_model_weights = './model_files/fast_res50_256x192.pth'
        ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector  = YOLO(detector_weights)
        self.pose_model, cfg = self.build_pose_model(pose_model_cfg, pose_model_weights)

        dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        
        self.transformation = SimpleTransform(
                dataset=dataset,
                scale_factor=0,
                input_size=cfg.DATA_PRESET.IMAGE_SIZE,
                output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
                rot=0,
                sigma=2,
                train=False,
                add_dpg=False
            )

    def video_pipeline(self, source):
        # video / webcam
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f'Cannot open: {source}'

        elapsed_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            start_time = perf_counter()
            result = self.process_frame(frame)
            pts = np.array([r.keypoints for r in result])
            print(pts)

            frame = self.draw_result(frame, result)
                
            elapsed_times.append(perf_counter() - start_time)
            cv2.imshow('Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"Average time: {(1000 * np.sum(elapsed_times) / len(elapsed_times)):.2f} ms")

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame, det_conf=0.4):
        results = self.detector.predict(
            frame,
            conf=det_conf,
            device=self.device,
            verbose=False
        )
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        names = results[0].boxes.cls.cpu().numpy()
        detections = [
            Detection(box=box, name=int(name), yolo_conf=conf)
            for box, name, conf in zip(boxes, names, scores)
        ]

        person_dets = [d for d in detections if d.is_person]

        if len(person_dets) > 0:
            images = []
            bboxes_resized = []
            for d in person_dets:
                x1, y1, x2, y2 = map(int, d.box.tolist())
                img, bbox_resized = self.transformation.test_transform(frame, [x1, y1, x2, y2])
                images.append(img)
                bboxes_resized.append(bbox_resized)

            images = torch.stack(images, dim=0)

            with torch.no_grad():
                heatmap = self.pose_model(images.to(self.device)).cpu().numpy()   # (B, 17, hm_h, hm_w)
        
            for d, hm, bbox_r in zip(person_dets, heatmap, bboxes_resized):
                preds, maxvals = t.heatmap_to_coord_simple(
                        hms=hm,
                        bbox = bbox_r
                    )
                d.keypoints = preds
                d.pose_conf = maxvals

        return detections

    def build_pose_model(self, cfg_path, checkpoint_path):
        cfg = update_config(cfg_path)
        model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device).eval()
        print(f'Loaded AlphaPose model from {checkpoint_path}')
        return model, cfg
    
    def draw_result(self, frame, detections: list[Detection]):
        names_to_cls = self.detector.names

        

        for det in detections:
            x1, y1, x2, y2 = map(int, det.box.tolist())
            color = BOUNDING_BOX_COLORS[int(det.name) % len(BOUNDING_BOX_COLORS)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, names_to_cls[int(det.name)], (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"{det.yolo_conf:.2f}", (x2 - 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if det.has_pose:
                frame = draw_joints(frame, det.keypoints, det.pose_conf)

        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True,
                        help='AlphaPose config yaml')
    parser.add_argument('--checkpoint', required=True,
                        help='AlphaPose .pth checkpoint')
    parser.add_argument('--source', default='0',
                        help='0=webcam, or image/video path')
    parser.add_argument('--det-conf', type=float, default=0.4)
    parser.add_argument('--det-weights', type=str, default='./detector/yolo26/data/yolo26m.pt')
    args = parser.parse_args()
    main(args)