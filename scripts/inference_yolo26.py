import argparse
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform
import alphapose.utils.transforms as t
from time import perf_counter

from dataclasses import dataclass, field
import numpy as np

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), # Head
    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
    (5, 7), (5, 9), (6, 8), (8, 10), # Arms
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs

]

HEAD = [0, 1, 2, 3, 4, (0, 1), (0, 2), (1, 2), (1, 3), (2, 4)]

TORSO = [5, 6, 11, 12, (5, 6), (5, 11), (6, 12), (11, 12)]

ARMS = [7, 8, 9, 10, (5, 7), (5, 9), (6, 8), (8, 10)]

LEGS = [13, 14, 15, 16, (11, 13), (13, 15), (12, 14), (14, 16)]

COLORS = {
    "head":  (0, 255, 255),  # yellow
    "torso": (255, 0, 255),  # magenta
    "arms":  (0, 255, 0),    # green
    "legs":  (255, 0, 0)     # blue
}

# BGR order for OpenCV
BOUNDING_BOX_COLORS = [
    (0, 165, 255),    # Orange
    (255, 0, 128),    # Purple
    (100, 200, 0),    # Emerald Green
    (147, 20, 255),   # Hot Pink
]

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
    
    print("Average time: ", np.sum(elapsed_times) / len(elapsed_times))

    cap.release()
    cv2.destroyAllWindows()

class VideoInference():
    def __init__(
            self,
            detector_weights = './detector/yolo26/data/yolo26x.pt',
            pose_model_cfg = './configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
            pose_model_weights = './model_files/fast_res50_256x192.pth'
        ):
        dataset = DummyDataset()
        self.transformation = SimpleTransform(
                dataset=dataset,
                scale_factor=0,
                input_size=(256, 192),
                output_size=(64, 48),
                rot=0,
                sigma=2,
                train=False,
                add_dpg=False
            )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector  = YOLO(detector_weights)
        self.pose_model, cfg = self.build_pose_model(pose_model_cfg, pose_model_weights)

    def process_frame(self, frame, det_conf=0.4):
        results = self.detector.predict(
            frame,
            conf=det_conf,
            device=self.device,
            verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None and len(results[0].boxes) > 0 else None
        if boxes is None:
            result = {
                "image": frame,
                "boxes": [],
                "names": [],
                "yolo_conf": [],
                "keypoints": [],
                "pose_conf": []
            }
            return result
        
        scores = results[0].boxes.conf.cpu().numpy()
        names = results[0].boxes.cls.cpu().numpy()

        boxes_person = [b for b, n in zip(boxes, names) if n == 0]

        if len(boxes_person) > 0:
            images = []
            bboxes_resized = []
            for box in boxes_person:
                x1, y1, x2, y2 = map(int, box.tolist())
                img, bbox_resized = self.transformation.test_transform(frame, [x1, y1, x2, y2])
                images.append(img)
                bboxes_resized.append(bbox_resized)

            images = torch.stack(images, dim=0)

            with torch.no_grad():
                heatmap = self.pose_model(images.to(self.device)).cpu().numpy()   # (B, 17, hm_h, hm_w)
        
            keypoints = []
            pose_conf = []
            for i in range(heatmap.shape[0]):
                preds, maxvals = t.heatmap_to_coord_simple(
                        hms=heatmap[i],
                        bbox = bboxes_resized[i]
                    )
                keypoints.append(preds)
                pose_conf.append(maxvals)
        else:
            keypoints = []
            pose_conf = []

        result = {
            "image": frame,
            "boxes": boxes,
            "names": names,
            "yolo_conf": scores,
            "keypoints": keypoints,
            "pose_conf": pose_conf
        }

        return result

    def build_pose_model(self, cfg_path, checkpoint_path):
        cfg = update_config(cfg_path)
        model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.to(self.device).eval()
        print(f'Loaded AlphaPose model from {checkpoint_path}')
        return model, cfg

    def draw_joints(self, img, keypoints, kp_score: list = None, thresh: float = 0.3, skeleton: bool = True):
        """Draw pose joints and skeleton

        Args:
            img (_type_): _description_
            keypoints (_type_): _description_
            kp_score (list, optional): _description_. Defaults to None.
            thresh (float, optional): _description_. Defaults to 0.3.
            skeleton (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        img = img.copy()
        K = keypoints.shape[0]

        low_conf = []
        for i in range(K):
            x, y = keypoints[i]

            # skip low confidence points
            if kp_score is not None and kp_score[i] < thresh:
                low_conf.append(i)
                continue

            x, y = int(x), int(y)

            if i in HEAD:
                color = COLORS['head']
            elif i in ARMS:
                color = COLORS['arms']
            elif i in TORSO:
                color = COLORS['torso']
            elif i in LEGS:
                color = COLORS['legs']
            else:
                color = (0, 0, 0)

            cv2.circle(
                img,
                (x, y),
                radius=3,
                color=color,
                thickness=-1
            )
        
        if skeleton:
            for (a, b) in COCO_PAIRS:
                if a in low_conf or b in low_conf:
                    continue
                if (a, b) in HEAD:
                    color = COLORS['head']
                elif (a, b) in ARMS:
                    color = COLORS['arms']
                elif (a, b) in TORSO:
                    color = COLORS['torso']
                elif (a, b) in LEGS:
                    color = COLORS['legs']
                else:
                    color = (0, 0, 0)

                x1, y1 = map(int, keypoints[a])
                x2, y2 = map(int, keypoints[b])
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

        return img
    
    def draw_result(self, frame, result: dict):
        """Draw bounding boxes and joints

        Args:
            frame (_type_): _description_
            result (dict): dictionary containing the inferencew results

        Returns:
            _type_: annotated frame
        """
        if len(result['boxes']) <= 0:
            return frame

        names_to_cls = self.detector.names
        i = 0
        for box, name, yolo_conf in zip(
            result['boxes'], result['names'], result['yolo_conf']
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            color = BOUNDING_BOX_COLORS[int(name) % len(BOUNDING_BOX_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(frame, names_to_cls[name], (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(frame, f"{yolo_conf:.2f}", (x2 - 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if name == 0:
                frame = self.draw_joints(frame, result['keypoints'][i], result['pose_conf'][i])
                i += 0
        return frame

class DummyDataset:
    def __init__(self):
        self.joint_pairs = [
            (1,2),(3,4),(5,6),(7,8),
            (9,10),(11,12),(13,14),(15,16)
        ]
        self.num_joints = 17

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