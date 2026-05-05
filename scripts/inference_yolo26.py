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
            classes=[0],
            conf=det_conf,
            device=self.device,
            verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None and len(results[0].boxes) > 0 else None
        if boxes is None:
            result = {
                "image": frame,
                "boxes": [],
                "yolo_conf": [],
                "keypoints": [],
                "pose_conf": []
            }
            return result
        
        scores = results[0].boxes.conf.cpu().numpy()

        images = []
        bboxes_resized = []
        for box in boxes:
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

        result = {
            "image": frame,
            "boxes": boxes,
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

    def draw_joints(self, img, keypoints, kp_score=None, thresh=0.3, skeleton = True):
        """
        keypoints: (K, 2)
        kp_score: (K,) optional confidence
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
    
    def draw_result(self, frame, result):
        if len(result['boxes']) <= 0:
            return frame
        for box, yolo_conf, keypoints, pose_conf in zip(result['boxes'], result['yolo_conf'], result['keypoints'], result['pose_conf']):
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, f'{yolo_conf:.2f}', (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            frame = self.draw_joints(frame, keypoints, pose_conf)
        return frame

class DummyDataset:
    def __init__(self):
        self.joint_pairs = [
            (1,2),(3,4),(5,6),(7,8),
            (9,10),(11,12),(13,14),(15,16)
        ]
        self.num_joints = 17


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