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

from dataclasses import dataclass
import numpy as np

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

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.detector  = YOLO(detector_weights)
        self.pose_model, cfg = self.build_pose_model(pose_model_cfg, pose_model_weights)

        dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

        self.n_joints = dataset.num_joints

        # print(dataset.joint_pairs)
        
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
                frame = self.draw_joints(frame, det.keypoints, det.pose_conf)

        return frame

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

        joint_pairs, joint_color, lines_color = self.get_joint_pairs(K, 'coco')



        low_conf = []
        for i in range(K):
            x, y = keypoints[i]

            # skip low confidence points
            if kp_score is not None and kp_score[i] < thresh:
                low_conf.append(i)
                continue

            x, y = int(x), int(y)

            color = joint_color[i]

            cv2.circle(
                img,
                (x, y),
                radius=3,
                color=color,
                thickness=-1
            )
        
        if skeleton:
            for pair, line_color in zip(joint_pairs, lines_color):
                
                a, b = pair
                if a in low_conf or b in low_conf:
                    continue

                x1, y1 = map(int, keypoints[a])
                x2, y2 = map(int, keypoints[b])
                cv2.line(img, (x1, y1), (x2, y2), line_color, 2)

        return img

    def get_joint_pairs(self, kp_num, format):
        if kp_num == 17:
            if format == 'coco':

                l_pair = [
                    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), # Head
                    (5, 6), (5, 11), (6, 12), (11, 12), # Torso
                    (5, 7), (5, 9), (6, 8), (8, 10), # Arms
                    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
                ]

                line_color = [COLORS['head']] * 5 + [COLORS['torso']] * 4 + [COLORS['arms']] * 4 + [COLORS['legs']] * 4

                HEAD = [0, 1, 2, 3, 4]

                TORSO = [5, 6, 11, 12]

                ARMS = [7, 8, 9, 10]

                LEGS = [13, 14, 15, 16]

                p_color = self.get_point_colors(HEAD, TORSO, ARMS, LEGS)

            elif format == 'mpii':
                l_pair = [
                    (8, 9), (11, 12), (11, 10), (2, 1), (1, 0),
                    (13, 14), (14, 15), (3, 4), (4, 5),
                    (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)
                ]
                p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
            else:
                raise NotImplementedError
        elif kp_num == 136:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
                (17, 18), (18, 19), (19, 11), (19, 12),
                (11, 13), (12, 14), (13, 15), (14, 16),
                (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
                (26, 27),(27, 28),(28, 29),(29, 30),(30, 31),(31, 32),(32, 33),(33, 34),(34, 35),(35, 36),(36, 37),(37, 38),#Face
                (38, 39),(39, 40),(40, 41),(41, 42),(43, 44),(44, 45),(45, 46),(46, 47),(48, 49),(49, 50),(50, 51),(51, 52),#Face
                (53, 54),(54, 55),(55, 56),(57, 58),(58, 59),(59, 60),(60, 61),(62, 63),(63, 64),(64, 65),(65, 66),(66, 67),#Face
                (68, 69),(69, 70),(70, 71),(71, 72),(72, 73),(74, 75),(75, 76),(76, 77),(77, 78),(78, 79),(79, 80),(80, 81),#Face
                (81, 82),(82, 83),(83, 84),(84, 85),(85, 86),(86, 87),(87, 88),(88, 89),(89, 90),(90, 91),(91, 92),(92, 93),#Face
                (94,95),(95,96),(96,97),(97,98),(94,99),(99,100),(100,101),(101,102),(94,103),(103,104),(104,105),#LeftHand
                (105,106),(94,107),(107,108),(108,109),(109,110),(94,111),(111,112),(112,113),(113,114),#LeftHand
                (115,116),(116,117),(117,118),(118,119),(115,120),(120,121),(121,122),(122,123),(115,124),(124,125),#RightHand
                (125,126),(126,127),(115,128),(128,129),(129,130),(130,131),(115,132),(132,133),(133,134),(134,135)#RightHand
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                    (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
        
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                        (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
        elif kp_num == 133:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 7), (7, 9), (6, 8), (8, 10),# Body
                (11, 13), (12, 14), (13, 15), (14, 16),
                (18, 19), (21, 22), (20, 22), (17, 19), (15, 19), (16, 22), 
                (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), 
                (35, 36), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (48, 49), 
                (50, 51), (51, 52), (52, 53), (54, 55), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), 
                (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), 
                (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), 
                (91, 92), (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98), (98, 99), (91, 100), (100, 101), (101, 102), 
                (102, 103), (91, 104), (104, 105), (105, 106), (106, 107), (91, 108), (108, 109), (109, 110), (110, 111), (112, 113), 
                (113, 114), (114, 115), (115, 116), (112, 117), (117, 118), (118, 119), (119, 120), (112, 121), (121, 122), (122, 123), 
                (123, 124), (112, 125), (125, 126), (126, 127), (127, 128), (112, 129), (129, 130), (130, 131), (131, 132)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
        
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                        (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255)]
        elif kp_num == 68:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
                (17, 18), (18, 19), (19, 11), (19, 12),
                (11, 13), (12, 14), (13, 15), (14, 16),
                (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
                (26, 27), (27, 28), (28, 29), (29, 30), (26, 31), (31, 32), (32, 33), (33, 34), 
                (26, 35), (35, 36), (36, 37), (37, 38), (26, 39), (39, 40), (40, 41), (41, 42), 
                (26, 43), (43, 44), (44, 45), (45, 46), (47, 48), (48, 49), (49, 50), (50, 51), 
                (47, 52), (52, 53), (53, 54), (54, 55), (47, 56), (56, 57), (57, 58), (58, 59), 
                (47, 60), (60, 61), (61, 62), (62, 63), (47, 64), (64, 65), (65, 66), (66, 67)
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                    (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
        
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                        (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
        elif kp_num == 26:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),# Body
                (17, 18), (18, 19), (19, 11), (19, 12),
                (11, 13), (12, 14), (13, 15), (14, 16),
                (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),# Foot
            ]
            p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                    (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (77, 255, 255)] # foot
        
            line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                        (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                        (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36), 
                        (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
        elif kp_num == 21:
            l_pair = [
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), 
                (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (21, 22), (22, 23),
                (23, 24), (24, 25), (21, 26), (26, 27), (27, 28), (28, 29), (21, 30), 
                (30, 31), (31, 32), (32, 33), (21, 34), (34, 35), (35, 36), (36, 37), 
                (21, 38), (38, 39), (39, 40), (40, 41)
            ]
            p_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255) ]
        
            line_color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
                    (255, 255, 255) ]
        else:
            raise NotImplementedError
        
        return l_pair, p_color, line_color

    def get_point_colors(self, HEAD, TORSO, ARMS, LEGS):
        ordered_points = HEAD + TORSO + ARMS + LEGS
        ordered_idx = np.argsort(ordered_points)
        p_color = [COLORS['head']] * len(HEAD) + [COLORS['torso']] * len(TORSO) + [COLORS['arms']] * len(ARMS) + [COLORS['legs']] * len(LEGS)

        p_color = [p_color[i] for i in ordered_idx]
        return p_color


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