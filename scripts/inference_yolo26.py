# pose_pipeline.py
# Run from AlphaPose repo root:
#   python pose_pipeline.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \
#                           --checkpoint pretrained_models/fast_res50_256x192.pth \
#                           --source 0   (or path/to/video.mp4 or image.jpg)

import argparse
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_pose_model(cfg_path, checkpoint_path):
    cfg = update_config(cfg_path)
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f'Loaded AlphaPose model from {checkpoint_path}')
    return model, cfg


def preprocess_crop(crop, input_size=(192, 256)):
    """Resize crop to model input size, normalize, return tensor."""
    w, h = input_size  # 192 x 256
    resized = cv2.resize(crop, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(DEVICE)


def heatmap_to_coords(heatmap, box_x1, box_y1, box_w, box_h):
    """
    Convert AlphaPose heatmap output to image-space keypoint coords.
    heatmap: (1, num_joints, hm_h, hm_w) tensor
    Returns: (num_joints, 2) numpy array of (x, y) coords in original image space
             (num_joints,)   numpy array of confidence scores
    """
    hm = heatmap[0].cpu().numpy()          # (num_joints, hm_h, hm_w)
    num_joints, hm_h, hm_w = hm.shape

    coords = np.zeros((num_joints, 2), dtype=np.float32)
    scores = np.zeros(num_joints, dtype=np.float32)

    for j in range(num_joints):
        flat_idx = np.argmax(hm[j])
        py = flat_idx // hm_w
        px = flat_idx % hm_w
        scores[j] = hm[j, py, px]

        # map from heatmap space -> crop space -> image space
        x = (px + 0.5) / hm_w * box_w + box_x1
        y = (py + 0.5) / hm_h * box_h + box_y1
        coords[j] = [x, y]

    return coords, scores


def draw_pose(frame, coords, scores, conf_thresh=0.3):
    for a, b in COCO_SKELETON:
        if scores[a] > conf_thresh and scores[b] > conf_thresh:
            x1, y1 = int(coords[a][0]), int(coords[a][1])
            x2, y2 = int(coords[b][0]), int(coords[b][1])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)

    for j in range(len(coords)):
        if scores[j] > conf_thresh:
            cx, cy = int(coords[j][0]), int(coords[j][1])
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)


def process_frame(frame, detector, pose_model, det_conf=0.4):
    results = detector.predict(frame, classes=[0], conf=det_conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return frame

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # clamp to frame bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        inp = preprocess_crop(crop)

        with torch.no_grad():
            heatmap = pose_model(inp)   # (1, 17, hm_h, hm_w)

        coords, scores = heatmap_to_coords(
            heatmap, x1, y1, x2 - x1, y2 - y1
        )
        draw_pose(frame, coords, scores)

    return frame


def run(args):
    detector  = YOLO('yolo26s.pt')
    pose_model, cfg = build_pose_model(args.cfg, args.checkpoint)

    source = args.source
    # try casting to int for webcam index
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # single image
    if isinstance(source, str) and source.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        frame = cv2.imread(source)
        result = process_frame(frame, detector, pose_model)
        cv2.imshow('Pose', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # video / webcam
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f'Cannot open: {source}'

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = process_frame(frame, detector, pose_model)
        cv2.imshow('Pose', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',        required=True,
                        help='AlphaPose config yaml')
    parser.add_argument('--checkpoint', required=True,
                        help='AlphaPose .pth checkpoint')
    parser.add_argument('--source',     default='0',
                        help='0=webcam, or image/video path')
    parser.add_argument('--det-conf',   type=float, default=0.4)
    args = parser.parse_args()
    run(args)