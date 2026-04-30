import os
import torch
import numpy as np

from detector.apis import BaseDetector

class YOLO26Detector(BaseDetector):
    def __init__(self, cfg, opt=None):
        super(YOLO26Detector, self).__init__()
        self.detector_cfg = cfg
        self.detector_opt = opt
        self.model_name = cfg.get('CONFIG', 'yolo26n.pt')
        self.confidence = cfg.get('CONFIDENCE', 0.4)
        self.nms_thresh = cfg.get('NMS_THRESH', 0.5)
        self.person_class = 0  # COCO class 0 = person
        self._model = None
        self.img_size = [800, 800]

    def load_model(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_name)
        # Disable warmup to avoid cuDNN init issues
        self._model.overrides['warmup_epochs'] = 0

    def image_preprocess(self, img_source):
        return img_source
    
    def images_detection(self, imgs, orig_dim_list):
        """
        imgs: list of image paths or numpy arrays (BGR)
        orig_dim_list: tensor of original (W, H) for each image
        Returns: dets tensor of shape (n, 8) ->
                 (batch_idx, x1, y1, x2, y2, conf, conf, class_idx)
        """
        if self._model is None:
            self.load_model()

        args = self.detector_opt
        device = str(args.device) if args and hasattr(args, 'device') else 'cpu'
        print("Device: ", device)
        dets_results = []

        if isinstance(imgs, torch.Tensor):
        # Undo normalization if it was applied (values will be 0-1)
            if imgs.max() <= 1.0:
                imgs = (imgs * 255).byte()
            # Convert to list of HWC numpy arrays in RGB
            img_list = []
            for i in range(imgs.shape[0]):
                arr = imgs[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                img_list.append(arr)
        else:
            img_list = imgs if isinstance(imgs, list) else [imgs]

        results = self._model.predict(
            source=img_list,
            conf=self.confidence,
            iou=self.nms_thresh,
            classes=[self.person_class],  # only detect persons
            verbose=False,
            imgsz = self.img_size,
            device = device
        )

        for batch_idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu()       # (n, 4)
            confs = boxes.conf.cpu()      # (n,)
            cls = boxes.cls.cpu()         # (n,)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                c = cls[i]
                dets_results.append([
                    float(batch_idx),
                    float(x1), float(y1),
                    float(x2), float(y2),
                    float(conf), float(conf),
                    float(c)
                ])

        if len(dets_results) == 0:
            return None

        dets_tensor = torch.FloatTensor(dets_results)
        if args and hasattr(args, 'device'):
            dets_tensor = dets_tensor.to(args.device)
        return dets_tensor

    def detect_one_img(self, img_name):
        """
        Used by the high-level API (demo_api.py).
        Returns list of dicts: [{'bbox': [x1,y1,x2,y2,score]}]
        """
        if self._model is None:
            self.load_model()

        results = self._model(
            img_name,
            conf=self.confidence,
            iou=self.nms_thresh,
            classes=[self.person_class],
            verbose=False,
        )

        dets = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes.xyxy)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                conf = boxes.conf[i].item()
                dets.append({'bbox': [x1, y1, x2, y2, conf]})
        return dets