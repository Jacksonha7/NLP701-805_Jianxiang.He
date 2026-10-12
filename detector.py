"""YOLO-World (MMDetection) open-vocabulary detector, condensed from TStar/interface_yolo.py."""

import os.path as osp

import torch


class YoloWorldDetector:
    def __init__(self, config_path, checkpoint_path, device="cuda:0", score_threshold=0.2, max_dets=50):
        from mmdet.apis import init_detector
        from mmengine.config import Config
        from mmengine.dataset import Compose

        cfg = Config.fromfile(config_path)
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(config_path))[0])
        cfg.load_from = checkpoint_path
        self.model = init_detector(cfg, checkpoint=checkpoint_path, device=device)

        pipeline = self.model.cfg.test_dataloader.dataset.pipeline
        pipeline[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = Compose(pipeline)

        self.score_threshold = score_threshold
        self.max_dets = max_dets
        self.texts = None

    def set_vocabulary(self, names):
        self.texts = [[n.strip()] for n in names] + [[" "]]
        self.model.reparameterize(self.texts)

    def detect(self, image):
        data = self.pipeline(dict(img_id=0, img=image, texts=self.texts))
        batch = dict(inputs=data["inputs"].unsqueeze(0), data_samples=[data["data_samples"]])
        with torch.no_grad():
            pred = self.model.test_step(batch)[0].pred_instances

        pred = pred[pred.scores.float() > self.score_threshold]
        if len(pred.scores) > self.max_dets:
            pred = pred[pred.scores.float().topk(self.max_dets)[1]]
        pred = pred.cpu().numpy()

        return [
            (self.texts[label][0], float(score), tuple(float(v) for v in box))
            for box, label, score in zip(pred["bboxes"], pred["labels"], pred["scores"])
        ]
