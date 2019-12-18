# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:40:34 2019

@author: Zonsor
"""
import os
import cv2
import json
import random
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from itertools import groupby
from pycocotools import mask as maskutil


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle


def show_example(dataset_dicts, metadata, num=3):
    for d in random.sample(dataset_dicts, num):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


setup_logger()
train_path = "train_images/"
json_file = os.path.join(train_path, "pascal_train.json")
# convert COCO format to Detectron2 format
register_coco_instances("VOC_dataset", {}, json_file, train_path)
dataset_dicts = load_coco_json(json_file, train_path, "VOC_dataset")

VOC_metadata = MetadataCatalog.get("VOC_dataset")
show_example(dataset_dicts, VOC_metadata, 3)

os.makedirs('train_results', exist_ok=True)
# ============ train ===========
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("VOC_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = "X-101-32x8d.pkl"  # pre-trained model file location
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 200000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# ============= training results ===========
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
cfg.DATASETS.TEST = ("VOC_dataset", )
predictor = DefaultPredictor(cfg)


json_file = os.path.join(train_path, "pascal_train.json")
cocoGt = COCO(json_file)

coco_dt_train = []
for imgid in cocoGt.imgs:
    filename = cocoGt.loadImgs(ids=imgid)[0]['file_name']
    print('predicting ' + filename)
    im = cv2.imread(train_path + filename)  # load image
    outputs = predictor(im)  # run inference of your model

    output_path = os.path.join('train_results', filename)
    v = Visualizer(im[:, :, ::-1],
                   metadata=VOC_metadata,
                   scale=3,
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, v.get_image()[:, :, ::-1])

    anno = outputs["instances"].to("cpu").get_fields()
    masks = anno['pred_masks'].numpy()
    categories = anno['pred_classes'].numpy()
    scores = anno['scores'].numpy()

    n_instances = len(scores)
    if len(categories) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid  # this imgid must be same as the key of test.json
            pred['category_id'] = int(categories[i]) + 1
            # save binary mask to RLE, e.g. 512x512 -> rle
            pred['segmentation'] = binary_mask_to_rle(masks[i, :, :])
            pred['score'] = float(scores[i])
            coco_dt_train.append(pred)

with open("submission_train.json", "w") as f:
    json.dump(coco_dt_train, f)

cocoDt = cocoGt.loadRes("submission_train.json")

imgIds = sorted(cocoGt.getImgIds())

cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
