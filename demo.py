# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:13:21 2019

@author: Zonsor
"""

import cv2
# You may need to restart your runtime prior to this, to let your installation take effect
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor  # make plt.show() wrong
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

im = cv2.imread("input.jpg")

cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the
# https://dl.fbaipublicfiles.... url, or use the following shorthand
cfg.MODEL.WEIGHTS = "../model_final_280758.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite('out.jpg', v.get_image()[:, :, ::-1])
