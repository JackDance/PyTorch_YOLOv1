#!/bin/bash

# train
python train.py --cuda -ms --batch_size 5 --tfboard

# test.py to visualize detection images
python test.py --cuda -d voc --trained_model weights/voc/yolo/yolo_110.pth -size 608

# eval.py to calculate mAP
python eval.py --cuda -d voc --trained_model weights/voc/yolo/yolo_110.pth -size 608