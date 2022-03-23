#!/bin/bash

# train
python train.py --cuda -ms --batch_size 5 --tfboard

# test
python test.py --cuda -d voc --trained_model weights/voc/yolo/yolo_110.pth -size 608
