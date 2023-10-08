#!/bin/bash

cd yolov5-copy

python detect.py --weights=yolov5s.pt --source=../workspace/car.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../workspace/
