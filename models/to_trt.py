#!/usr/bin/env python3
# python3 to_trt.py yolo12n

import sys
import os
import logging
import coloredlogs
import argparse

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level=logging.INFO)

def convert_model_to_trt(model_name, batch_size):
    trt_model = model_name+".engine"
    if os.path.exists(trt_model):
        logging.info('{} already exists.'.format(trt_model))
    else:
        logging.warning('Exporting onnx model to pytorch...')
        model = YOLO(model_name+".pt")
        logging.warning('Exporting pytorch model to tensorrt...')
        model.export(format="engine", batch=batch_size)
        logging.warning('Converted and saved.')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--batch_size', type=int)
args = parser.parse_args() 
model_name = args.model_name
batch_size = args.batch_size
if model_name is None and batch_size is None:
     logging.warning(f'Set model_name and batch_size.')
     exit()
try:
    logging.warning(f'Going to download {model_name} and convert it with batch size {batch_size}.')
    convert_model_to_trt(model_name, int(batch_size))
except IndexError as e:
    logging.error("{}".format(e))
    exit()