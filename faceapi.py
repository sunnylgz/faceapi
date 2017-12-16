#! /usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import base64
import facenet
import align.detect_face

__debug = True
def face_compare(image1,
                 image2,
                 options=None):

  image1 = base64.b64decode(image1)
  image2 = base64.b64decode(image2)

  if __debug:
    with open("image1.bin", "wb") as f:
      f.write(image1)
    with open("image2.bin", "wb") as f:
      f.write(image2)

  ret_dict = {
   'status': 0,
   'score': 0.1,
   'error': "No Error",
  }
  return ret_dict 

def face_location(image, options=None):

  image = base64.b64decode(image)

  minsize = 20 # minimum size of fac
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
  factor = 0.709 # scale factor
  with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

  bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

  return bounding_boxes

def main(args):
  with open("base64.txt", "r") as f:
    test_data = f.readline()
  #test_data = test_data.encode(encoding='utf-8')
  #print(test_data)
  ret_dict = face_compare(test_data, test_data)
  print("status: ", ret_dict["status"])
  print("error: ", ret_dict["error"])
  print("score: ", ret_dict["score"])

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

