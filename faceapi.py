#! /usr/bin/env python3

# TODO:
# 20171217, multi exception catch. multi exit() from function face_compare

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import io
import base64
import numpy as np
import facenet
import align.detect_face
from scipy import misc
from scipy import stats
import tensorflow as tf
from model_sel import facenet_model,c_normal_mean_stddev

__debug = True # = __debug__

if __debug:
  import datetime,time

minsize = 20 # minimum size of fac
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
margin = 44 # Margin for the crop around the bounding box (height, width) in pixels.
image_size = 160 # Image size (height, width) of cropped face in pixels.

if __debug:
  start_t = time.time()
  start_c = time.clock()

with tf.Graph().as_default():
  sess = tf.Session()
  with sess.as_default():
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

if __debug:
  end_t = time.time()
  end_c = time.clock()

  elapsed_real_time = end_t - start_t
  elapsed_user_time = end_c - start_c
  print("create mtcnn cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
  start_t,start_c = end_t,end_c

with tf.Graph().as_default():
  with tf.Session() as sess:
    # Load the model
    facenet.load_model(facenet_model)

    if __debug:
      end_t = time.time()
      end_c = time.clock()

      elapsed_real_time = end_t - start_t
      elapsed_user_time = end_c - start_c
      print("load face model cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
      start_t,start_c = end_t,end_c

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    emb_fun = lambda images : sess.run(embeddings, feed_dict={ images_placeholder: images, phase_train_placeholder:False })

MIN_INPUT_SIZE = 80
def faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor):
  #print(img.shape)
  h=img.shape[0]
  w=img.shape[1]
  minl=np.amin([h, w])
  #print("original image is %dx%d" % (w, h))

  scale = 1
  if minl > MIN_INPUT_SIZE:
    scale = minl // MIN_INPUT_SIZE
    hs=int(np.ceil(h/scale))
    ws=int(np.ceil(w/scale))
    #im_data = imresample(img, (hs, ws))
    im_data = misc.imresize(img, (hs, ws), interp='bilinear')
    #print("scaled image is %dx%d" % (ws, hs))
  else:
    im_data = img

  face_locations, points = align.detect_face.detect_face(im_data, minsize, pnet, rnet, onet, threshold, factor)
  #for face_location in face_locations:
  #  face_location[0:4] = face_location[0:4] * scale

  return face_locations, points, scale

def crop_face(img, pnet, rnet, onet):
  img_size = np.asarray(img.shape)[0:2]
  bounding_boxes, _, scale = faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor)
  if len(bounding_boxes) > 1:
    raise RuntimeError("detected multi faces")
  if len(bounding_boxes) < 1:
    raise RuntimeError("no faces detected")
  det = np.squeeze(bounding_boxes[0,0:4]) * scale
  bb = np.zeros(4, dtype=np.int32)
  bb[0] = np.maximum(det[0]-margin/2, 0)
  bb[1] = np.maximum(det[1]-margin/2, 0)
  bb[2] = np.minimum(det[2]+margin/2, img_size[1])
  bb[3] = np.minimum(det[3]+margin/2, img_size[0])
  cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
  aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
  prewhitened = facenet.prewhiten(aligned)

  return prewhitened,bb

def face_compare(image1,
                 image2,
                 options=None):

  ret_dict = {
   'status': 0,
   'score': 0.0,
   'error': 'No Error',
   'bounding_box': [[0,0,0,0],[0,0,0,0]]
  }

  image1 = base64.b64decode(image1)
  image2 = base64.b64decode(image2)
  if __debug:
    with open("image1.bin", "wb") as f:
      f.write(image1)
    with open("image2.bin", "wb") as f:
      f.write(image2)

  #image1 = tf.image.decode_image(image1, channels=3)
  #image2 = tf.image.decode_image(image2, channels=3)
  #img = misc.imread(os.path.expanduser("1.jpg"), mode='RGB')
  # TODO: need to add except handling for io.BytesIO()
  try:
    img1 = misc.imread(io.BytesIO(image1), mode='RGB')
    img2 = misc.imread(io.BytesIO(image2), mode='RGB')
  except OSError:
    ret_dict['error'] = 'IO Error'
    ret_dict['status'] = 1
    return ret_dict

  if __debug:
    start_t = time.time()
    start_c = time.clock()

  # detect face
  images = []
  bounding_boxes = []
  try:
    prewhitened1,bb1 = crop_face(img1, pnet, rnet, onet)
    images.append(prewhitened1)
    bounding_boxes.append(bb1)
  except RuntimeError as e:
    ret_dict['error'] = str(e) + "while parsing image 1"
    ret_dict['status'] = 1
    return ret_dict

  try:
    prewhitened2,bb2 = crop_face(img2, pnet, rnet, onet)
    images.append(prewhitened2)
    bounding_boxes.append(bb2)
  except RuntimeError as e:
    ret_dict['error'] = str(e) + "while parsing image 2"
    ret_dict['status'] = 1
    return ret_dict

  if __debug:
    end_t = time.time()
    end_c = time.clock()

    elapsed_real_time = end_t - start_t
    elapsed_user_time = end_c - start_c
    print("align faces cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
    start_t,start_c = end_t,end_c

  emb = emb_fun(images)

  if __debug:
    end_t = time.time()
    end_c = time.clock()

    elapsed_real_time = end_t - start_t
    elapsed_user_time = end_c - start_c
    print("calculate face features cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
    start_t,start_c = end_t,end_c

  dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
  if __debug:
    print("the distance between 2 input is ", dist)

  ret_dict['score'] = 1.0 - stats.norm(c_normal_mean_stddev[0], c_normal_mean_stddev[1]).cdf(dist)
  ret_dict['bounding_box']=bounding_boxes
  return ret_dict 

def face_location(image, options=None):

  raw_image = base64.b64decode(image)
  try:
    img = misc.imread(io.BytesIO(raw_image), mode='RGB')
  except OSError:
    print("Error: face_location() can't read image")
    return ()

  bounding_boxes, _, scale = faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor)
  for bounding_box in bounding_boxes:
    bounding_box[0:4] = bounding_box[0:4] * scale

  return bounding_boxes

def main(args):

  if len(args.input_files) < 2:
    print("Must provide at least 2 files to test compare()")
    return

  if args.image:
    with open(args.image, "r") as f:
      image = f.readline()

    bounding_boxes = face_location(image)
    print("finding faces on %s, " % (args.image), bounding_boxes)

  i = 0
  while i+1 < len(args.input_files):
    print("*" * 20)
    print("compare ", args.input_files[i], args.input_files[i+1])
    with open(args.input_files[i], "r") as f:
      test_data1 = f.readline()
    with open(args.input_files[i+1], "r") as f:
      test_data2 = f.readline()
    #test_data = test_data.encode(encoding='utf-8')
    #print(test_data)
    ret_dict = face_compare(test_data1, test_data2)
    print(ret_dict)
    i += 2
    print("*" * 20)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_files', type=str, nargs='+', help='Input files (coded as base64, and raw is jpg/png')
    parser.add_argument('--image', type=str, help='Test face_location on this file (coded as base64, and raw is jpg/png')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--debug', action='store_true', help='Run in debugging mode') # TODO: not used it. Use global variable __debug instead
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

