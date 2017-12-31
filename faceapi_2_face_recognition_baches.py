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
import face_recognition
import align.detect_face
from scipy import misc
from scipy import stats
import tensorflow as tf
from PIL import Image

__debug = True # = __debug__

if __debug:
  import datetime,time

minsize = 20 # minimum size of fac
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
margin = 44 # Margin for the crop around the bounding box (height, width) in pixels.
image_size = 160 # Image size (height, width) of cropped face in pixels.
facenet_model = "/home/ubuntu/share/source_code/facenet/20170512-110547/20170512-110547.pb"
c_normal_mean_stddev = [0.7, 0.2]

def crop_face(imgs):
  batch_bounding_boxes = face_recognition.batch_face_locations(imgs, number_of_times_to_upsample=0, batch_size=2)
  bbs = []
  prewhiteneds = []
  for frame_number_in_batch, bounding_boxes in enumerate(batch_bounding_boxes):
	  img = imgs[frame_number_in_batch]
	  img_size = np.asarray(img.shape)[0:2]
	  if len(bounding_boxes) > 1:
	    raise RuntimeError("detected multi faces")
	  if len(bounding_boxes) < 1:
	    raise RuntimeError("no faces detected")
	  det = np.squeeze(bounding_boxes[0])
	  bb = np.zeros(4, dtype=np.int32)
	  bb[0] = np.maximum(det[3]-margin/2, 0)
	  bb[1] = np.maximum(det[0]-margin/2, 0)
	  bb[2] = np.minimum(det[1]+margin/2, img_size[1])
	  bb[3] = np.minimum(det[2]+margin/2, img_size[0])
	  cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
	  aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
	  pil_image = Image.fromarray(aligned)
	  pil_image.show()
	  prewhitened = facenet.prewhiten(aligned)
	  prewhiteneds.append(prewhitened)
	  bbs.append(bb)

  return prewhiteneds,bbs

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

  if __debug:
    end_t = time.time()
    end_c = time.clock()

    elapsed_real_time = end_t - start_t
    elapsed_user_time = end_c - start_c
    print("create mtcnn cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
    start_t,start_c = end_t,end_c

  # detect face
  try:
	  images,bounding_boxes = crop_face([img1,img2])
  except RuntimeError as e:
    ret_dict['error'] = str(e) + "while cropping faces"
    ret_dict['status'] = 1
    return ret_dict


  if __debug:
    end_t = time.time()
    end_c = time.clock()

    elapsed_real_time = end_t - start_t
    elapsed_user_time = end_c - start_c
    print("align faces cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
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
      feed_dict = { images_placeholder: images, phase_train_placeholder:False }
      emb = sess.run(embeddings, feed_dict=feed_dict)

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

  image = base64.b64decode(image)

  with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

  bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

  return bounding_boxes

def main(args):

  if len(args.input_files) < 2:
    print("Must provide at least 2 files to test compare()")
    return
  with open(args.input_files[0], "r") as f:
    test_data1 = f.readline()
  with open(args.input_files[1], "r") as f:
    test_data2 = f.readline()
  #test_data = test_data.encode(encoding='utf-8')
  #print(test_data)
  ret_dict = face_compare(test_data1, test_data2)
  print(ret_dict)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_files', type=str, nargs='+', help='Input files (coded as base64, and raw is jpg/png')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--debug', action='store_true', help='Run in debugging mode') # TODO: not used it. Use global variable __debug instead
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

