#! /usr/bin/python3
"""find faces from input image based on mtcnn and locate the locations and landmarks
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
from scipy import misc
import numpy as np
import cv2
import datetime,time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

MIN_INPUT_SIZE = 320
def faster_face_detect(img):
  #print(img.shape)
  h=img.shape[0]
  w=img.shape[1]
  minl=np.amin([h, w])
  print("original image is %dx%d" % (w, h))

  scale = 1
  if minl > MIN_INPUT_SIZE:
    scale = minl // MIN_INPUT_SIZE
    hs=int(np.ceil(h/scale))
    ws=int(np.ceil(w/scale))
    #im_data = imresample(img, (hs, ws))
    im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    print("scaled image is %dx%d" % (ws, hs))
  else:
    im_data = img

  gray = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)

  face_locations = face_cascade.detectMultiScale(gray, 1.3, 5)
  return face_locations, scale

def main(args):
  
  total_cpu_time = 0
  total_real_time = 0

  video = args.video or 0 # if args.video == '': open camera
  videoCapture = cv2.VideoCapture(args.video)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
          int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print("fps = {}, size = {}".format(fps, size))

  if not videoCapture.isOpened():
    sys.exit("open video failed")

  if args.out:
    videoWriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'H264'), fps, size)
  count = 0
  while True:
    success, img = videoCapture.read()
    if not success:
      break

    start_t = time.time()
    start_c = time.clock()
    
    face_locations, scale = faster_face_detect(img)

    end_t = time.time()
    end_c = time.clock()

    total_cpu_time += end_c - start_c
    total_real_time += end_t - start_t

    #print("I found {} face(s) in this photograph.".format(len(face_locations)))

    for (x,y,w,h) in face_locations:

      # Print the location of each face in this image
      left, top, right, bottom = np.array((x, y, x+w, y+h)) * scale
      cv2.rectangle(img,(left, top), (right, bottom),(255,0,0),2)

    if args.out:
      videoWriter.write(img)
    else:
      cv2.imshow("Oto Video", img)

    count = count + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  print("The inference time cost: %.2fs, fps: %.2f" % (total_real_time, count/total_real_time))
  videoCapture.release()
  cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video', type=str, default = '', help='Video to load')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


