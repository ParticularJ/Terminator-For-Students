from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import os
import time


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  with tf.Session() as sess:

    result = sess.run(normalized)


  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  start_time = time.time()
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 331
  input_width = 331
  input_mean = 0
  input_std = 255
  # input_layer = "input"
  # output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()

  parser.add_argument("--image", default='/var/Data/xz/butterfly/data_augmentation_13_test', help="image to be processed")
  parser.add_argument("--graph", default='/var/Data/xz/butterfly/trained_models/pnasnet/output_graph.pb', help="graph/model to be executed")
  parser.add_argument("--labels", default='/var/Data/xz/butterfly/trained_models/pnasnet/output_labels.txt', help="name of file containing labels")

  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", default='Placeholder:0', help="name of input layer")
  parser.add_argument("--output_layer", default='final_result:0', help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer



  labels = load_labels(label_file)

  # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
  # for tensor_name in tensor_name_list:
  #     print(tensor_name, '\n')

  images = []
  true_y = []
  for data in os.listdir(file_name):
    true_y.append(labels.index(data[:11].lower()))
    t = read_tensor_from_image_file(
      os.path.join(file_name, data),
      input_height=input_height,
      input_width=input_width,
      input_mean=input_mean,
      input_std=input_std)
    images.append(t)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer

  graph = load_graph(model_file)
  output_tensor = graph.get_tensor_by_name(output_name)
  results_list = []
  logits = []
  count = 0



  with tf.Session(graph=graph) as sess:
    for image in images:
      start_time = time.time()
      results = sess.run(output_tensor, {
        input_name: image
      })
      print('num: {0}, using {1}.'.format(count, time.time()-start_time))
      count += 1
      results = np.squeeze(results)
      results_list.append(results)

  for predictions in results_list:
    top_k = predictions.argsort()[-1:][::-1]
    logits.append(top_k[0])
  print(time.time()-start_time)

  print(len(true_y))
  count = 0
  for i in range(len(true_y)):
    if true_y[i] != logits[i]:
      count += 1
  print(count)

