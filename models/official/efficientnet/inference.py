# coding=utf8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import json
import sys

from absl import flags
from tqdm import tqdm

import tensorflow as tf

import efficientnet_builder
import inception_preprocessing
import preprocessing

import numpy as np

flags.DEFINE_string('data_format', 'channels_last', '')
flags.DEFINE_string('model_name', 'efficientnet-b0', '')
flags.DEFINE_string('checkpoint_file', 'r152dsk_m0', '')
flags.DEFINE_string('image_path', 'iFood2019_raw/val_set', '')
flags.DEFINE_string('val_set', 'iFood2019_raw/val_info.csv', '')
flags.DEFINE_string('labels_file', 'iFood2019_raw/train_labels.txt', '')
flags.DEFINE_string('output_dir', 'models_to_blend', '')
flags.DEFINE_integer('image_size', 224, '')
flags.DEFINE_string('gpu_to_use', '0', '')
flags.DEFINE_boolean('is_test', False, '')
FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

def build_model(features, is_training, is_logits=False):
  """Build model with input features."""
  override_params = {}
  override_params['num_classes'] = 251
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
  logits, _ = efficientnet_builder.build_model(features, FLAGS.model_name, is_training, 
                  override_params=override_params)
  if not is_logits:
    probs = tf.nn.softmax(logits)
  else:
    probs = logits
  return probs


def restore_model(sess, checkpoint):
  """Restore variables from checkpoint dir."""

  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  ema_vars = list(set(ema_vars))
  var_dict = ema.variables_to_restore(ema_vars)
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, checkpoint)


def input_fn(image_files, batch_size):
  def _parse_function(image_file):
    image_buffer = tf.read_file(image_file)
    image = preprocessing.preprocess_image(image_buffer, False, FLAGS.image_size)
    image = tf.cast(image, tf.float32)

    return image

  dataset = tf.data.Dataset.from_tensor_slices(image_files)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          _parse_function,
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  iterator = dataset.make_one_shot_iterator()
  images = iterator.get_next()

  return images


def input_fn_for_placeholder(image_files, batch_size):
  def _parse_function(image_file):
    image_buffer = tf.read_file(image_file)
    image = preprocessing.preprocess_image(image_buffer, False, FLAGS.image_size)
    image = tf.cast(image, tf.float32)

    return image

  dataset = tf.data.Dataset.from_tensor_slices(image_files)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          _parse_function,
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  return dataset


def input_fn_10crop_for_placeholder(image_files, batch_size):
  def _parse_function(image_file):
    image_buffer = tf.read_file(image_file)
    images = inception_preprocessing.preprocess_image_ten_crop(
      image_buffer=image_buffer,
      output_height=FLAGS.image_size,
      output_width=FLAGS.image_size,
      num_channels=3)

    ''' 1crop test
    image = preprocessing.preprocess_image(image_buffer, False, FLAGS.image_size)
    images = [image, image, image, image, image, image, image, image, image, image]
    '''
    images = tf.cast(images, tf.float32)

    return images

  dataset = tf.data.Dataset.from_tensor_slices(image_files)
  dataset = dataset.prefetch(buffer_size=batch_size)
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          _parse_function,
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=False))

  return dataset


def run_inference_prob(file_list, batch_size, ckpt_dir):
  #num_images = len(file_list)
  num_images = 1000
  height, width = FLAGS.image_size, FLAGS.image_size
  label2idx = {}
  with open(FLAGS.labels_file, 'r') as fp:
    for line in fp:
      idx, label = line.strip().split(':')
      label2idx[label] = int(idx)

  gt_list = []
  with open(FLAGS.val_set, 'r') as fp:
    for line in fp:
      file_name, label = line.strip().split(",")
      gt_list.append(int(label2idx[label]))

  with tf.Graph().as_default(), tf.Session() as sess:
    images = input_fn(file_list, batch_size)
    probs = build_model(images, is_training=False)

    sess.run(tf.global_variables_initializer())
    restore_model(sess, ckpt_dir)

    pred_idx = []
    pred_prob = []
    for _ in range(num_images // batch_size):
      out_probs = sess.run(probs)
      idx = np.argsort(out_probs)[::-1]
      pred_idx.append(idx[:3])
      pred_prob.append([out_probs[pid] for pid in idx[:3]])
  
  top1_cnt, top3_cnt = 0.0, 0.0
  for i, label in enumerate(gt_list[:1000]):
    top1_cnt += label in pred_idx[i][:1]
    top3_cnt += label in pred_idx[i][:3]
    if i % 100 == 0:
      print('Step {}: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(
          i, 100 * top1_cnt / (i + 1), 100 * top3_cnt / (i + 1)))
      sys.stdout.flush()
  top1, top3 = 100 * top1_cnt / num_images, 100 * top3_cnt / num_images
  print('Final: top1_acc = {:4.2f}%  top3_acc = {:4.2f}%'.format(top1, top3))


def run_inference_topk(file_list, batch_size, ckpt_dir):
  num_images = len(file_list)
  height, width = FLAGS.image_size, FLAGS.image_size
  label2idx = {}
  with open(FLAGS.labels_file, 'r') as fp:
    for line in fp:
      idx, label = line.strip().split(':')
      label2idx[label] = int(idx)

  gt_list = []
  with open(FLAGS.val_set, 'r') as fp:
    for line in fp:
      file_name, label = line.strip().split(",")
      gt_list.append(int(label2idx[label]))

  with tf.Graph().as_default(), tf.Session() as sess:
    images = input_fn(file_list, batch_size)
    probs = build_model(images, is_training=False)
    sm_topk = tf.nn.top_k(probs, 3)
    print('@PROBS = {}'.format(probs))

    sess.run(tf.global_variables_initializer())
    restore_model(sess, ckpt_dir)

    t_idx=0
    correct=0.
    correct_topk=0.
    for _ in range(num_images // batch_size):
      result = sess.run(sm_topk)
      for i in range(len(result.indices)):
        label_idx = result.indices[i][0]
        label_idx_topk = map(int, result.indices[i])
        confidence = result.values[i][0]
        
        if int(label_idx) == gt_list[t_idx]:
          correct += 1.
        if gt_list[t_idx] in label_idx_topk:
          correct_topk += 1.

        t_idx += 1
  
  print('top1 acc = {}'.format(correct / t_idx))
  print('top3 acc = {}'.format(correct_topk / t_idx))


def run_inference_topk_10crop_placeholder(file_list, filename_list, batch_size, ckpt_dir):
  if tf.gfile.IsDirectory(ckpt_dir):
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  else:
    checkpoint = ckpt_dir

  num_images = len(file_list)
  height, width = FLAGS.image_size, FLAGS.image_size


  dataset = input_fn_10crop_for_placeholder(file_list, batch_size)

  num_crops = 10
  ten_crop_logits = []

  if not FLAGS.is_test:
    label2idx = {}
    with open(FLAGS.labels_file, 'r') as fp:
      for line in fp:
        idx, label = line.strip().split(':')
        label2idx[label] = int(idx)

    filename2idx = {}
    with open(FLAGS.val_set, 'r') as fp:
      for line in fp:
        file_name, label = line.strip().split(",")
        filename2idx[file_name] = label2idx[label]

    gt_list = []
    with open(FLAGS.val_set, 'r') as fp:
      for line in fp:
        file_name, label = line.strip().split(",")
        gt_list.append(int(label2idx[label]))

  with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(tf.float32, [None, num_crops, height, width, 3])
    images_tensors = tf.split(images, num_crops, axis=1) #[(batch_size, h, w, c)]
    for i in tqdm(range(num_crops)):
      crop_images = tf.squeeze(images_tensors[i], axis=1) # (batch, h, w, c)
      logits = build_model(crop_images, is_training=False, is_logits=True)
      ten_crop_logits.append(logits) #[10, batch, 251]


    ten_crop_logits = tf.stack(ten_crop_logits, axis=1)
    avg_logits = tf.reduce_mean(ten_crop_logits, axis=1)
    probs = tf.nn.softmax(avg_logits)
    sm_topk = tf.nn.top_k(probs, 3)

    top1 = tf.argmax(avg_logits, axis=1)

    np_logits = np.zeros((num_images, int(avg_logits.get_shape()[1])), dtype=np.float32)
    np_preds = np.zeros(num_images, dtype=np.int64)
    np_labels = np.zeros(num_images, dtype=np.int64)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess.run(tf.global_variables_initializer())
    restore_model(sess, checkpoint)

    image_i = 0
    t_idx=0
    correct=0.
    correct_topk=0.
    for _ in range(num_images // batch_size):
      images_tensors = next_element
      images_input = sess.run(images_tensors)
      emb, _top1 = sess.run([avg_logits, top1], feed_dict={images: images_input})
      '''
      for i in range(len(result.indices)):
        label_idx = result.indices[i][0]
        label_idx_topk = map(int, result.indices[i])
        confidence = result.values[i][0]

        #print('({},{}) => {}'.format(label_idx, confidence, gt_list[t_idx]))
        
        if int(label_idx) == gt_list[t_idx]:
          correct += 1.
        if gt_list[t_idx] in label_idx_topk:
          correct_topk += 1.
        t_idx += 1

      '''
      for i in tqdm(range(len(emb))):
        np_logits[image_i, :] = emb[i]
        if not FLAGS.is_test:
          np_preds[image_i] = _top1[i]
          np_labels[image_i] = filename2idx[filename_list[image_i]]
        image_i += 1

  #print('top1 acc = {}'.format(correct / t_idx))
  #print('top3 acc = {}'.format(correct_topk / t_idx))


  if not FLAGS.is_test:
    acc = np.count_nonzero(np_preds == np_labels) / num_images
    print('acc = {}'.format(acc))

  model_name = os.path.dirname(checkpoint).split("/")[-1] + '-' + checkpoint.split("/")[-1].split("-")[-1]

  print("@@MODEL_NAME = {}".format(model_name))
  out_file = os.path.join(FLAGS.output_dir, model_name)
  np.save(out_file, np_logits)


def run_inference_topk_placeholder(file_list, batch_size, ckpt_dir):
  print("@@@PLACEHOLDER")
  num_images = 1000

  dataset = input_fn_for_placeholder(file_list, batch_size)

  height, width = FLAGS.image_size, FLAGS.image_size
  label2idx = {}
  with open(FLAGS.labels_file, 'r') as fp:
    for line in fp:
      idx, label = line.strip().split(':')
      label2idx[label] = int(idx)

  gt_list = []
  with open(FLAGS.val_set, 'r') as fp:
    for line in fp:
      file_name, label = line.strip().split(",")
      gt_list.append(int(label2idx[label]))


  with tf.Graph().as_default(), tf.Session() as sess:
    images = tf.placeholder(tf.float32, [None, height, width, 3])
    probs = build_model(images, is_training=False)
    sm_topk = tf.nn.top_k(probs, 3)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess.run(tf.global_variables_initializer())
    restore_model(sess, ckpt_dir)


    t_idx=0
    correct=0.
    correct_topk=0.
    for _ in range(num_images // batch_size):
      images_tensors = next_element
      images_input = sess.run(images_tensors)
      result = sess.run(sm_topk, feed_dict={images: images_input})

      for i in range(len(result.indices)):
        label_idx = result.indices[i][0]
        label_idx_topk = map(int, result.indices[i])
        confidence = result.values[i][0]
        
        if int(label_idx) == gt_list[t_idx]:
          correct += 1.
        if gt_list[t_idx] in label_idx_topk:
          correct_topk += 1.

        t_idx += 1
  
  print('top1 acc = {}'.format(correct / t_idx))
  print('top3 acc = {}'.format(correct_topk / t_idx))


def main(unused_argv):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use

  #batch_size=128
  batch_size=64

  filename_list = []
  with open(FLAGS.val_set, 'r') as fp:
    for line in fp:
      if FLAGS.is_test:
        file_name = line.strip()
      else:
        file_name, _ = line.strip().split(",")
      filename_list.append(file_name)


  fullpath_list = []
  for filename in filename_list:
    fullpath_list.append(os.path.join(FLAGS.image_path, filename))

  num_images = len(filename_list)

  #run_inference_topk(fullpath_list, batch_size, FLAGS.checkpoint_file)
  #run_inference_topk_placeholder(fullpath_list, batch_size, FLAGS.checkpoint_file)
  run_inference_topk_10crop_placeholder(fullpath_list, filename_list, batch_size, FLAGS.checkpoint_file)

if __name__ == '__main__':
  tf.app.run()
