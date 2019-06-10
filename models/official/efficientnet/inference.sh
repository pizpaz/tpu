#!/bin/bash
source ~/user/blackrussian/c3/source.me
export HADOOP_USER_NAME=blackrussian

TRAIN_OUTPUT_DIR="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/models_to_blend_triain"
TRAIN_IMAGE_PATH="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/train_set"
TRAIN_SET="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/train_labels.csv"

#VAL_OUTPUT_DIR="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/models_to_blend"
VAL_OUTPUT_DIR="./tt1"
VAL_IMAGE_PATH="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/val_set"
VAL_SET="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/val_labels.csv"

TEST_OUTPUT_DIR="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/models_to_blend_test"
TEST_IMAGE_PATH="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/test_set"
TEST_SET="/home1/irteam/user/blackrussian/wagon/data_collection/raw/iFood2019/test_info.csv"


#LOOPS="/home1/irteam/user/blackrussian/wagon/research_models/tpu/models/official/efficientnet/train_model/iFood2019-b0-m3/archive/model.ckpt-99954@efficientnet-b0@224"
LOOPS="/home1/irteam/user/blackrussian/wagon/research_models/tpu/models/official/efficientnet/train_model_tpu/iFood-TPU-b3-m2/model.ckpt-10164@efficientnet-b3@300"

for LOOP in $LOOPS
do
  echo "=== ${LOOP}"
  CKPT_FILE=`echo ${LOOP} | cut -d '@' -f 1`
  MODEL_NAME=`echo ${LOOP} | cut -d '@' -f 2`
  IMAGE_SIZE=`echo ${LOOP} | cut -d '@' -f 3`

  #CKPT_FILE="hdfs://c3/user/blackrussian/models/iFood2019//iFood2019-mobilenetv1-m43/model.ckpt-16667"
  echo "CKPT=${CKPT_FILE}"


  echo "=== VALIDATION"
  python ./inference.py \
    --data_format=channels_last \
    --checkpoint_file=${CKPT_FILE} \
    --model_name=${MODEL_NAME} \
    --is_test=False \
    --image_path=${VAL_IMAGE_PATH} \
    --output_dir=${VAL_OUTPUT_DIR} \
    --val_set=${VAL_SET} \
    --labels_file='/home1/irteam/user/blackrussian/wagon/data_collection/tfrecords/iFood2019/train_labels.txt' \
    --data_name='iFood2019' \
    --image_size=${IMAGE_SIZE} \
    --gpu_to_use=7

  echo "=== TEST"
  python ./inference.py \
    --data_format=channels_last \
    --checkpoint_file=${CKPT_FILE} \
    --model_name=${MODEL_NAME} \
    --hub_url=${HUB_URL} \
    --is_test=True \
    --image_path=${TEST_IMAGE_PATH} \
    --output_dir=${TEST_OUTPUT_DIR} \
    --val_set=${TEST_SET} \
    --labels_file='/home1/irteam/user/blackrussian/wagon/data_collection/tfrecords/iFood2019/train_labels.txt' \
    --data_name='iFood2019' \
    --image_size=${IMAGE_SIZE} \
    --gpu_to_use=7
done
