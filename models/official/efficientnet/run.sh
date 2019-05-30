
: '
visible_devices="0,1"
num_gpus=`echo "${visible_devices}" | awk -F "," '{print NF}'`
python main.py --use_tpu=False --data_dir=/home2/irteam/user/blackrussian/wagon/data_collection/tfrecords/imagenet --model_dir=./m2 --train_batch_size=32 --model_name=efficientnet-b4 --eval_batch_size=32 --moving_average_decay=0.0
'

source ~/user/blackrussian/c3/source.me
export HADOOP_USER_NAME=blackrussian


##iFOOD2019
DATA_DIR=/home1/irteam/user/blackrussian/wagon/data_collection/tfrecords/iFood2019
NUM_LABEL_CLASSES=251
NUM_TRAIN_IMAGES=118475
NUM_EVAL_IMAGES=11994
## 1epoch => 1,851 step when batch_size=64
## 1epoch => 462 step when batch_size=256

visible_devices="0,1,2,3,4,5,6,7"
num_gpus=`echo "${visible_devices}" | awk -F "," '{print NF}'`
export CUDA_VISIBLE_DEVICES=${visible_devices}
MODEL_DIR=hdfs://c3/user/blackrussian/models/EfficientNet/iFood2019-m1
BATCH_SIZE=256
MODEL_NAME=efficientnet-b4
TRAIN_STEPS=46200

# base learning rate 0.016 when batch size is 256.
python main.py \
	--use_tpu=False \
  --base_learning_rate=0.016 \
  --num_gpus=${num_gpus} \
	--data_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--model_name=${MODEL_NAME} \
	--train_batch_size=${BATCH_SIZE} \
	--num_train_images=${NUM_TRAIN_IMAGES} \
	--num_eval_images=${NUM_EVAL_IMAGES} \
	--num_label_classes=${NUM_LABEL_CLASSES} \
	--train_steps=${TRAIN_STEPS} \
	--moving_average_decay=0.0
