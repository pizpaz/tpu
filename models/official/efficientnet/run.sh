
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

visible_devices="0"
num_gpus=`echo "${visible_devices}" | awk -F "," '{print NF}'`
export CUDA_VISIBLE_DEVICES=${visible_devices}

MODEL_NAME=efficientnet-b0
MODEL_DIR=./train_model/iFood2019-b0-m1
mkdir -p ${MODEL_DIR}
BATCH_SIZE=32
LR=0.002 # base learning rate 0.016 when batch size is 256.

#BATCH_SIZE=256, 100epoch
#TRAIN_STEPS=46200
#STEPS_PER_EVAL=462

#BATCH_SIZE=32, 100epoch
TRAIN_STEPS=370200
STEPS_PER_EVAL=3702

PCKPT=/home1/irteam/user/blackrussian/wagon/research_models/tpu/models/official/efficientnet/weight/efficientnet-b0

python main.py \
  --transpose_input=False \
	--use_tpu=False \
  --base_learning_rate=${LR} \
  --num_gpus=${num_gpus} \
	--data_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--model_name=${MODEL_NAME} \
	--train_batch_size=${BATCH_SIZE} \
	--num_train_images=${NUM_TRAIN_IMAGES} \
	--num_eval_images=${NUM_EVAL_IMAGES} \
	--num_label_classes=${NUM_LABEL_CLASSES} \
	--train_steps=${TRAIN_STEPS} \
	--steps_per_eval=${STEPS_PER_EVAL} \
  --pretrained_model_checkpoint_path=${PCKPT} \
	--moving_average_decay=0.0

: '
visible_devices="0"
num_gpus=`echo "${visible_devices}" | awk -F "," '{print NF}'`
export CUDA_VISIBLE_DEVICES=${visible_devices}

MODEL_NAME=efficientnet-b3
#MODEL_DIR=./train_model/iFood2019-b3-m2
MODEL_DIR=./www5-onegpu
mkdir -p ${MODEL_DIR}
#BATCH_SIZE=256
BATCH_SIZE=32
LR=0.016 # base learning rate 0.016 when batch size is 256.
TRAIN_STEPS=46200
STEPS_PER_EVAL=462
PCKPT=/home1/irteam/user/blackrussian/wagon/research_models/tpu/models/official/efficientnet/weight/efficientnet-b3

python main.py \
  --transpose_input=False \
	--use_tpu=False \
  --base_learning_rate=${LR} \
  --num_gpus=${num_gpus} \
	--data_dir=${DATA_DIR} \
	--model_dir=${MODEL_DIR} \
	--model_name=${MODEL_NAME} \
	--train_batch_size=${BATCH_SIZE} \
	--num_train_images=${NUM_TRAIN_IMAGES} \
	--num_eval_images=${NUM_EVAL_IMAGES} \
	--num_label_classes=${NUM_LABEL_CLASSES} \
	--train_steps=${TRAIN_STEPS} \
	--steps_per_eval=${STEPS_PER_EVAL} \
  --pretrained_model_checkpoint_path=${PCKPT} \
	--moving_average_decay=0.0
'

: '
visible_devices="0,1,2,3,4,5,6,7"
num_gpus=`echo "${visible_devices}" | awk -F "," '{print NF}'`
export CUDA_VISIBLE_DEVICES=${visible_devices}

MODEL_NAME=efficientnet-b3
#MODEL_DIR=hdfs://c3/user/blackrussian/models/EfficientNet/iFood2019-b3-m1
MODEL_DIR=./train_model/iFood2019-b3-m1
mkdir -p ${MODEL_DIR}
BATCH_SIZE=256
LR=0.016 # base learning rate 0.016 when batch size is 256.
TRAIN_STEPS=46200
PCKPT=/home1/irteam/user/blackrussian/wagon/research_models/tpu/models/official/efficientnet/weight/efficientnet-b3

python main.py \
	--use_tpu=False \
  --base_learning_rate=${LR} \
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
  #--pretrained_model_checkpoint_path=${PCKPT} \
'
