#!/bin/bash
rm ./data/pig_dataset/reinforcement_dataset.pkl
rm ./data/pig_dataset/addition.pkl
for var in {1..100}
do
	echo ${var}
	python train.py -dataset=pig_dataset -model_name=maven_rein_${var}th -input_dim=85 -learning_type=reinforcement -hidden1=128 -class_num=21
	python predict.py -dataset=pig_dataset -model_name=rein_${var}th -mode=test -input_dim=85 -hidden1=128 -learning_type=reinforcement -class_num=21
	python predict.py -dataset=pig_dataset -model_name=rein_${var}th -mode=val -input_dim=85 -hidden1=128 -learning_type=reinforcement -class_num=21
done