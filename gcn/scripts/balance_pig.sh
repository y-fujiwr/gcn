#!/bin/bash
for var in {0..9}
do
	echo ${var}
	python train.py -dataset=pig_dataset -model_name=node_${var}th -input_dim=85 -learning_type=node -hidden1=128 -class_num=21
	python predict.py -dataset=pig_dataset -model_name=node_${var}th -mode=test -input_dim=85 -hidden1=128 -learning_type=node -class_num=21
done
for var in {0..9}
do
	echo ${var}
	python train.py -dataset=pig_dataset -model_name=method_${var}th -input_dim=85 -learning_type=method -hidden1=128 -class_num=21
	python predict.py -dataset=pig_dataset -model_name=method_${var}th -mode=test -input_dim=85 -hidden1=128 -class_num=21
done


