#!/bin/bash
for var in {0..9}
do
	echo ${var}
	python train.py -dataset=maven_dataset -model_name=node_${var}th -input_dim=85 -learning_type=node -class_num=24 -hidden1=128
	python predict.py -dataset=maven_dataset -model_name=node_${var}th -mode=test -input_dim=85 -class_num=24 -hidden1=128 -learning_type=node
done
for var in {0..9}
do
	echo ${var}
	python train.py -dataset=maven_dataset -model_name=method_${var}th -input_dim=85 -learning_type=method -class_num=24 -hidden1=128
	python predict.py -dataset=maven_dataset -model_name=method_${var}th -mode=test -input_dim=85 -class_num=24 -hidden1=128 -learning_type=method
done

for var in {0..9}
do
	echo ${var}
	python train.py -dataset=ant_dataset -model_name=node_${var}th -input_dim=85 -learning_type=node -hidden1=128
	python predict.py -dataset=ant_dataset -model_name=node_${var}th -mode=test -input_dim=85 -hidden1=128 -learning_type=node
done
for var in {0..9}
do
	echo ${var}
	python train.py -dataset=ant_dataset -model_name=method_${var}th -input_dim=85 -learning_type=method -hidden1=128
	python predict.py -dataset=ant_dataset -model_name=method_${var}th -mode=test -input_dim=85 -hidden1=128 -learning_type=method
done
