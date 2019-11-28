#!/bin/bash
rm ./data/maven_dataset/reinforcement_dataset.pkl
rm ./data/maven_dataset/addition.pkl
for var in {1..100}
do
	echo ${var}
	python train.py -dataset=maven_dataset -epochs=2500 -model_name=maven_rein_${var}th -input_dim=85 -learning_type=reinforcement -class_num=24 -hidden1=128
	python predict.py -dataset=maven_dataset -model_name=maven_rein_${var}th -mode=test -input_dim=85 -class_num=24 -hidden1=128 -learning_type=reinforcement
	python predict.py -dataset=maven_dataset -model_name=maven_rein_${var}th -mode=val -input_dim=85 -class_num=24 -hidden1=128 -learning_type=reinforcement
done