#!/bin/bash
rm ./data/bcb_dataset/reinforcement_dataset.pkl
rm ./data/bcb_dataset/addition.pkl
for var in {1..100}
do
	python train.py -dataset=bcb -model_name=bcb_rein_${var}th -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
	python predict.py -dataset=bcb -model_name=bcb_rein_${var}th -mode=test -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
	python predict.py -dataset=bcb -model_name=bcb_rein_${var}th -mode=val -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
done
