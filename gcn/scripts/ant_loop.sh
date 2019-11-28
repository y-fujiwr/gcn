#!/bin/bash
for var in {0..100}
do
	echo ${var}
	python train.py -dataset=ant_dataset -epochs=2500 -model_name=ant_rein_${var}th -input_dim=85 -learning_type=reinforcement -hidden1=128
	python predict.py -dataset=ant_dataset -model_name=ant_rein_${var}th -mode=test -input_dim=85 -hidden1=128
	python predict.py -dataset=ant_dataset -model_name=ant_rein_${var}th -mode=val -input_dim=85 -hidden1=128
done