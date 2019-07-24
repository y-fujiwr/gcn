#!/bin/bash
for var in {0..10}
do
	echo ${var}
	python train.py -dataset=balance_node20 -model_name=${var}th
	python predict.py -dataset=balance_node20 -model_name=${var}th -mode=test
done