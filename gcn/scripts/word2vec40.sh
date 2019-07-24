#!/bin/bash
for var in {0..100}
do
	echo ${var}th
	python train.py -dataset=word2vec40 -model_name=${var}th -class_num=40
	python predict.py -dataset=word2vec40 -model_name=${var}th -class_num=40 -mode=val
done