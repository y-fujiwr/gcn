#!/bin/bash
for var in {0..37}
do
	echo ${var}th
#	python train.py -dataset=googlejam -class_num=11 -model_name=${var}th -hidden1=128 -dropout=0.1
	python predict.py -dataset=googlejam -class_num=11 -model_name=${var}th -hidden1=128 -dropout=0.1 -mode=test
done
