#!/bin/bash
for var in {0..500}
do
	echo ${var}th
	python predict.py -dataset=small_add_missing40 -model_name=${var}th -class_num=40 -mode=test
done