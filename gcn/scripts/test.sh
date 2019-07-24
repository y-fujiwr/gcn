#!/bin/bash
for var in 144 151 209 251 309 391 392
do
	echo ${var}th
	python predict.py -dataset=small_add_missing40 -model_name=${var}th -mode=test -class_num=40
done
#特にaccuracyが高いモデルに対して適合率を算出するスクリプト