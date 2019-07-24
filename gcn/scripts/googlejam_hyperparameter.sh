#!/bin/bash
for var in 2 4 8 16 32
do
	echo layer = ${var}
	python train.py -layers=${var} -dataset=googlejam -class_num=11 -model_name=layer${var}
	python predict.py -layers=${var} -dataset=googlejam -class_num=11 -model_name=layer${var} -mode=test
done
for var in 16 32 64 128 256 512
do
	echo hidden1 = ${var}
	python train.py -hidden1=${var} -dataset=googlejam -class_num=11 -model_name=hidden${var}
	python predict.py -hidden1=${var} -dataset=googlejam -class_num=11 -model_name=hidden${var} -mode=test
done
for var in {1..9}
do
	echo dropout = 0.${var}
	python train.py -dropout=0.${var} -dataset=googlejam -class_num=11 -model_name=dropout${var}
	python predict.py -dropout=0.${var} -dataset=googlejam -class_num=11 -model_name=dropout${var} -mode=test
done