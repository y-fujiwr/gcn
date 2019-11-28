#!/bin/bash
#rm ./data/bcb_dataset/reinforcement_dataset.pkl
#rm ./data/bcb_dataset/addition.pkl
#python train.py -dataset=bcb -model_name=default -input_dim=85 -hidden1=128 -class_num=43
python predict.py -dataset=bcb -model_name=default -mode=test -input_dim=85 -hidden1=128 -class_num=43

#python train.py -dataset=bcb -model_name=method -input_dim=85 -hidden1=128 -class_num=43 -learning_type=method
python predict.py -dataset=bcb -model_name=method -mode=test -input_dim=85 -hidden1=128 -learning_type=method -class_num=43

#python train.py -dataset=bcb -model_name=node -input_dim=85 -hidden1=128 -class_num=43 -learning_type=node
python predict.py -dataset=bcb -model_name=node -mode=test -input_dim=85 -hidden1=128 -learning_type=node -class_num=43
for var in {1..100}
do
	#python train.py -dataset=bcb -model_name=bcb_rein_${var}th -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
	python predict.py -dataset=bcb -model_name=bcb_rein_${var}th -mode=test -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
	#python predict.py -dataset=bcb -model_name=bcb_rein_${var}th -mode=val -input_dim=85 -hidden1=128 -class_num=43 -learning_type=reinforcement
done
