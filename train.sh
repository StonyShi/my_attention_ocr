




cd python

python gen_run.py -t 15 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out
python gen_run.py -t 15 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out
python gen_run.py -t 5 -fs 28 -new_h 32 -new_w 320 -w 2 -c 20000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out2
python gen_run.py -t 5 -fs 28 -new_h 32 -new_w 320 -w 2 -c 20000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out3



python gen_record.py --dataset_name=train --dataset_dir=out --dataset_nums=10000 --output_dir=datasets/train

python gen_record.py --dataset_name=test --dataset_dir=out2 --dataset_nums=20000 --output_dir=datasets/test


python gen_record.py --dataset_name=validation --dataset_dir=out3 --dataset_nums=20000 --output_dir=datasets/validation


python train.py --checkpoint_inception=./resource/inception_v3.ckpt --dataset_name=my_data


