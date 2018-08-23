




cd python

#生成固定文本
python gen_address.py
python gen_ids.py
python gen_name.py
python gen_plate.py

ls *_list.txt

cat *_list.txt > all_list.txt

#python gen_news.py -c 2000 -l cn -wk
#python gen_news.py -c 2000 -l en -wk
#python gen_news.py -c 5000 -l cn -w 北京


#-redata datasets/data

python gen_run.py -t 20 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out -i all_list.txt
python gen_run.py -t 20 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out -i text/cns.txt
python gen_run.py -t 20 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out -i text/cns2.txt
python gen_run.py -t 20 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out -i text/cns3.txt
python gen_run.py -t 20 -fs 28 -new_h 32 -new_w 320 -w 2 -c 200000 -news -mxw 18 -miw 15 -l cn -e png -aug  --output_dir out -i text/ens.txt

python gen_run2.py -t 20 -b 3  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i all_list.txt
python gen_run2.py -t 20 -b 1  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i all_list.txt
python gen_run2.py -t 20 -b 3  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns.txt
python gen_run2.py -t 20 -b 1  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns.txt
python gen_run2.py -t 20 -b 3  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns2.txt
python gen_run2.py -t 20 -b 1  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns2.txt
python gen_run2.py -t 20 -b 3  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns3.txt
python gen_run2.py -t 20 -b 1  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/cns3.txt
python gen_run2.py -t 20 -b 3  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/ens.txt
python gen_run2.py -t 20 -b 1  -w 2 -f 28 -na 1 -c 100000 -k 5 -rk -d -do  -l cn -e png  -mxw 18 -miw 15 -news --output_dir out4  -i text/ens.txt

python gen_run.py -t 15 -fs 28 -new_h 32 -new_w 320 -w 2 -c 50000 -news -mxw 18 -miw 15 -l cn -e png  --output_dir out2 -i text/cns.txt
python gen_run.py -t 15 -fs 28 -new_h 32 -new_w 320 -w 2 -c 60000 -news -mxw 18 -miw 15 -l cn -e png  --output_dir out3 -i text/cns.txt


python gen_record.py --dataset_name=train --dataset_dir=out --dataset_nums=10000 --output_dir=datasets/train

python gen_record.py --dataset_name=train4 --dataset_dir=out4 --dataset_nums=10000 --output_dir=datasets/train

python gen_record.py --dataset_name=test --dataset_dir=out2 --dataset_nums=10000 --output_dir=datasets/test

python gen_record.py --dataset_name=validation --dataset_dir=out3 --dataset_nums=10000 --output_dir=datasets/validation



var=`ls out |wc -l`
var4=`ls out4 |wc -l`
var2=`expr $var + $var4`
sed -i 's/430000/'"$var2"'/g' datasets/my_data.py

var=`ls out2 |wc -l`
sed -i 's/24000/'"$var"'/g' datasets/my_data.py

var=`ls out3 |wc -l`
sed -i 's/22000/'"$var"'/g' datasets/my_data.py


#python train.py --checkpoint_inception=./resource/inception_v3.ckpt --dataset_name=my_data > output.log 2>&1 &

#python train.py --checkpoint=../attention_ocr_2017_08_09/model.ckpt-399731 --dataset_name=my_data > output.log 2>&1 &

python train.py --checkpoint=./model.ckpt-118564 --dataset_name=my_data > output.log 2>&1 &


#python eval.py --dataset_name=my_data --split_name=test

#python eval.py --dataset_name=my_data --split_name=validation --eval_log_dir=valid_logs


#tar -zcvf my_attention_ocr.tar.gz --exclude=**/venv --exclude=/venv --exclude=**/**/inception_v3** --exclude=.git --exclude=**/.git*  --exclude=.idea*  my_attention_ocr/
