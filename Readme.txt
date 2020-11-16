

20200518
增加LMMS向量
#Train on LARGE
cd /userhome/Openi_data_backup/DecE2S/CapDecE2S&&export BERT_BASE_DIR=/userhome/Openi_data_backup/DecE2S/BERT_large&&
python train.py --task_name=WM --do_train=true --data_dir=./data/traindata --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./GlossLmmsBERT_large_output/

python predict.py --task_name=WM --do_predict=true --data_dir=../CapDecE2S/data/LMMS_SE07 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=../CapDecE2S/GlossLmmsBERT_large_output_run1/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_large_lmms_SE07_prediction.txt ./semeval2007/semeval2007.gold.key.txt 73.5% 73.1%run1 74.2%run2
java Scorer DecE2S_large_lmms_SE13_prediction.txt ./semeval2013/semeval2013.gold.key.txt 72.6% 73.3%run1 72.9%run2
java Scorer DecE2S_large_lmms_SE15_prediction.txt ./semeval2015/semeval2015.gold.key.txt 70.4% 70.7%run1 71.5%run2
java Scorer DecE2S_large_lmms_SE2_prediction.txt ./senseval2/senseval2.gold.key.txt 78.8% 79.6%run1 80.1%run2
java Scorer DecE2S_large_lmms_SE3_prediction.txt ./senseval3/senseval3.gold.key.txt 80.7% 79.1%run1 80.5%run2

java Scorer ./semeval2007/semeval2007.gold.key.txt DecE2S_lmms_SE07_prediction.txt  73.8%
java Scorer ./semeval2013/semeval2013.gold.key.txt DecE2S_lmms_SE13_prediction.txt  72.9%
java Scorer ./semeval2015/semeval2015.gold.key.txt DecE2S_lmms_SE15_prediction.txt 76.8%
java Scorer ./senseval2/senseval2.gold.key.txt DecE2S_lmms_SE2_prediction.txt 80.8%
java Scorer ./senseval3/senseval3.gold.key.txt DecE2S_lmms_SE3_prediction.txt 80.3%


纠正评价

java Scorer ./semeval2007/semeval2007.gold.key.txt DecE2S_large_lmms_run1_SE07_prediction.txt  73.5% 73.4%run1 74.2%run2
java Scorer ./semeval2013/semeval2013.gold.key.txt DecE2S_large_lmms_run1_SE13_prediction.txt  72.6% 73.7%run1 72.9%run2
java Scorer ./semeval2015/semeval2015.gold.key.txt DecE2S_large_lmms_run1_SE15_prediction.txt  70.4% 77.9%run1 71.5%run2
java Scorer ./senseval2/senseval2.gold.key.txt DecE2S_large_lmms_run1_SE2_prediction.txt  78.8% 81.7%run1 80.1%run2
java Scorer ./senseval3/senseval3.gold.key.txt DecE2S_large_lmms_run1_SE3_prediction.txt  80.7% 80.1%run1 80.5%run2


java Scorer ./semeval2007/semeval2007.gold.key.txt DecE2S_large_lmms_run2_SE07_prediction.txt  73.5% 73.4%run1 74.5%run2
java Scorer ./semeval2013/semeval2013.gold.key.txt DecE2S_large_lmms_run2_SE13_prediction.txt  72.6% 73.7%run1 73.2%run2
java Scorer ./semeval2015/semeval2015.gold.key.txt DecE2S_large_lmms_run2_SE15_prediction.txt  70.4% 77.9%run1 78.8%run2
java Scorer ./senseval2/senseval2.gold.key.txt DecE2S_large_lmms_run2_SE2_prediction.txt  78.8% 81.7%run1 82.2%run2
java Scorer ./senseval3/senseval3.gold.key.txt DecE2S_large_lmms_run2_SE3_prediction.txt  80.7% 80.1%run1 81.4%run2


java Scorer ./semeval2007/semeval2007.gold.key.txt DecE2S_large_lmms_SE07_prediction.txt  73.5% 73.4%run1 74.5%run2
java Scorer ./semeval2013/semeval2013.gold.key.txt DecE2S_large_lmms_SE13_prediction.txt  72.6% 73.7%run1 73.2%run2
java Scorer ./semeval2015/semeval2015.gold.key.txt DecE2S_large_lmms_SE15_prediction.txt  70.4% 77.9%run1 78.8%run2
java Scorer ./senseval2/senseval2.gold.key.txt DecE2S_large_lmms_SE2_prediction.txt  78.8% 81.7%run1 82.2%run2
java Scorer ./senseval3/senseval3.gold.key.txt DecE2S_large_lmms_SE3_prediction.txt  80.7% 80.1%run1 81.4%run2


----------目录下对应测试--------------
python predict.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE07 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_large_output_run1/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/predict_run0

java Scorer ./semeval2007/semeval2007.gold.key.txt predict_run0/CapsDecE2S_large_lmms_SE07_prediction.txt 
java Scorer ./semeval2013/semeval2013.gold.key.txt predict_run0/CapsDecE2S_large_lmms_SE13_prediction.txt  
java Scorer ./semeval2015/semeval2015.gold.key.txt predict_run0/CapsDecE2S_large_lmms_SE15_prediction.txt  
java Scorer ./senseval2/senseval2.gold.key.txt  predict_run0/CapsDecE2S_large_lmms_SE2_prediction.txt
java Scorer ./senseval3/senseval3.gold.key.txt  predict_run0/CapsDecE2S_large_lmms_SE3_prediction.txt
java Scorer all/all.gold.key.txt predict_run0/CapsDecE2S_large_lmms_all_prediction.txt


java Scorer ./semeval2007/semeval2007.gold.key.txt predict_run1/CapsDecE2S_large_lmms_run1_SE07_prediction.txt 
java Scorer ./semeval2013/semeval2013.gold.key.txt predict_run1/CapsDecE2S_large_lmms_run1_SE13_prediction.txt  
java Scorer ./semeval2015/semeval2015.gold.key.txt predict_run1/CapsDecE2S_large_lmms_run1_SE15_prediction.txt  
java Scorer ./senseval2/senseval2.gold.key.txt  predict_run1/CapsDecE2S_large_lmms_run1_SE2_prediction.txt
java Scorer ./senseval3/senseval3.gold.key.txt  predict_run1/CapsDecE2S_large_lmms_run1_SE3_prediction.txt
java Scorer all/all.gold.key.txt predict_run1/CapsDecE2S_large_lmms_run1_all_prediction.txt

java Scorer ./semeval2007/semeval2007.gold.key.txt predict_run2/CapsDecE2S_large_lmms_run2_SE07_prediction.txt 
java Scorer ./semeval2013/semeval2013.gold.key.txt predict_run2/CapsDecE2S_large_lmms_run2_SE13_prediction.txt  
java Scorer ./semeval2015/semeval2015.gold.key.txt predict_run2/CapsDecE2S_large_lmms_run2_SE15_prediction.txt  
java Scorer ./senseval2/senseval2.gold.key.txt  predict_run2/CapsDecE2S_large_lmms_run2_SE2_prediction.txt
java Scorer ./senseval3/senseval3.gold.key.txt  predict_run2/CapsDecE2S_large_lmms_run2_SE3_prediction.txt
java Scorer all/all.gold.key.txt predict_run2/CapsDecE2S_large_lmms_run2_all_prediction.txt

#Train
cd /userhome/DecE2S/CapDecE2S&&export BERT_BASE_DIR=/userhome/Openi_data_backup/DecE2S/BERT_base/uncased_L-12_H-768_A-12&&python run_classifier.py --task_name=WM --do_train=true --data_dir=./data/GlossLmmsBERT --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./GlossLmmsBERT_base_output/

#Test
python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE07 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_base_output/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_lmms_SE07_prediction.txt ./semeval2007/semeval2007.gold.key.txt

python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE13 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_base_output/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_lmms_SE13_prediction.txt ./semeval2013/semeval2013.gold.key.txt

python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE15 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_base_output/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_lmms_SE15_prediction.txt ./semeval2015/semeval2015.gold.key.txt


python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE2 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_base_output/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_lmms_SE2_prediction.txt ./senseval2/senseval2.gold.key.txt


python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE3 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossLmmsBERT_base_output/model.ckpt-198143 --max_seq_length=128 --output_dir=./data/

java Scorer DecE2S_lmms_SE3_prediction.txt ./senseval3/senseval3.gold.key.txt

#######################################################################################


cd /userhome/DecE2S/DecE2S_BERT_large&&
export BERT_BASE_DIR=/userhome/DecE2S/BERT_base/uncased_L-24_H-1024_A-16&&
python run_classifier.py --task_name=WM \
--do_train=true --data_dir=./data/GlossLmmsBERT \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 --train_batch_size=32 \
--learning_rate=2e-5 --num_train_epochs=10.0 \
--output_dir=./GlossLmmsBERT_large_output/




20191208
使用GlossBERT数据在large上进行训练，只是用targetword表示的模型效果还不错，所以抓紧在匹配模型上测试下
云脑命令
cd /userhome/DecE2S/DecE2S_BERT_large/&&export BERT_BASE_DIR=/userhome/DecE2S/BERT_large&&python run_classifier.py --task_name=WM --do_train=true --data_dir=./data/GlossBERT --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./GlossBERT_large_output/
测试命令
python run_probability.py --task_name=WM --do_predict=true --data_dir=./data/SE07 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossBERT_large_output/model.ckpt-150000 --max_seq_length=128 --output_dir=./data/

python run_probability.py --task_name=WM --do_predict=true --data_dir=./data/SE07 --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./GlossBERT_large_output_v3/model.ckpt-297214 --max_seq_length=128 --output_dir=./data/






20191127
参数目录wm_large_output,是DecE2S_BERT_large在semcor匹配数据上(./data/wm/train.tsv)的训练参数，对应WSD测试结果文件在./data/Trained_on_semcor_and_gloss/
参数目录gloss_large_output_v2，是DecE2S_BERT_large在gloss_data数据(./data/gloss/train.tsv)上的单独的训练参数，DecE2S_trained_gloss_***_prediction.txt

20191118
capsules*3+BERT_large[-1:-4]结构在wm上训练，并用测试集当成验证dev.tsv原验证集备份dev_bk.tsv

云脑任务命令：
cd /userhome/DecE2S/DecE2S_BERT_large/&&export BERT_BASE_DIR=/userhome/DecE2S/BERT_large&&python run_classifier.py --task_name=WM --do_train=true --do_eval=true --data_dir=./data/wm --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./wm_large_output/

训练完，测试结果
python run_classifier.py --task_name=WM --do_eval=true --data_dir=./data/wm --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./wm_large_output/model.ckpt-93750 --max_seq_length=128 --output_dir=./wm_large_output/


承接上一步wm参数，在gloss上继续训练,暂时用wm测试集(dev.tsv)进行验证
cd /userhome/DecE2S/DecE2S_BERT_large/&&export BERT_BASE_DIR=/userhome/DecE2S/BERT_large&&python run_classifier.py --task_name=WM --do_train=true --do_eval=true --data_dir=./data/gloss --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=./wm_large_output/model.ckpt-93750 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./gloss_large_output/


-------------------------------------------
capsule_bert目录下的程序是使用胶囊网络做分解的程序，使用WoM数据集或者WiC数据集都可以训练，
在WoM训练和测试的命令为
在WoM训练和测试的命令为：
在WiC训练和测试的命令为：
ON WIC
python run_classifier.py --task_name=WM --do_train=true --do_eval=true --data_dir=./data/word_match/wic --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./tmp/wm_output/

python run_classifier.py --task_name=WM --do_eval=true --data_dir=./data/word_match/wic --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=/tmp/mrpc_output/model.ckpt-1696 --max_seq_length=128 --output_dir=./tmp/wm_output/
python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/word_match/wic --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=/tmp/mrpc_output/model.ckpt-1696 --max_seq_length=128 --output_dir=./tmp/wm_output/

ON WM_semcor
python run_classifier.py --task_name=WM --do_train=true --do_eval=true --data_dir=./data/word_match/wm --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./tmp/wm_output/

python run_classifier.py --task_name=WM --do_eval=true --data_dir=./data/word_match/wm --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=/tmp/mrpc_output/model.ckpt-93749 --max_seq_length=128 --output_dir=./tmp/wm_output/
python run_classifier.py --task_name=WM --do_predict=true --data_dir=./data/word_match/wm --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=/tmp/mrpc_output/model.ckpt-93749 --max_seq_length=128 --output_dir=./tmp/wm_output/

1.已经训练好的模型，在WoM上的参数保存在./tmp/8377_wm，可以直接输出测试结果，得到output.tsv，并将改文件复制到类标目录，
cp output.tsv /home/gdls1080/STORE_DISK/semantic/src/BERT/wm_bert/data/word_match/output.txt
2.将对应文件的类标从数据中提取出来，目录wm_bert/data/word_match/wm: cat test.tsv | awk -F '\t' '{print $1}' >../gold.txt
3.输出文件和类标文件合并成r.txt文件paste -d"\t" gold.txt output.txt > r.txt，并使用s.py文件进行统计python s.py

单独的Bert模型在../wm_bert，是只用bert做匹配的模型，在WoM上也训练了参数，保存在系统目录/tmp/wm_output下，准确率为82.3

注意：
1.wm_bert和capsule_bert文件都有read_weight.py文件，通过model_fn_builder/model_fn/q_sense_weight, p_sense_w两个变量作为返回结果，
不过在wm_bert的模型构造时可能会多出两个不用的参数，ids_a和ids_b，记得注释掉。
2.wm_bert和capsule_bert两个模型processor名字参数分别为MRPC和WM， 记得在不同的模型中传不同的参数