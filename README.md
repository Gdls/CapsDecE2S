# CapsDecE2S
The source code for the CapsDecE2S model(coming soon).
![The structure of CapsDecE2S model.](https://github.com/Gdls/CapsDecE2S/blob/master/CapsDecE2S.png)

#### Description
This repository includes the source code for the paper "Decomposing Word Embedding with the Capsule Network".

#### Code Structure:
>CapsDecE2S<br>
├── __init__.py<br>
├── train.py<br>
├── predict.py<br>
├── modeling.py<br>
├── match_utils.py<br>
├── optimization.py<br>
├── tokenization.py<br>
├── model.py<br>
├── bert_config.json<br>
├── vocab.txt<br>
├── data<br>
│   ├── Score.class<br>
│   ├── Score.java<br>
│   ├── statistic.py<br>
│   ├── traindata<br>
│   │  ├── train.tsv<br>
│   ├── LMMS_SE\*<br>
│   │  ├── test.tsv<br>
│   ├── gold<br>
│   │  ├── \*.gold.key.txt<br>
│   ├── lmms<br>
│   │  ├── lmms1024_emb.npy<br>
├── Results<br>
│   ├── predict_run0.png<br>
│   ├── predict_run1.png<br>
│   ├── predict_run2.png<br>
├── CapsDecE2S.png<br>
└── README.md<br>

* "train.py" the training file.<br>
* "predict.py" the prediction file.<br>
* "modeling.py" the main structure of CapsDecE2S, including the backbone class("class Model()")<br> 
* "match_utils.py" function file
* "bert_config.json/vocab.txt" configure file
* "data/traindata/train.tsv" training data(see title.txt for the format)
* "data/LMMS_SE\*/test.tsv" test file for SE07,SE13,SE15,SE2,SE3
* "data/gold/\*.gold.key.txt" gold label file for each test file
* "data/lmms/lmms1024_emb.npy" lmms embedding file
* "data/statistic.py" converting the prediciton file into the format of [system-output]
* "data/java\*" the evaluation script, command "java Scorer [gold-standard] [system-output]"<br>
* "results/\*.png" the screenshots of three predictions on each test set.<br>

#### Train Model
* export BERT_BASE_DIR=BERT_large # download the BERT large parameters here

* python train.py --task_name=WM --do_train=true --data_dir=./data/traindata --vocab_file=./vocab.txt --bert_config_file=./bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=10.0 --output_dir=./output/

#### Prediction
#example prediction on SE07
* python predict.py --task_name=WM --do_predict=true --data_dir=./data/LMMS_SE07 --vocab_file=./vocab.txt --bert_config_file=./bert_config.json --init_checkpoint=./output/model.ckpt-*** --max_seq_length=128 --output_dir=./data/

* python statistical.py SE07 #output file CapsDecE2S_large_lmms_SE07_prediction.txt
* java Score gold/semeval2007.gold.key.txt CapsDecE2S_large_lmms_SE07_prediction.txt

#### Requirements
Libraries: ubuntu = 16.04, cuda = 10.2, cudnn = 8, GPU card = NVIDIA Tesla V100 * 1<br>
Dependencies: python > 3.5, tensorflow > 1.10.0, pdb, numpy, tdqm, codecs<br>
