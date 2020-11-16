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
* "data/java\*" the evaluation script, command "java Scorer [gold-standard] [system-output]"<br>
* "results/\*.png" the screenshots of three predictions on each test set.<br>

#### Requirements
Libraries: ubuntu = 16.04, cuda = 10.2, cudnn = 8, GPU card = NVIDIA Tesla V100 * 1<br>
Dependencies: python > 3.5, tensorflow > 1.10.0, pdb, numpy, tdqm, codecs<br>
