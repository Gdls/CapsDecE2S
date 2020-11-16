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

* "modeling.py" the main structure of CapsDecE2S, including the backbone class("class Model()") and two functions("\def routing()" and "\def sense Global Local att()").<br> 
	* The class "class Model()" is the backbone of the CapsDecE2S model;<br>
	* The function "\def routing()" is for the embedding decomposing module;<br>
	* The function "\def sense Global Local att()" is used to calculate the global and local attention.<br>
* "results/pred/\*" the predictions on each test set.<br>
* "results/gold/\*" the gold labels on each test set.<br>
* "results/java\*" the evaluation script, command "java Scorer [gold-standard] [system-output]"<br>

#### Requirements
Libraries: ubuntu = 16.04, cuda = 10.2, cudnn = 8, GPU card = NVIDIA Tesla V100 * 1<br>
Dependencies: python > 3.5, tensorflow > 1.10.0, pdb, numpy, tdqm, codecs<br>
