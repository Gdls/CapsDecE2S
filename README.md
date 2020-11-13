# CapsDecE2S
The source code for the CapsDecE2S model(coming soon).

#### Description
This repository includes the source code for the paper "Decomposing Word Embedding with the Capsule Network".
#### CapsDecE2S structure
![The structure of CapsDecE2S model.](https://github.com/Gdls/CapsDecE2S/blob/master/CapsDecE2S.png)

#### Code Structure:
>CapsDecE2S<br>
├── Results<br>
│   ├── Score.class<br>
│   ├── Score.java<br>
│   ├── predict_run0.png<br>
│   ├── predict_run1.png<br>
│   ├── predict_run2.png<br>
│   ├── pred<br>
│   │  ├── CapsDecE2S_large_lmms_run1_all_prediction.txt<br>
│   │  ├── CapsDecE2S_large_lmms_run1_SE07_prediction.txt<br>
│   │  ├── CapsDecE2S_large_lmms_run1_SE13_prediction.txt<br>
│   │  ├── CapsDecE2S_large_lmms_run1_SE15_prediction.txt<br>
│   │  ├── CapsDecE2S_large_lmms_run1_SE2_prediction.txt<br>
│   │  ├── CapsDecE2S_large_lmms_run1_SE3_prediction.txt<br>
│   ├── gold<br>
│   │  ├── all.gold.key.txt<br>
│   │  ├── semeval2007.gold.key.txt<br>
│   │  ├── semeval2013.gold.key.txt<br>
│   │  ├── semeval2015.gold.key.txt<br>
│   │  ├── senseval2.gold.key.txt<br>
│   │  ├── senseval3.gold.key.txt<br>
├── model.py<br>
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