import os
import sys
import codecs
import pdb
title = sys.argv[1]#"SE13"
fgold = codecs.open("../data/LMMS_"+title+"/test.tsv","r","utf-8")
goldlist = fgold.read().splitlines()
i = 0
mid_dict = {}
while i < len(goldlist):
  line = goldlist[i]
  mid = line.split("\t")[13]
  if mid not in mid_dict.keys():
    mid_dict[mid] = []
    mid_dict[mid].append(i)
  else:
    mid_dict[mid].append(i)
  i = i+1

fgold.close()
output_dict = {}
fw = codecs.open("./CapsDecE2S_large_lmms_"+title+"_prediction.txt","w","utf-8")
fpredict = codecs.open("../data/output.tsv","r","utf-8")
datalist = fpredict.read().splitlines()
for mid in mid_dict.keys(): 
  _start = mid_dict[mid][0]
  _end = mid_dict[mid][-1]
  gold_block = goldlist[_start:_end+1]
  mid_block = [float(elem) for elem in datalist[_start:_end+1]]
  sense_index = mid_block.index(max(mid_block))
  selection = gold_block[sense_index]
  output_dict[mid] = selection.split("\t")[11]

sortlist = sorted(output_dict.items(),key=lambda asd:asd[0],reverse=False)
for elem in sortlist:
  fw.write(elem[0]+" "+elem[1]+"\n")
fpredict.close()