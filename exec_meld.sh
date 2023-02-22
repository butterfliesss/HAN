#!bin/bash
# Var assignment
LR=1.0e-5 # for word2vec 1.0e-4
GPU='0'
du=1024 # 300 # 
dc=1024 # 300 #
dataset="MELD" 
data_path="Data/MELD/MELD_data.pt"
vocab_path="Data/MELD/MELD_vocab.pt"
emodict_path="Data/MELD/MELD_emodict.pt"
tr_emodict_path="Data/MELD/MELD_tr_emodict.pt"
embedding="Data/MELD/MELD_embedding.pt"
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $Enc - $Dec $iter ---
python -u EmoMain.py -epochs 25 -lr $LR -gpu $GPU -d_h1 $du -d_h2 $dc -report_loss 1038 -dataset $dataset -data_path $data_path -vocab_path $vocab_path -emodict_path $emodict_path -tr_emodict_path $tr_emodict_path -embedding $embedding
done > meld_han.txt 2>&1 &