#!bin/bash
# Var assignment
LR=1.0e-4 # for word2evc 5.0e-4
GPU='0'
du=1024 # 1024
dc=1024 # 1024
dataset="IEMOCAP6"
data_path="Data/IEMOCAP6/IEMOCAP6_data.pt"
vocab_path="Data/IEMOCAP6/IEMOCAP6_vocab.pt"
emodict_path="Data/IEMOCAP6/IEMOCAP6_emodict.pt"
tr_emodict_path="Data/IEMOCAP6/IEMOCAP6_tr_emodict.pt"
embedding="Data/IEMOCAP6/IEMOCAP6_embedding.pt"
echo ========= lr=$LR ==============
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $Enc - $Dec $iter ---
python -u EmoMain.py -epochs 60 -lr $LR -gpu $GPU -d_h1 $du -d_h2 $dc -report_loss 96 -dataset $dataset -data_path $data_path -vocab_path $vocab_path -emodict_path $emodict_path -tr_emodict_path $tr_emodict_path -embedding $embedding
done > iemocap_han.txt 2>&1 &
