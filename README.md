# BertSum-Experiment-Results
This repo is to host the artifacts gathered from a number of experiments in training and testing with BertSum (https://github.com/nlpyang/BertSum) in different configurations.

Based on https://paperswithcode.com/sota/document-summarization-on-cnn-daily-mail , BertSum is the most advanced NLP model for Extractive and Abstractive Summarization of text, based on Google Language’s BERT pretrained model.

I am trying to fine tune this model to generate short summaries from technical document, say, Azure docs.

## Environment
### Hardware 
Standard NV6_Promo (6 vcpus, 56 GiB memory)<br/>
Linux (ubuntu 18.04)<br/>
Add a 1TB SDD data disk to the VM by following https://docs.microsoft.com/en-us/azure/virtual-machines/linux/attach-disk-portal<br/>
Mounted the datadisk as<br/>
sudo mount /dev/sdc1 /datadrive

### Software
NVIDIA Driver Version: 440.59<br/>       
CUDA Version: 10.2

Python 3.6.9<br/>
dill (0.3.1.1)<br/>
pandas (1.0.1)<br/>
pyrouge (0.1.3)<br/>
pytorch-pretrained-bert (0.6.2)<br/>
pytorch-transformers (1.2.0)<br/>
tensorboardX (2.0)<br/>
torch (1.1.0)<br/>
ROUGE 1.5.5

To install ROUGE correctly, I first follow https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/. After ROUGE installed, I installed pytorch by command pip3 install pytorch.

The repo for BertSum was cloned from https://github.com/nlpyang/BertSum to /datadrive on 2/25/2020.

### Dataset
The processed CNN/DM PT files are downloaded from https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6

## Test 1
Test 1 is to test the model that on the top if Bert adding two lyaers of transformer to serve as a classfier.
### Fine tuning the model
Keep only first three PT files (*0.train.pt, *0.valid.pt and *0.test.pt) and rename them to<br/> 
.test.pt<br/>
.valid.pt<br/>
.test.pt<br/>
Run the classification training command<br/>
python3 train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm/ -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 1000 -decay_method noam -train_steps 10000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 1000 &

### Test 
Get the first few paragraphs from a Microsoft Azure Databricks document, azuredatabrickst.txt.<br/>
Tokenize the document, ABD.json<br/>
I copy the first sentence as "tgt" and rename the file to cnndm_sample.train.0.json and then generate the PT file with<br/>
python3 preprocess.py -mode format_to_bert -raw_path ../jack_test -save_path ../jack_test -oracle_mode greedy -n_cpus 1 -log_file ../logs/preprocess.log
Rename the PT file to .test.pt and copy to ../bert_data/cnndm<br/>

Run the test with the best model step<br/>
python3 train.py -mode test  -bert_data_path ../bert_data/cnndm/  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file ../logs/bert_test -test_from ../models/bert_classifier/model_step_4000.pt -block_trigram true &<br/>

### Result
cat cnndm_step4000.candidate<br/>
the hierarchical namespace organizes objects/files into a hierarchy of directories for efficient data access .<q>designed from the start to service multiple petabytes of information while sustaining hundreds of gigabits of throughput, data lake storage gen2 allows you to easily manage massive amounts of data .<q>a common object store naming convention uses slashes in the name to mimic a hierarchical directory structure .

cat cnndm_step4000.gold<br/>
data lake storage gen2 makes azure storage the foundation for building enterprise data lakes on azure .

## Test 2
This tes is to test that similar architecture but with more transformer layers. The same data files, but this time we add all the transformers (parameters increased from 100M to 120M).<br/>

Run this command for training:<br/>
python3 train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/cnndm/ -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 512 -decay_method noam -train_steps 10000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 1000 -ff_size 2048 -inter_layers 2 -heads 8 &

Run this command for testing:<br/>
python3 train.py -mode test  -bert_data_path ../bert_data/cnndm/  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file ../logs/bert_test -test_from ../models/bert_transformer/model_step_10000.pt -block_trigram true &

### Result
cat cnndm_step10000.candidate<br/>
the hierarchical namespace organizes objects/files into a hierarchy of directories for efficient data access .<q>a fundamental part of data lake storage gen2 is the addition of a hierarchical namespace to blob storage .<q>a common object store naming convention uses slashes in the name to mimic a hierarchical directory structure .<br/>

cat cnndm_step10000.gold<br/>
data lake storage gen2 makes azure storage the foundation for building enterprise data lakes on azure .<br/>

## Test 3
This is to test the architecture as in Test 2 but with RNN layers for abstractive (rewriting).

The run command is<br/>
python3 train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_rnn -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 2000 -batch_size 512 -decay_method noam -train_steps 10000 -accum_count 2 -log_file ../logs/bert_rnn -use_interval true -warmup_steps 1000 -rnn_size 768 -dropout 0.1

### Result
cat cnndm_step4000.candidate<br/>
designed from the start to service multiple petabytes of information while sustaining hundreds of gigabits of throughput, data lake storage gen2 allows you to easily manage massive amounts of data .<q>a common object store naming convention uses slashes in the name to mimic a hierarchical directory structure .<q>the hierarchical namespace organizes objects/files into a hierarchy of directories for efficient data access .<br/>
  
cat cnndm_step4000.gold<br/>
data lake storage gen2 makes azure storage the foundation for building enterprise data lakes on azure .<br/>

## Discussion
Though we only have one document in process and the fine tuning data set is from CNN/DM (thus nothing with our test document), we can see that Bert + Tran + RNN provides a result more human readable.
