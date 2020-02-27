# BertSum-Experiment-Results
This repo is to host the artifacts gathered from a number of experiments in training and testing with BertSum (https://github.com/nlpyang/BertSum) in different configurations.

Based on https://paperswithcode.com/sota/document-summarization-on-cnn-daily-mail , BertSum is the most advanced NLP model for Extractive and Abstractive Summarization of text, based on Google Language’s BERT pretrained model.

I am trying to fine tune this model to generate short summaries from technical document, say, Azure docs.

## Environment
### Hardware 
Standard NV6_Promo (6 vcpus, 56 GiB memory)<br/>
Linux (ubuntu 18.04)

### Software
NVIDIA Driver Version: 440.59       
CUDA Version: 10.2

Python 3.6.9
dill (0.3.1.1)
pandas (1.0.1)
pyrouge (0.1.3)
pytorch-pretrained-bert (0.6.2)
pytorch-transformers (1.2.0)
tensorboardX (2.0)
torch (1.1.0)
ROUGE 1.5.5

To install ROUGE correctly, I first follow https://poojithansl7.wordpress.com/2018/08/04/setting-up-rouge/. After ROUGE installed, I installed pytorch by command pip3 install pytorch.

The repo for BertSum was cloned.

### Dataset
The processed CNN/DM PT files are downloaded from https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6

## Test 1
### Fine tuning the model
Keep only first three PT files (*0.train.pt, *0.valid.pt and *0.test.pt) and rename them to 
.test.pt
.valid.pt
.test.pt
Run the classification training command
python3 train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm/ -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0  -gpu_ranks 0 -world_size 1 -report_every 50 -save_checkpoint_steps 1000 -batch_size 1000 -decay_method noam -train_steps 10000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 1000 &

### Test 
Get the first few paragraphs from a Microsoft Azure Databricks document.
Tokenize the document
Generate the PT file with
python3 preprocess.py -mode format_to_bert -raw_path ../json_data -save_path ../json_data -oracle_mode greedy -n_cpus 1 -log_file ../logs/preprocess.log
Rename the PT file to .test.pt and copy to ../bert_data/cnndm
Run the test with the best model step
python3 train.py -mode test  -bert_data_path ../bert_data/cnndm/  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file ../logs/bert_test -test_from ../models/bert_classifier/model_step_4000.pt -block_trigram true &
We get the results!
cat cnndm_step4000.candidate
the hierarchical namespace organizes objects/files into a hierarchy of directories for efficient data access .<q>designed from the start to service multiple petabytes of information while sustaining hundreds of gigabits of throughput, data lake storage gen2 allows you to easily manage massive amounts of data .<q>a common object store naming convention uses slashes in the name to mimic a hierarchical directory structure .

cat cnndm_step4000.gold
data lake storage gen2 makes azure storage the foundation for building enterprise data lakes on azure .

