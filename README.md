# MultiRC #
Dataset page: https://cogcomp.seas.upenn.edu/multirc/

REPORT (WIP): https://www.overleaf.com/5821917254rswjnrsfhhzy

MultiRC (Multi-Sentence Reading Comprehension) is a dataset of short paragraphs and multi-sentence questions that can be answered from the content of the paragraph.

## Repository Structure ##
The repo consists of following files/folders: 
1. ***baseline***: The jiant software toolkit 
    1. **output**: consists of folders of experiments we conducted and their respective results.
        1. results.tsv consists of cumulative evaluation results over the runs
        2. log.log files have the complete log for respective runs
        3. params.conf have a copy of the configurations used for that run
    2. **data**: consists various task datasets with each folder containing train, validation and test data
    3. **jiant/config**: contains the default and custom configurations files.
    4. **jiant**: Implementation scripts and related modules (hugging face, allennlp).
    5. **user_config.sh**: contains the default environment variables(user configurations) like project base directory path, dataset path
2. ***dataset***: The train and dev datasets
3. ***Docs***: Related research papers
4. ***Topic notes***

##Configurations ##
(subset of configurations from default.conf which we have overriden on custom config files)

Argument| Description| Value(eg:)
---------| -------------|----
max_seq_len| Maximum sequence length, in tokens. **Mainly needed for MultiRC, to avoid over-truncating**| 256
batch_size| Training batch size| 32
lr| Initial learning rate| 0.0003
dropout| Dropout rate| 0.1
val_interval| Interval (in steps) at which you want to evaluate your model on the validation set during pretraining. A step is a batch update| 100
input_module| The word embedding or contextual word representation layer| "bert-base-uncased"
transformers_output_mode | How to handle the embedding layer of the BERT| "top"
classifier| The type of the final layer(s) in classification and regression tasks| log_reg
pair_attn| If true, use attn in sentence-pair classification/regression tasks| 0
optimizer| Use 'bert_adam' for reproducing BERT experiments| "bert_adam"
||          **Stopping criteria**        ||
min_lr| Minimum learning rate. Training will stop when our explicit LR decay lowers| 0.000001
max_vals| Maximum number of validation checks. Will stop once these many validation steps are done| 1000
max_epochs| Maximum number of epochs (full pass over a task's training data)| -1
||         **Task specification**        ||
pretrain_tasks| List of tasks to pretrain on| "sst"
target_tasks| (MultiRC in our case) list of target tasks to (train and) test on| "multirc"
do_pretrain| Run pre-train on tasks mentioned in pretrain_tasks| 1
do_target_task_training| After do_pretrain, train on the target tasks in target_tasks| 1
do_full_eval| Test on the tasks in target_tasks| 1 
load_model| If true, restore from checkpoint when starting do_pretrain. No impact on do_target_task_training| 1 
load_target_train_checkpoint| load the specified model_state checkpoint for target_training| none
load_eval_checkpoint| load the specified model_state checkpoint for evaluation| none
write_preds| list of splits for which predictions need to be written to disk| 1
||        **Misc**                     ||
exp_name| Name of the current experiment|  
run_name| Name of the run given an experiment|





###Approach ###
1. We pick the SuperGLUE baseline BERT model and understand the code.
2. We try and improve the model to obtain better scores on the Multi-RC dataset
3. Research paper references- 
a. Google T5
b. Facebook RoBERTa
c. Google BERT

### ISSUES ###

### TBD ###

### PROGRESS TRACK ###

Complete overview of JIANT: https://arxiv.org/pdf/2003.02249.pdf 

March 16: Tuned baseline model jiant to execute task 'MultiRC'

##Setup steps ##

- Setup environment: 

        conda env create -f environment.yml

- Activate environment: 
        
        conda activate jiant

- Install remaining packages:
~~pip install torchvision 
pip install allennlp
pip install jsondiff
pip install -U sacremoses
pip install pyhocon
pip install transformers
pip install python-Levenshtein~~

- Download dataset: 

        python scripts/download_superglue_data.py --data_dir data --tasks MultiRC

- Setup environment variables > 

        cp user_config_template.sh user_config.sh:

eg: 

    export JIANT_DATA_DIR=/home/varun/PycharmProjects/jiant/data
    export JIANT_PROJECT_PREFIX=/home/varun/PycharmProjects/jiant

source user_config.sh

- Experiment configurations > cp jiant/config/demo.conf jiant/config/multirc.conf:
eg changes:

        pretrain_tasks="sst"

        target_tasks="multirc"

        input_module = bert-base-cased

- Run task:

        source user_config.sh; 
        python main.py --config_file jiant/config/multirc.conf

### To Visualize logs ###

        pip install tensorboard

        tensorboard --logdir=${LOG_DIR} 

