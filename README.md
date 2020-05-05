# Folder Structure

 - MutliRC_BERT_QA (BERT QA model)
    - exploratory_analysis: has code and analysis related to BERT QA model
 - MultiRC_NER (NER model - best so far)
    - data: tokenized data
    - evaluations: output prediction files
    - models: Trained model, config file and vocab
    - MultiRC_NER notebook: code for training the NER model on training data
    - MultiRC_NER_eval: code for evaluating the trained NER model on evaluation data
    - parser.py: converts the given MultiRC data from original format to the NER format
- MultiRC_NLI (NLI model)
    - preprocess_multirc.py: convert the given MultiRC data from original format to the NLI format
    - lib:
- baseline (JIANT toolkit)
- dataset
    - contains the data downloaded from MultiRC
- docs
    - contains collection of research papers we referenced
 

# Updates 
- Changed evaluate.py to include softmax(logits) i.e confidence (for labels 0 and 1) in the output json for validation and test.
- Added sample json outputs
- Added files for best model performance (Accuracy- 58%)
- Analysed confidence probabilities: Model is very underconfident and most options are labelled as TRUE(1). 
[confidence-analysis](https://imgur.com/xGWsrdN)
- Contacted the developers for gaining information on the performance, seems like they don't know how it degraded when they updated the toolkit to incorporate new Allen AI and Huggingface module versions(Issue thread- https://github.com/nyu-mll/jiant/issues/1052).
- After manually checking results, it is observed that a particular option with any resemblance to a portion of the paragraph is marked TRUE without taking the question into context. 
- Researched multi-hop approaches such as Multi-hop Question Answering via Reasoning Chains[Chen et. al. 2019](https://arxiv.org/pdf/1910.02610.pdf) that produces reasoning which seems a logical solution to trim down the paragraph into the most relevant for the particular question. This gave us the following idea.
- Analysed BERT-QA(fine-tuned on SQuAd) and other fine-tuned BERT models(on STS-B, QNLI) on MultiRC dataset, details in experiments/ folder. While it was able to give partially correct answers, it's single span approach failed in answering multihop questions(as expected). One important observation- frozen BERT without any pre-training gave approximately the same results. This highlights the challenging characteristics of the dataset and provides reason for the low-confident model, as it could not learn or find patterns necessary to answer the questions. Added python script in "MultiRC_BERT_QA/".
- Implemented approach in Repurposing Entailment for Multi-Hop Question Answering Tasks[Trivedi et. al. 2019](https://arxiv.org/pdf/1904.09380v1.pdf)(pre-trained on QNLI). This entailment approach gave the best-yet results of F1 score:63. The approach of formulating a hypothesis as the answer based on context to the question and transforming the dataset gave us ideas for one of our other main approaches, NER-based QA, also recommended by our project mentor. This entailment approach was further analysed on confidence metrics.
- Added task into the baseline model for the above approach and dataset transformation script under branch "MultiRC_NLI/"
- FINAL APPROACH = Implemented Named-entity-recognition based approach. Idea- Using the concept of BIO tagging to train the model on correct tags for the correct answer and vice-versa for the wrong answers. Pre-requisite- Tranformed the MultiRC dataset into an NER dataset with different tags, one each for- paragraph, question, correct and incorrect answer.
- Added colab notebooks with the required data for the above approach in the repository under MultiRC_NER/

## To Do
- Analyse the implementation of Entailment-based approach in terms of confidence and micro-analysis on samples of data.
- Analyse the F1 score and confidence of the NER based QA model.

# MultiRC #
Dataset page: https://cogcomp.seas.upenn.edu/multirc/

Analysis: https://docs.google.com/spreadsheets/d/1zLZw-e5Anm17ah5RsUGOAzEJpmqOGGp-nA_72XfQN-E/edit?usp=sharing

REPORT : https://www.overleaf.com/read/zfbzkqjzxwrb

PROGRESS Slides : https://docs.google.com/presentation/d/1Z8hRQzUXM6ZboHXiayK_s2NtFMi9Ek0osfTT1MWxj9s/edit?usp=sharing

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
5. ***experiments***: Details of expermimental analysis of a different approach.
6. ***MultiRC_NER***: An NER-based QA approach for MultiRC dataset. Required notebooks and data.

## Configurations ##

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


### STEPS ###

Complete overview of JIANT: https://arxiv.org/pdf/2003.02249.pdf 

Tuning baseline model jiant to execute task 'MultiRC'

##Setup steps ##

- Setup environment: 

        conda env create -f environment.yml

- Activate environment: 
        
        conda activate jiant

- Install remaining packages:

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


### Approach ###
1. Pick the SuperGLUE baseline BERT model and understand/explore the codebase.
2. Improve the model over baseline scores on the Multi-RC dataset.
3. Additional model references- 
a. Google T5
b. Facebook RoBERTa
c. Google BERT

### Improvements ###
1. Exploring confidence probabilities
2. Increasing the low EM(exact-match) score
3. Increasing F1-score over baseline results
