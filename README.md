# multi_rc
Dataset page: https://cogcomp.seas.upenn.edu/multirc/

MultiRC (Multi-Sentence Reading Comprehension) is a dataset of short paragraphs and multi-sentence questions that can be answered from the content of the paragraph.

Approach:
1. We pick the SuperGLUE baseline BERT model and understand the code.
2. We try and improve the model to obtain better scores on the Multi-RC dataset
3. Research paper references- 
a. Google T5
b. Facebook RoBERTa
c. Google BERT

- ISSUES:

TBD


PROGRESS TRACK -

Complete overview of JIANT: https://arxiv.org/pdf/2003.02249.pdf 

March 16: Tuned baseline model jiant to execute task 'MultiRC'

Steps:

- Setup environment: conda env create -f environment.yml

- Activate environmentt: conda activate jiant

- Install remaining packages:
pip install torchvision 
pip install allennlp
pip install jsondiff
pip install -U sacremoses
pip install pyhocon
pip install transformers
pip install python-Levenshtein

- Download dataset: python scripts/download_superglue_data.py --data_dir data --tasks MultiRC

- Setup environment variables > cp user_config_template.sh user_config.sh:

$JIANT_PROJECT_PREFIX=/home/varun/PycharmProjects/jiant
 
$JIANT_DATA_DIR=/home/varun/PycharmProjects/jiant/data

source user_config.sh

- Configuring environment > cp jiant/config/demo.conf jiant/config/multirc.conf:

pretrain_tasks="multirc" (TBD)

target_tasks="multirc"

input_module = bert-base-cased

- Allows same pre/target tasks > jiant/config/defaults.conf : allow_reuse_of_pretraining_parameters = 1

- Run task:

python main.py --config_file jiant/config/multirc.conf

To Visualize logs:

pip install tensorboard

tensorboard --logdir=${LOG_DIR} 

REPORT (WIP): 

https://www.overleaf.com/read/gxckmzynvxnk



