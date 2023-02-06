# KICE
Code for paper "KICE: A Knowledge Consolidation and Expansion Framework for Relation Extraction"

## Get Start
The required python packages are documented in `requirements.txt`

### Data Preparation
The expect structure of files:

    KICE
     |-- data
     |  |-- tacred
     |  |  |-- train.json  
     |  |  |-- dev.json  
     |  |  |-- test.json
     |  |  |-- rel2id.json
     |  |-- retacred
     |  |  |-- train.json  
     |  |  |-- dev.json  
     |  |  |-- test.json
     |  |  |-- rel2id.json
 
To remove the ":" and "_" in the relation mentions:

    cd data/tacred
    python preprocess.py

To sample seed training set and dev set:
    
    cd data
    bash sample.sh
    

## Knowledge Stimulation

Use the seed data to prompt-tune the PLM for pattern extraction and learn the initial RE model.

    bash preprocess.sh

## Knowledge Consolidation

Apply the previous RE model on unlabeled set and pick new rules from high-confidence ones.

    bash run_boot.sh ${step} ${last_step}

You should input the current step number `${step}` and previous step number `${last_step}`. 
For example, to conduct the first Knowledge Consolidation step, we could run `bash run_boot.sh 1 0`.

## Knowledge Expansion

Sample the most confusing unlabeled data for extra human annotation and build new rules. (In our experiment, these data are labeled with their ground truth label in original dataset.)

    bash run_act.sh ${step} ${last_step}

If you want to take the KnowPrompt as the backbone, just replace the `train.sh`, 
`train_self.sh`, `pred.sh` and `get_query.sh` with the corresponding files in `KnowPrompt/scripts`.
