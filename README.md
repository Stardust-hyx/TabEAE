# TabEAE
This is the official implementation of the paper **Revisiting Event Argument Extraction: Can EAE Models Learn Better When Being Aware of Event Co-occurrences?** (ACL 2023).

## Quick links

- [TabEAE](#tabeae)
  - [Acknowledgement](#acknowledgement)
  - [Quick links](#quick-links)
  - [Preparation](#preparation)
    - [Environment](#environment)
    - [Data](#data)
  - [Run](#run)

## Preparation

### Environment
To run our code, please install all the dependency packages by using the following command:

```
conda create --name TabEAE python==3.7.11
pip install -r requirements.txt
```

### Data
We conduct experiments on 4 datasets: ACE05, RAMS, WikiEvents and MLEE.
- ACE05: This dataset is under the License of [LDC User Agreement for Non-Members](https://catalog.ldc.upenn.edu/license/ldc-non-members-agreement.pdf) and is not freely available. Access it from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06) and preprocess the data following [EEQA (2020'EMNLP)](https://github.com/xinyadu/eeqa/tree/master/proc).
Then run ./data/ace_eeqa/convert.py to aggregate events occurring in the same sentence into one instance with multiple events.
- RAMS: This dataset is under the License of [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- WikiEvents: This dataset is under the [MIT License](https://mit-license.org/).
- MLEE: This dataset is under the License of [CC BY-NC-SA 3.0](http://creativecommons.org/licenses/by-nc-sa/3.0/).

For RAMS/WikiEvents/MLEE, we write a script for data processing. Run the following command in the root directory of the repo.

```bash
bash ./data/download_dataset.sh
```  

Please make sure your data folder structure look like below.
```bash
TableEAE
  └── data
      ├── ace_eeqa
      │   └── data_final
      │       ├── train_convert.json
      │       ├── dev_convert.json
      │       └── test_convert.json
      ├── RAMS_1.0
      │   └── data_final
      │       ├── train.jsonlines
      │       ├── dev.jsonlines
      │       └── test.jsonlines
      ├── WikiEvent
      │   └── data_final
      │       ├── train.jsonl
      │       ├── dev.jsonl
      │       └── test.jsonl
      ├── MLEE
      │   └── data_final
      │       ├── train.json
      │       └── test.json
      ├── prompts
      │   ├── prompts_ace_full.csv
      │   ├── prompts_wikievent_full.csv
      │   └── prompts_rams_full.csv
      └── dset_meta
          ├── description_ace.csv
          ├── description_rams.csv
          └── description_wikievent.csv
```

## Run

Run the following command to train a TabEAE model.
```bash
bash ./scripts/train_{ace|rams|wikievent|mlee}.sh
```
Folders will be created automatically to store: 

1. Subfolder `checkpoint`: model parameters with best dev set result
2. File `log.txt`: recording hyper-parameters, training process and evaluation result
3. File `best_dev_results.log`/`best_test_related_results.log`: showing prediction results of checkpoints on every sample in dev/test set.

You could see hyperparameter settings in `./scripts/train_[dataset].sh` and `config_parser.py`. We give most of hyperparameters a brief explanation in `config_parser.py`.

After training, run the following command to infer with the TabEAE model and reproduce the reported results.
```bash
bash ./scripts/infer_{ace|rams|wikievent|mlee}.sh
```

The extraction results will go into ./Infer/{ace|rams|wikievent|mlee}

## Acknowledgement
Part of the code is borrowed from the [PAIE repository](https://github.com/mayubo2333/PAIE).

## Cite
If you find our work helpful, please consider citing our paper.
```bash
@inproceedings{he-etal-2023-revisiting,
    title = "Revisiting Event Argument Extraction: Can {EAE} Models Learn Better When Being Aware of Event Co-occurrences?",
    author = "He, Yuxin  and
      Hu, Jingyue  and
      Tang, Buzhou",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.701",
    pages = "12542--12556",
}
```bash
