import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    'ace_eeqa':{
        "train_file": './data/ace_eeqa/data_final/train_convert.json',
        "dev_file": './data/ace_eeqa/data_final/dev_convert.json', 
        "test_file": './data/ace_eeqa/data_final/test_convert.json',
        "max_span_num_file": "./data/dset_meta/role_num_ace.json",
    },
    'rams':{
        "train_file": './data/RAMS_1.0/data_final/train.jsonlines',
        "dev_file": './data/RAMS_1.0/data_final/dev.jsonlines',
        "test_file": './data/RAMS_1.0/data_final/test.jsonlines',
        "max_span_num_file": "./data/dset_meta/role_num_rams.json",
    },
    "wikievent":{
        "train_file": './data/WikiEvent/data_final/train.jsonl',
        "dev_file": './data/WikiEvent/data_final/dev.jsonl',
        "test_file": './data/WikiEvent/data_final/test.jsonl',
        "max_span_num_file": "./data/dset_meta/role_num_wikievent.json",
    },
    "MLEE":{
        "train_file": './data/MLEE/data_final/train.json',
        "dev_file": './data/MLEE/data_final/train.json',
        "test_file": './data/MLEE/data_final/test.json',
        "role_name_mapping": './data/MLEE/MLEE_role_name_mapping.json',
    },
}


def build_processor(args, tokenizer):
    if args.dataset_type not in _DATASET_DIR: 
        raise NotImplementedError("Please use valid dataset name")
    args.train_file=_DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']

    args.role_name_mapping = None
    if args.dataset_type=="MLEE":
        with open(_DATASET_DIR[args.dataset_type]['role_name_mapping']) as f:
            args.role_name_mapping = json.load(f)

    processor = MultiargProcessor(args, tokenizer)
    return processor

