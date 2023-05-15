import csv
import json
import ipdb
import jsonlines
import torch

from random import sample
from itertools import chain
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import MAX_NUM_EVENTS
import copy                             
import logging
logger = logging.getLogger(__name__)

class Events:
    def __init__(self, doc_id, context, event_type_2_events):
        self.doc_id = doc_id
        self.context = context
        self.event_type_2_events = event_type_2_events


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                 enc_text, dec_text,
                 enc_tokens, dec_tokens, 
                 old_tok_to_new_tok_index,  
                 event_type, event_trigger, argument_type,
                 enc_input_ids, enc_mask_ids, 
                 dec_input_ids, dec_mask_ids,
                 answer_text, start_position=None, end_position=None):

        self.example_id = example_id
        self.feature_id = feature_id
        self.enc_text = enc_text
        self.dec_text = dec_text
        self.enc_tokens = enc_tokens
        self.dec_tokens = dec_tokens
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.event_type = event_type
        self.event_trigger = event_trigger
        self.argument_type = argument_type
        
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.dec_input_ids = dec_input_ids
        self.dec_mask_ids = dec_mask_ids

        self.answer_text = answer_text
        self.start_position = start_position
        self.end_position = end_position


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "argument_type: {}\n".format(self.argument_type)
        s += "enc_tokens: {}\n".format(self.enc_tokens)
        s += "dec_tokens: {}\n".format(self.dec_tokens)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_input_ids: {}\n".format(self.dec_input_ids)
        s += "dec_mask_ids: {}\n".format(self.dec_mask_ids)
        
        s += "answer_text: {}\n".format(self.answer_text)
        s += "start_position: {}\n".format(self.start_position)
        s += "end_position: {}\n".format(self.end_position) 
        return s


class DSET_processor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.template_dict, self.argument_dict = self._read_roles(self.args.role_path)
        self.collate_fn = None


    def _read_jsonlines(self, input_file):
        lines = []
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                lines.append(obj)
        return lines


    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.load(f)


    def _read_roles(self, role_path):
        template_dict = {}
        role_dict = {}

        if 'MLEE' in role_path:
            with open(role_path) as f:
                role_name_mapping = json.load(f)
                for event_type, mapping in role_name_mapping.items():
                    roles = list(mapping.keys())
                    role_dict[event_type] = roles

            return None, role_dict

        with open(role_path, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                event_type_arg, template = line
                template_dict[event_type_arg] = template
                
                event_type, arg = event_type_arg.split('_')
                if event_type not in role_dict:
                    role_dict[event_type] = []
                role_dict[event_type].append(arg)

        return template_dict, role_dict

    def _create_example(self, lines, over_sample=None, max_num_event=MAX_NUM_EVENTS):
        W = self.args.window_size

        examples = []
        for line in lines:
            doc_id = line["id"]
            context = line['context']
            events = line["events"]
            num_events = len(events)
            if num_events < 1:
                print('[num_events < 1]', doc_id)
                continue

            events = sorted(events, key=lambda x: x['trigger'])

            context_length = len(context)
            if context_length > W:
                for event in events:
                    self.invalid_arg_num += len(event['args'])
                print('[context_length > W] %s\t\t%d' % (doc_id, context_length))
                continue

            if num_events > max_num_event:
                for event in events[max_num_event:]:
                    self.invalid_arg_num += len(event['args'])
                events = events[:max_num_event]
                print('[num_events > max_num_event] %s\t\t%d' % (doc_id, num_events))

            assert len(events) <= MAX_NUM_EVENTS

            if self.args.single:
                for event in events:
                    event_type = event['event_type']
                    event_type_2_events = {event_type: [event]}
                    examples.append(Events(doc_id, context, event_type_2_events))

                continue

            event_type_2_events = dict()
            for event in events:
                event_type = event['event_type']

                if event_type not in event_type_2_events:
                    event_type_2_events[event_type] = [event]
                else:
                    event_type_2_events[event_type].append(event)

            examples.append(Events(doc_id, context, event_type_2_events))

            if over_sample == 'double' and num_events > 1:
                examples.append(Events(doc_id, context, event_type_2_events))

            elif over_sample == 'power' and num_events > 1:
                power_set = []
                def dfs(tmp, n):
                    if len(tmp) > 1:
                        power_set.append(tmp[:])
                    for i in range(n, num_events):
                        tmp.append(events[i])
                        dfs(tmp, i+1)
                        tmp.pop()
                dfs([], 0)

                for i, events_ in enumerate(power_set):
                    event_type_2_events = dict()
                    for event in events_:
                        event_type = event['event_type']

                        if event_type not in event_type_2_events:
                            event_type_2_events[event_type] = [event]
                        else:
                            event_type_2_events[event_type].append(event)

                    examples.append(Events('%d-%s' % (i, doc_id), context, event_type_2_events))

        logger.info("{} examples collected. {} arguments dropped.".format(len(examples), self.invalid_arg_num))
        print("{} examples collected. {} arguments dropped.".format(len(examples), self.invalid_arg_num))

        return examples

    def create_example(self, file_path, set_type):
        self.invalid_arg_num = 0
        lines = self._read_jsonlines(file_path)
        if self.args.dataset_type=='ace_eeqa':
            return self._create_example(lines, over_sample=('power' if set_type=='train' else None), max_num_event=6)
        elif self.args.dataset_type=='rams':
            return self._create_example(lines, over_sample=('power' if set_type=='train' else None))
        elif self.args.dataset_type=='wikievent':
            return self._create_example(lines, over_sample=('double' if set_type=='train' else None))
        elif self.args.dataset_type=='MLEE':
            return self._create_example(lines, over_sample=None)
        else:
            raise NotImplementedError()
    
    def convert_examples_to_features(self, examples):
        features = []
        for (example_idx, example) in enumerate(examples):
            sent = example.sent  
            event_type = example.type
            event_args = example.args
            event_trigger = example.trigger['text']
            event_args_name = [arg['role'] for arg in event_args]
            enc_text = " ".join(sent)

            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                old_tok_to_char_index.append(curr)
                curr += len(tok)+1
            assert(len(old_tok_to_char_index)==len(sent))

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            enc_tokens = self.tokenizer.convert_ids_to_tokens(enc_input_ids)  
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for char_idx in old_tok_to_char_index:
                new_tok = enc.char_to_token(char_idx)
                old_tok_to_new_tok_index.append(new_tok)    
    
            for arg in self.argument_dict[event_type.replace(':', '.')]:
                dec_text = 'Argument ' + arg + ' in ' + event_trigger + ' event ?' + " "
                     
                dec = self.tokenizer(dec_text)
                dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]
                dec_tokens = self.tokenizer.convert_ids_to_tokens(dec_input_ids) 
                while len(dec_input_ids) < self.args.max_dec_seq_length:
                    dec_input_ids.append(self.tokenizer.pad_token_id)
                    dec_mask_ids.append(self.args.pad_mask_token)
        
                start_position, end_position, answer_text = None, None, None
                if arg in event_args_name:
                    arg_idx = event_args_name.index(arg)
                    event_arg_info = event_args[arg_idx]
                    answer_text = event_arg_info['text']
                    # index before BPE, plus 1 because having inserted start token
                    start_old, end_old = event_arg_info['start'], event_arg_info['end']
                    start_position = old_tok_to_new_tok_index[start_old]
                    end_position = old_tok_to_new_tok_index[end_old] if end_old<len(old_tok_to_new_tok_index) else old_tok_to_new_tok_index[-1]+1 
                else:
                    start_position, end_position = 0, 0
                    answer_text = "__ No answer __"

                feature_idx = len(features)
                features.append(
                      InputFeatures(example_idx, feature_idx, 
                                    enc_text, dec_text,
                                    enc_tokens, dec_tokens,
                                    old_tok_to_new_tok_index,
                                    event_type, event_trigger, arg,
                                    enc_input_ids, enc_mask_ids, 
                                    dec_input_ids, dec_mask_ids,
                                    answer_text, start_position, end_position
                                )
                )
        return features

    
    def convert_features_to_dataset(self, features):

        all_enc_input_ids = torch.tensor([f.enc_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_enc_mask_ids = torch.tensor([f.enc_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_input_ids = torch.tensor([f.dec_input_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        all_dec_mask_ids = torch.tensor([f.dec_mask_ids for f in features], \
            dtype=torch.long).to(self.args.device)
        
        all_start_positions = torch.tensor([f.start_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_end_positions = torch.tensor([f.end_position for f in features], \
            dtype=torch.long).to(self.args.device)
        all_example_idx = torch.tensor([f.example_id for f in features], \
            dtype=torch.long).to(self.args.device)
        all_feature_idx = torch.tensor([f.feature_id for f in features], \
            dtype=torch.long).to(self.args.device)

        dataset = TensorDataset(all_enc_input_ids, all_enc_mask_ids,
                                all_dec_input_ids, all_dec_mask_ids,
                                all_start_positions, all_end_positions,
                                all_example_idx, all_feature_idx,
                            )
        return dataset


    def generate_dataloader(self, set_type):
        assert (set_type in ['train', 'dev', 'test'])
        if set_type=='train':
            file_path = self.args.train_file
        elif set_type=='dev':
            file_path = self.args.dev_file
        else:
            file_path = self.args.test_file
        
        examples = self.create_example(file_path, set_type)
        if set_type=='train' and self.args.keep_ratio<1.0:
            sample_num = int(len(examples)*self.args.keep_ratio)
            examples = sample(examples, sample_num)
            logger.info("Few shot setting: keep ratio {}. Only {} training samples remained.".format(\
                self.args.keep_ratio, len(examples))
            )

        features = self.convert_examples_to_features(examples, self.args.role_name_mapping)
        dataset = self.convert_features_to_dataset(features)

        if set_type != 'train':
            dataset_sampler = SequentialSampler(dataset)
        else:
            dataset_sampler = RandomSampler(dataset)
        if self.collate_fn:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size, collate_fn=self.collate_fn)
        else:
            dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.args.batch_size)
        return examples, features, dataloader, self.invalid_arg_num