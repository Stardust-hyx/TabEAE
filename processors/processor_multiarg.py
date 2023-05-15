import os
import re
import sys
sys.path.append("../")
import torch
import numpy as np
from copy import deepcopy
from itertools import chain

from torch.utils.data import Dataset
from processors.processor_base import DSET_processor
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, list_event_type, list_event_trigger,
                enc_text, enc_input_ids, enc_mask_ids,
                trigger_enc_token_index,
                dec_table_ids, dec_table_attention_mask, dec_prompt_lens,
                list_arg_slots, list_target_info,
                old_tok_to_new_tok_index, full_text, list_roles,
                list_arg_2_prompt_slots, cum_event_nums_per_type,
                list_dec_prompt_ids
        ):

        self.example_id = example_id
        self.feature_id = feature_id
        self.list_event_type = list_event_type
        self.list_event_trigger = list_event_trigger
        self.num_events = len(list_event_trigger)
        
        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.trigger_enc_token_index = trigger_enc_token_index

        self.dec_table_ids = dec_table_ids
        self.dec_table_attention_mask = dec_table_attention_mask
        self.dec_prompt_lens = dec_prompt_lens
        
        self.list_arg_slots = list_arg_slots
        self.list_target_info = list_target_info

        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index
        self.full_text = full_text
        self.list_roles = list_roles
        self.list_arg_2_prompt_slots = list_arg_2_prompt_slots
        self.cum_event_nums_per_type = cum_event_nums_per_type

        self.list_dec_prompt_ids = list_dec_prompt_ids

    def init_pred(self):
        self.pred_dict_tok = [dict() for _ in range(self.num_events)]
        self.pred_dict_word = [dict() for _ in range(self.num_events)]
    
    def add_pred(self, role, span, event_index):
        pred_dict_tok = self.pred_dict_tok[event_index]
        pred_dict_word = self.pred_dict_word[event_index]
        if role not in pred_dict_tok:
            pred_dict_tok[role] = list()
        if span not in pred_dict_tok[role]:
            pred_dict_tok[role].append(span)

            if span != (0, 0):
                if role not in pred_dict_word:
                    pred_dict_word[role] = list()
                word_span = self.get_word_span(span)         # convert token span to word span 
                if word_span not in pred_dict_word[role]:
                    pred_dict_word[role].append(word_span)

    def set_gt(self):
        self.gt_dict_tok = [dict() for _ in range(self.num_events)]
        for i, target_info in enumerate(self.list_target_info):
            for k, v in target_info.items():
                self.gt_dict_tok[i][k] = [(s,e) for (s,e) in zip(v["span_s"], v["span_e"])]

        self.gt_dict_word = [dict() for _ in range(self.num_events)]
        for i, gt_dict_tok in enumerate(self.gt_dict_tok):
            gt_dict_word = self.gt_dict_word[i]
            for role, spans in gt_dict_tok.items():
                for span in spans:
                    if span != (0, 0):
                        if role not in gt_dict_word:
                            gt_dict_word[role] = list()
                        word_span = self.get_word_span(span)
                        gt_dict_word[role].append(word_span)

    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id 
        return new_tok_index_to_old_tok_index

    def get_word_span(self, span):
        """
        Given features with gt/pred token-spans, output gt/pred word-spans
        """
        if span==(0, 0):
            raise AssertionError()
        # offset = 0 if dset_type=='ace_eeqa' else self.event_trigger[2]
        offset = 0
        span = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1]-1, min(self.old_tok_index.keys()))

        while span[0] not in self.old_tok_index:
            span[0] += 1 
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1 
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_types: {}\n".format(self.list_event_type)
        s += "trigger_words: {}\n".format(self.list_event_trigger)

        s += "dec_table_ids: {}\n".format(self.dec_table_ids)
        s += "dec_table_attention_mask:\n"
        for line in self.dec_table_attention_mask[:150, :150].tolist():
            s += " {}\n".format(line)
        s += "list_arg_2_prompt_slots: {}\n".format(self.list_arg_2_prompt_slots)
        s += "list_arg_slots:\n{}\n".format(self.list_arg_slots)
        s += "list_roles:\n{}\n".format(self.list_roles)

        return s

class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
 
    @staticmethod
    def collate_fn(batch):
        
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])
        dec_table_ids = torch.tensor([f.dec_table_ids for f in batch])
        dec_table_attention_mask = torch.stack([f.dec_table_attention_mask for f in batch])
        dec_prompt_lens = [f.dec_prompt_lens for f in batch]

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])
        
        list_target_info = [f.list_target_info for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        list_arg_slots = [f.list_arg_slots for f in batch]
        trigger_enc_token_index = [f.trigger_enc_token_index for f in batch]
        list_roles = [f.list_roles for f in batch]
        list_arg_2_prompt_slots = [f.list_arg_2_prompt_slots for f in batch]
        cum_event_nums_per_type = [f.cum_event_nums_per_type for f in batch]

        list_dec_prompt_ids = list(chain(*[f.list_dec_prompt_ids for f in batch]))
        list_len_prompt_ids = [len(x) for x in list_dec_prompt_ids]
        max_batch_len = max(list_len_prompt_ids)
        list_dec_prompt_ids = torch.LongTensor([ids + [1 for _ in range(max_batch_len - len(ids))]
                                                for ids in list_dec_prompt_ids])
        list_len_prompt_ids = torch.LongTensor(list_len_prompt_ids)


        return enc_input_ids, enc_mask_ids, \
                dec_table_ids, dec_table_attention_mask, dec_prompt_lens, \
                list_target_info, old_tok_to_new_tok_index, list_arg_slots, list_roles, \
                example_idx, feature_idx, \
                trigger_enc_token_index, \
                list_arg_2_prompt_slots, cum_event_nums_per_type, \
                list_dec_prompt_ids, list_len_prompt_ids


class MultiargProcessor(DSET_processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer) 
        self.set_dec_input()
        self.collate_fn = ArgumentExtractionDataset.collate_fn
        

    def set_dec_input(self):
        self.arg_query=False
        self.prompt_query=True
    
    @staticmethod
    def _read_prompt_group(prompt_path):
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts


    def create_dec_qury(self, arg, event_trigger):
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
                
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]

        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)

        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1

        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e


    def convert_examples_to_features(self, examples, role_name_mapping=None):
        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        features = []

        for example in examples:
            example_id = example.doc_id
            context = example.context 
            event_type_2_events = example.event_type_2_events

            list_event_type = []
            triggers = []
            for event_type, events in event_type_2_events.items():
                list_event_type += [e['event_type'] for e in events]
                triggers += [tuple(e['trigger']) for e in events]
                 
            set_triggers = list(set(triggers))
            set_triggers = sorted(set_triggers)

            trigger_overlap = False
            for t1 in set_triggers:
                for t2 in set_triggers:
                    if t1[0]==t2[0] and t1[1]==t2[1]:
                        continue
                    if (t1[0] < t2[1] and t2[0] < t1[1]) or (t2[0] < t1[1] and t1[0] < t2[1]):
                        trigger_overlap = True
                        break
            if trigger_overlap:
                print('[trigger_overlap]', event_type_2_events)
                exit(0)

            # NOTE: extend trigger full info in features
            offset = 0
            marked_context = deepcopy(context)
            marker_indice = list(range(len(triggers)))
            for i, t in enumerate(set_triggers):
                t_start = t[0]; t_end = t[1]
                marked_context = marked_context[:(t_start+offset)] + ['<t-%d>' % marker_indice[i]] + \
                                context[t_start: t_end] + ['</t-%d>' % marker_indice[i]] + context[t_end:]
                offset += 2
            enc_text = " ".join(marked_context)

            # change the mapping to idx2tuple (start/end word idx)
            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in marked_context:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr+len(tok)-1]) # exact word start char and end char index
                curr += len(tok)+1

            enc = self.tokenizer(enc_text, add_special_tokens=True)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)
            
            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                new_tok = [new_tok_s, new_tok_e]
                # print(new_tok)
                old_tok_to_new_tok_index.append(new_tok)

            trigger_enc_token_index = []
            for t in triggers:
                t_start = t[0]; t_end = t[1]
                new_t_start = old_tok_to_new_tok_index[t_start][0]
                new_t_end = old_tok_to_new_tok_index[t_end - 1][1]
                trigger_enc_token_index.append([new_t_start, new_t_end])

            # print(example_id)
            # print(marked_context)
            # print(enc_text)
            # print(enc)
            # enc_tokens = Tokenizer.convert_ids_to_tokens(enc_input_ids)
            # enc_tokens_ = dict()
            # for i, token in enumerate(enc_tokens):
            #     enc_tokens_[i] = token
            # print(enc_tokens_)
            # print(trigger_enc_token_index)

            dec_table_ids = []

            """ Deal with prompt template """
            list_arg_2_prompt_slots = []
            list_num_prompt_slots = []
            list_dec_prompt_ids = []
            list_arg_2_prompt_slot_spans = []
            for i, event_type in enumerate(event_type_2_events):
                dec_prompt_text = prompts[event_type].strip()
                assert dec_prompt_text
                dec_prompt = self.tokenizer(dec_prompt_text, add_special_tokens=True)
                dec_prompt_ids = dec_prompt["input_ids"]
                
                arg_list = self.argument_dict[event_type.replace(':', '.')] 
                arg_2_prompt_slots = dict()
                arg_2_prompt_slot_spans = dict()
                num_prompt_slots = 0
                if os.environ.get("DEBUG", False): arg_set=set()
                for arg in arg_list:
                    prompt_slots = {
                        "tok_s":list(), "tok_e":list(),
                    }
                    prompt_slot_spans = []
                    
                    if role_name_mapping is not None:
                        arg_ = role_name_mapping[event_type][arg]
                    else:
                        arg_ = arg
                    # Using this more accurate regular expression might further improve rams results
                    for matching_result in re.finditer(r'\b'+re.escape(arg_)+r'\b', dec_prompt_text.split('.')[0]): 
                        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slot_spans.append((tok_prompt_s, tok_prompt_e))
                        tok_prompt_s += len(dec_table_ids)
                        tok_prompt_e += len(dec_table_ids)
                        prompt_slots["tok_s"].append(tok_prompt_s); prompt_slots["tok_e"].append(tok_prompt_e)
                        num_prompt_slots += 1

                    arg_2_prompt_slots[arg] = prompt_slots
                    arg_2_prompt_slot_spans[arg] = prompt_slot_spans

                dec_table_ids += dec_prompt_ids
                list_arg_2_prompt_slots.append(arg_2_prompt_slots)
                list_num_prompt_slots.append(num_prompt_slots)
                list_dec_prompt_ids.append(dec_prompt_ids)
                list_arg_2_prompt_slot_spans.append(arg_2_prompt_slot_spans)

            dec_prompt_lens = len(dec_table_ids)
            
            row_index = 0
            list_trigger_pos = []
            list_arg_slots = []
            list_target_info = []
            list_roles = []
            """ Deal with target arguments """
            for i, (event_type, events) in enumerate(event_type_2_events.items()):
                arg_2_prompt_slots = list_arg_2_prompt_slots[i]
                num_prompt_slots = list_num_prompt_slots[i]
                dec_prompt_ids = list_dec_prompt_ids[i]
                arg_2_prompt_slot_spans = list_arg_2_prompt_slot_spans[i]
                for event in events:
                    row_index += 1
                    dec_event_ids = [self.tokenizer.mask_token_id] * (1 + num_prompt_slots)   # 1 is for the place holder of event trigger

                    list_trigger_pos.append(len(dec_table_ids))

                    arg_slots = []
                    cursor = len(dec_table_ids) + 1
                    event_args = event['args']

                    arg_set = set([tuple(arg[:2]) for arg in event_args])

                    event_args_name = [arg[-1] for arg in event_args]
                    target_info = dict()
                    for arg, prompt_slots in arg_2_prompt_slots.items():
                        num_slots = len(prompt_slots['tok_s'])
                        arg_slots.append([cursor + x for x in range(num_slots)])
                        cursor += num_slots

                        arg_target = {"text": list(), "span_s": list(), "span_e": list()}
                        answer_texts, start_positions, end_positions = list(), list(), list()
                        if arg in event_args_name:
                            # Deal with multi-occurance
                            if os.environ.get("DEBUG", False): arg_set.add(arg)
                            arg_idxs = [j for j, x in enumerate(event_args_name) if x == arg]
                            if os.environ.get("DEBUG", False): counter[0] += 1; counter[1]+=len(arg_idxs)

                            for arg_idx in arg_idxs:
                                event_arg_info = event_args[arg_idx]
                                answer_text = event_arg_info[2]; answer_texts.append(answer_text)
                                start_old, end_old = event_arg_info[0], event_arg_info[1]
                                start_position = old_tok_to_new_tok_index[start_old][0]; start_positions.append(start_position)
                                end_position = old_tok_to_new_tok_index[end_old-1][1]; end_positions.append(end_position)

                        arg_target["text"] = answer_texts
                        arg_target["span_s"]= start_positions
                        arg_target["span_e"] = end_positions
                        target_info[arg] = arg_target

                    assert sum([len(slots) for slots in arg_slots]) == num_prompt_slots
                    
                    dec_table_ids += dec_event_ids
                    list_arg_slots.append(arg_slots)
                    list_target_info.append(target_info)
                    roles = self.argument_dict[event_type.replace(':', '.')]
                    assert len(roles) == len(arg_slots)
                    list_roles.append(roles)

            max_dec_seq_len = self.args.max_dec_seq_length
            assert len(dec_table_ids)<=max_dec_seq_len, f"\n{example.doc_id}\n{dec_table_ids}"
            while len(dec_table_ids) < max_dec_seq_len:
                dec_table_ids.append(self.tokenizer.pad_token_id)

            assert len(list_trigger_pos) == len(list_arg_slots) == len(list_target_info)

            """ Stucture-aware Attention Mask """
            dec_table_attention_mask = torch.zeros((max_dec_seq_len, max_dec_seq_len), dtype=torch.int64)
            # prompt ~ prompt
            dec_table_attention_mask[:dec_prompt_lens, :dec_prompt_lens] = 1

            event_nums_per_type = [len(events) for events in event_type_2_events.values()]
            cum_event_nums_per_type = np.cumsum(event_nums_per_type)
            cusor = 0
            for i, (arg_2_prompt_slots, dec_prompt_ids) in enumerate(zip(list_arg_2_prompt_slots, list_dec_prompt_ids)):
                event_index_start = cum_event_nums_per_type[i-1] if i > 0 else 0
                event_index_end = cum_event_nums_per_type[i]

                arg_slots = list_arg_slots[event_index_start: event_index_end]
                # print(arg_slots[0])
                # print()
                assert len(arg_slots[0]) == len(arg_2_prompt_slots)
                for j, prompt_slots in enumerate(arg_2_prompt_slots.values()):
                    arg_slots_same_role = [arg_slot[j] for arg_slot in arg_slots]
                    for k, (start, end) in enumerate(zip(prompt_slots['tok_s'], prompt_slots['tok_e'])):
                        arg_slots_same_cloumn = [arg_slot[k] for arg_slot in arg_slots_same_role]
                        # prompt_slots -> arg_slots
                        dec_table_attention_mask[start: end, arg_slots_same_cloumn] = 1
                        # arg_slots <- prompt_slots 
                        dec_table_attention_mask[arg_slots_same_cloumn, start: end] = 1
                        # # arg_slots ~ arg_slots in the same cloumn
                        # for pos in arg_slots_same_cloumn:
                        #     dec_table_attention_mask[pos, arg_slots_same_cloumn] = 1

                len_prompt = len(dec_prompt_ids)
                list_trigger_pos_ = list_trigger_pos[event_index_start: event_index_end]
                # prompt -> triggers
                dec_table_attention_mask[cusor: cusor + len_prompt, list_trigger_pos_] = 1
                # # triggers <- prompt
                # dec_table_attention_mask[list_trigger_pos_, cusor: cusor + len_prompt] = 1
                cusor += len_prompt

            # triggers ~ triggers
            for trigger_pos in list_trigger_pos:
                dec_table_attention_mask[trigger_pos, list_trigger_pos] = 1
            
            for i, trigger_pos in enumerate(list_trigger_pos):
                arg_slots = list_arg_slots[i]
                num_arg_slots = sum([len(slots) for slots in arg_slots])
                # triggers ~ arg_slots
                dec_table_attention_mask[trigger_pos, trigger_pos+1: trigger_pos+1+num_arg_slots] = 1
                dec_table_attention_mask[trigger_pos+1: trigger_pos+1+num_arg_slots, trigger_pos] = 1
                # # arg_slot itself
                # for j in range(num_arg_slots):
                #     dec_table_attention_mask[trigger_pos+1+j, trigger_pos+1+j] = 1
                # arg_slots ~ arg_slots in the same row
                dec_table_attention_mask[trigger_pos+1: trigger_pos+1+num_arg_slots, trigger_pos+1: trigger_pos+1+num_arg_slots] = 1

            feature_idx = len(features)
            feature = InputFeatures(example_id, feature_idx, list_event_type, triggers,
                                enc_text, enc_input_ids, enc_mask_ids,
                                trigger_enc_token_index,
                                dec_table_ids, dec_table_attention_mask, dec_prompt_lens,
                                list_arg_slots, list_target_info,
                                old_tok_to_new_tok_index, example.context, list_roles,
                                list_arg_2_prompt_slots, cum_event_nums_per_type,
                                list_dec_prompt_ids
                    )
            features.append(feature)
            # if len(feature.list_event_trigger) > 3 and len(event_type_2_events) == 2:
            #     print(example_id)
            #     print(context)
            #     print(marked_context)
            #     print(enc)
            #     enc_tokens = Tokenizer.convert_ids_to_tokens(enc_input_ids)
            #     enc_tokens_ = dict()
            #     for i, token in enumerate(enc_tokens):
            #         enc_tokens_[i] = token
            #     print(enc_tokens_)
            #     print(trigger_enc_token_index)
            #     print(event_type_2_events)
            #     print(cum_event_nums_per_type)
            #     print(feature)
            #     exit(0)

        if os.environ.get("DEBUG", False): print('\033[91m'+f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"+'\033[0m')
        return features

    
    def convert_features_to_dataset(self, features):
        dataset = ArgumentExtractionDataset(features)
        return dataset
