# paie model
from itertools import chain
import torch
import torch.nn as nn
from .modeling_roberta_ import RobertaModel_, RobertaPreTrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple, seq_len_to_mask


class Seq2Table(RobertaPreTrainedModel):
    def __init__(self, config, decode_layer_start=17, num_prompt_pos=0, num_event_embed=0):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel_(config, decode_layer_start=decode_layer_start)
        self.decode_layer_start = decode_layer_start
        self.dec_input_drop = nn.Dropout(0.1)
        self.w_prompt_start = nn.Parameter(torch.zeros(config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.zeros(config.hidden_size, ))

        self.num_prompt_pos = num_prompt_pos
        if self.num_prompt_pos > 0:
            self.event_type_embed = nn.Embedding(num_prompt_pos, config.hidden_size, _weight=torch.zeros(num_prompt_pos, config.hidden_size), padding_idx=0)

        self.num_event_embed = num_event_embed
        if self.num_event_embed > 0:
            self.event_embed = nn.Embedding(num_event_embed, config.hidden_size, _weight=torch.zeros(num_event_embed, config.hidden_size), padding_idx=0)

        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

    def reset(self):
        self.w_prompt_start = nn.Parameter(torch.rand(self.config.hidden_size, ))
        self.w_prompt_end = nn.Parameter(torch.rand(self.config.hidden_size, ))

        if self.num_prompt_pos > 0:
            self.roberta._init_weights(self.event_type_embed)

        if self.num_event_embed > 0:
            self.roberta._init_weights(self.event_embed)

    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        dec_table_ids=None,
        dec_table_attention_mask=None,
        dec_prompt_lens=None,
        trigger_enc_token_index=None,
        list_arg_slots=None,
        list_target_info=None,
        old_tok_to_new_tok_indexs=None,
        list_roles=None,
        list_arg_2_prompt_slots=None,
        cum_event_nums_per_type=None,
        list_dec_prompt_ids=None,
        list_len_prompt_ids=None
    ):
        """
        Args:
            multi args post calculation
        """
        enc_outputs = self.roberta(
            input_ids=enc_input_ids,
            attention_mask=enc_mask_ids,
            output_hidden_states=True,
            fully_encode=True
        ).hidden_states

        decoder_context = enc_outputs[self.decode_layer_start]
        if self.config.context_representation == 'decoder':
            context_outputs = enc_outputs[-1]
        else:
            context_outputs = decoder_context


        """ Transfer dec_table_ids into dec_table_embeds """
        input_shape = dec_table_ids.size()
        batch_size, table_seq_len = input_shape

        dec_table_embeds = torch.zeros((batch_size, table_seq_len, self.config.hidden_size),
                                        dtype=torch.float32, device=self.config.device)

        prompt_attention_mask = torch.zeros_like(list_dec_prompt_ids)
        for i, len_prompt_ids in enumerate(list_len_prompt_ids):
            prompt_attention_mask[i, :len_prompt_ids] = 1

        dec_prompt_embeds = self.roberta(
            input_ids=list_dec_prompt_ids,
            attention_mask=prompt_attention_mask,
            cross_attention=False
        ).last_hidden_state

        cusor = 0
        list_num_event_types = [len(x) for x in cum_event_nums_per_type]
        assert sum(list_num_event_types) == len(dec_prompt_embeds)
        for i, num_event_types in enumerate(list_num_event_types):
            assert sum(list_len_prompt_ids[cusor: cusor + num_event_types]) == dec_prompt_lens[i]
            cum_len = 0
            list_len_prompt_ids_ = list_len_prompt_ids[cusor: cusor + num_event_types]

            if self.num_prompt_pos > 0:
                pos = torch.arange(num_event_types, device=self.config.device)
                if self.training:
                    pos = torch.randperm(num_event_types, device=self.config.device)

            for j, len_prompt_ids in enumerate(list_len_prompt_ids_):
                dec_table_embeds[i, cum_len: cum_len + len_prompt_ids] = dec_prompt_embeds[cusor, :len_prompt_ids]
                if self.num_prompt_pos > 0:
                    dec_table_embeds[i, cum_len: cum_len + len_prompt_ids] += self.event_type_embed(pos[j])
                cum_len += len_prompt_ids
                cusor += 1

        # init arg slots' embeds with prompt slots' embeds
        for i, (list_arg_2_prompt_slots_, list_arg_slots_, cum_event_nums_per_type_) in \
            enumerate(zip(list_arg_2_prompt_slots, list_arg_slots, cum_event_nums_per_type)):
            # print(list_arg_2_prompt_slots_)
            # print(list_arg_slots_)
            # print(cum_event_nums_per_type_)
            # print()
            dec_table_embeds_ = dec_table_embeds[i].detach()
            for j, arg_2_prompt_slots in enumerate(list_arg_2_prompt_slots_):
                event_index_start = cum_event_nums_per_type_[j-1] if j > 0 else 0
                event_index_end = cum_event_nums_per_type_[j]
                arg_slots = list_arg_slots_[event_index_start: event_index_end]
                for k, prompt_slots in enumerate(arg_2_prompt_slots.values()):
                    arg_slots_same_role = [arg_slot[k] for arg_slot in arg_slots]
                    for s, (start, end) in enumerate(zip(prompt_slots['tok_s'], prompt_slots['tok_e'])):
                        prompt_slot_embed = dec_table_embeds_[start: end]
                        prompt_slot_embed = torch.mean(prompt_slot_embed, dim=0)
                        arg_slots_same_cloumn = [arg_slot[s] for arg_slot in arg_slots_same_role]
                        dec_table_embeds[i, arg_slots_same_cloumn] = prompt_slot_embed

        if self.num_event_embed > 0:
            pos = torch.arange(self.num_event_embed, device=self.config.device)
            if self.training:
                pos = torch.randperm(self.num_event_embed, device=self.config.device)

        for i, (encoder_output, trigger_index, list_arg_slots_) in \
            enumerate(zip(decoder_context, trigger_enc_token_index, list_arg_slots)):
            
            dec_trigger_index = [arg_slots[0][0] - 1 for arg_slots in list_arg_slots_]
            assert len(trigger_index) == len(dec_trigger_index)
            for j, (trigger_start, trigger_end) in enumerate(trigger_index):
                # copy triggers' representation
                dec_trigger_index_ = dec_trigger_index[j]
                trigger_embed = encoder_output[trigger_start: trigger_end]
                trigger_embed = torch.mean(trigger_embed, dim=0)
                dec_table_embeds[i, dec_trigger_index_] = trigger_embed
                if self.num_event_embed > 0:
                    dec_table_embeds[i, dec_trigger_index_] += self.event_embed(pos[j])

                # add markers' representation to arg_slots
                arg_slots = list_arg_slots_[j]
                arg_slots = list(chain(*arg_slots))
                dec_table_embeds[i, arg_slots] += (encoder_output[trigger_start-1] + encoder_output[trigger_end]) / 2
                dec_table_embeds[i, arg_slots] /= 2

        # dec_table_embeds = self.dec_input_drop(dec_table_embeds)

        decoder_table_outputs = self.roberta(
                inputs_embeds=dec_table_embeds,
                attention_mask=dec_table_attention_mask,
                encoder_hidden_states=decoder_context,
                encoder_attention_mask=enc_mask_ids,
                cross_attention=True,
        )
        decoder_table_outputs = decoder_table_outputs.last_hidden_state   #[bs, table_seq_len, H]

        logit_lists = list()
        total_loss = 0.
        for i, (context_output, decoder_table_output, list_arg_slots_, list_roles_, old_tok_to_new_tok_index) in \
            enumerate(zip(context_outputs, decoder_table_outputs, list_arg_slots, list_roles, old_tok_to_new_tok_indexs)):
            
            batch_loss = list()
            cnt = 0
            
            list_output = list()
            # iterate event by event
            for j, (arg_slots, roles) in enumerate(zip(list_arg_slots_, list_roles_)):
                if self.training:
                    target_info = list_target_info[i][j]

                output = dict()
                for (slots, arg_role) in zip(arg_slots, roles):
                    start_logits_list = list()
                    end_logits_list = list()
                    for slot in slots:
                        query_sub = decoder_table_output[slot].unsqueeze(0)
                        
                        start_query = (query_sub*self.w_prompt_start).unsqueeze(-1) # [1, H, 1]
                        end_query = (query_sub*self.w_prompt_end).unsqueeze(-1)     # [1, H, 1]

                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()  
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)
                    
                    output[arg_role] = [start_logits_list, end_logits_list]

                    if self.training:
                        # calculate loss
                        target = target_info[arg_role] # "arg_role": {"text": ,"span_s": ,"span_e": }
                        predicted_spans = list()
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index, self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()

                        target_spans = [[s,e] for (s,e) in zip(target["span_s"], target["span_e"])]
                        if len(target_spans)<len(predicted_spans):
                            # need to consider whether to make more 
                            pad_len = len(predicted_spans) - len(target_spans)
                            target_spans = target_spans + [[0,0]] * pad_len
                            target["span_s"] = target["span_s"] + [0] * pad_len
                            target["span_e"] = target["span_e"] + [0] * pad_len
                            
                        if self.config.bipartite:
                            idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                        else:
                            idx_preds = list(range(len(predicted_spans)))
                            idx_targets = list(range(len(target_spans)))
                            if len(idx_targets) > len(idx_preds):
                                idx_targets = idx_targets[0:len(idx_preds)]
                            idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                            idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                        cnt += len(idx_preds)
                        start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds], torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                        end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds], torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                        batch_loss.append((start_loss + end_loss)/2)

                list_output.append(output)

            logit_lists.append(list_output)
            if self.training: # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))/cnt
            
        if self.training:
            return total_loss/len(context_outputs), logit_lists
        else:
            return [], logit_lists
    