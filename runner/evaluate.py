import torch
import logging
logger = logging.getLogger(__name__)

from utils import get_best_indexes, get_best_index


class BaseEvaluator:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        metric_fn_dict=None,
    ):

        self.cfg = cfg
        self.eval_loader = data_loader
        self.model = model
        self.metric_fn_dict = metric_fn_dict

    
    def _init_metric(self):
        self.metric_val_dict = {metric:None for metric in self.metric_fn_dict}


    def calculate_one_batch(self, batch):
        inputs, named_v = self.convert_batch_to_inputs(batch)
        with torch.no_grad():
            _, outputs_lists = self.model(**inputs)
        return outputs_lists, named_v


    def evaluate_one_batch(self, batch):
        outputs_lists, named_v = self.calculate_one_batch(batch)
        self.collect_fn(outputs_lists, named_v, batch)


    def evaluate(self):
        self.model.eval()
        self.build_and_clean_record()
        self._init_metric()
        for batch in self.eval_loader:
            self.evaluate_one_batch(batch)
        output = self.predict()
        return output


    def build_and_clean_record(self):
        raise NotImplementedError()


    def collect_fn(self, outputs_list, named_v, batch):
        raise NotImplementedError()

     
    def convert_batch_to_inputs(self, batch):
        return NotImplementedError()


    def predict(self):
        raise NotImplementedError()


class Evaluator(BaseEvaluator):
    def __init__(
        self, 
        cfg=None, 
        data_loader=None, 
        model=None, 
        metric_fn_dict=None,
        features=None,
        set_type=None,
        invalid_num=0,
    ):
        super().__init__(cfg, data_loader, model, metric_fn_dict)
        self.features = features
        self.set_type = set_type
        self.invalid_num = invalid_num

    
    def convert_batch_to_inputs(self, batch):
        inputs = {
            'enc_input_ids':            batch[0].to(self.cfg.device), 
            'enc_mask_ids':             batch[1].to(self.cfg.device), 
            'dec_table_ids':            batch[2].to(self.cfg.device),
            'dec_table_attention_mask': batch[3].to(self.cfg.device),
            'dec_prompt_lens':          batch[4],
            'list_target_info':         None,
            'old_tok_to_new_tok_indexs':batch[6],
            'list_arg_slots':           batch[7],
            'list_roles':               batch[8],
            'trigger_enc_token_index':  batch[11],
            'list_arg_2_prompt_slots':  batch[12],
            'cum_event_nums_per_type':  batch[13],
            'list_dec_prompt_ids':      batch[14].to(self.cfg.device),
            'list_len_prompt_ids':      batch[15].to(self.cfg.device),
        }

        named_v = {
            "list_roles":                batch[8],
            "feature_ids":              batch[10],
        }
        return inputs, named_v


    def build_and_clean_record(self):
        self.record = {
            'feature_id_list': list(),
            "event_index_list": list(),
            "role_list": list(),
            "full_start_logit_list": list(),
            "full_end_logit_list": list()
        }


    def collect_fn(self, outputs_lists, named_v, batch):   
        bs = len(batch[0])
        for i in range(bs):
            predictions = outputs_lists[i]
            feature_id = named_v["feature_ids"][i].item()
            list_roles = named_v["list_roles"][i]
            for j, (prediction, roles) in enumerate(zip(predictions, list_roles)):
                for arg_role in roles:
                    [start_logits_list, end_logits_list] = prediction[arg_role] # NOTE base model should also has these kind of output
                    for (start_logit, end_logit) in zip(start_logits_list, end_logits_list):
                        self.record["feature_id_list"].append(feature_id)
                        self.record["event_index_list"].append(j)
                        self.record["role_list"].append(arg_role)
                        self.record["full_start_logit_list"].append(start_logit)
                        self.record["full_end_logit_list"].append(end_logit)

    
    def predict(self):
        for feature in self.features:
            feature.init_pred()
            feature.set_gt()

        pred_list = []
        for s in range(0, len(self.record["full_start_logit_list"]), self.cfg.infer_batch_size):
            sub_max_locs, cal_time, mask_time, score_time = get_best_indexes(self.features, self.record["feature_id_list"][s:s+self.cfg.infer_batch_size], \
                self.record["full_start_logit_list"][s:s+self.cfg.infer_batch_size], self.record["full_end_logit_list"][s:s+self.cfg.infer_batch_size], self.cfg)
            pred_list.extend(sub_max_locs)
        for (pred, feature_id, event_index, role) in zip(pred_list, self.record["feature_id_list"], self.record["event_index_list"], self.record["role_list"]):
            pred_span = (pred[0].item(), pred[1].item())
            feature = self.features[feature_id]
            feature.add_pred(role, pred_span, event_index)

        for metric, eval_fn in self.metric_fn_dict.items():
            perf_c, perf_i = eval_fn(self.features, self.invalid_num)
            self.metric_val_dict[metric] = (perf_c, perf_i)
            logger.info('{}-Classification. {} ({}): R {} P {} F {}'.format(
                metric, self.set_type, perf_c['gt_num'], perf_c['recall'], perf_c['precision'], perf_c['f1']))
            logger.info('{}-Identification. {} ({}): R {} P {} F {}'.format(
                metric, self.set_type, perf_i['gt_num'], perf_i['recall'], perf_i['precision'], perf_i['f1']))

        return self.metric_val_dict['span']