import torch.nn as nn
import logging
from torch.utils.data import DataLoader, RandomSampler
logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        processor=None
    ):

        self.cfg = cfg
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self._init_metric()

        self.processor = processor

    def _init_metric(self):
        self.metric = {
            "global_steps": 0,
            "smooth_loss": 0.0,
        }


    def write_log(self):
        logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
        logger.info('lr: {}'.format(self.scheduler.get_last_lr()[0]))
        logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
        self.metric['smooth_loss'] = 0.0


    def train_one_step(self):
        self.model.train()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            if self.processor is not None:
                # re-generate training dataset
                print('re-generate training dataset')
                features = self.processor.convert_examples_to_features(self.examples, 'train', self.cfg.marker_range)
                dataset = self.processor.convert_features_to_dataset(features)
                dataset_sampler = RandomSampler(dataset)
                self.dataloader = DataLoader(dataset, sampler=dataset_sampler, batch_size=self.cfg.batch_size, collate_fn=self.processor.collate_fn)

            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        inputs = self.convert_batch_to_inputs(batch)
        loss, _= self.model(**inputs)

        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()

        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        
        self.metric['smooth_loss'] += loss.item()/self.cfg.logging_steps
        if (self.metric['global_steps']+1)%self.cfg.gradient_accumulation_steps==0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.metric['global_steps'] += 1


    def convert_batch_to_inputs(self, batch):
        raise NotImplementedError()


class Trainer(BaseTrainer):
    def __init__(self, cfg=None, data_loader=None, model=None, optimizer=None, scheduler=None, processor=None):
        super().__init__(cfg, data_loader, model, optimizer, scheduler, processor)

    def convert_batch_to_inputs(self, batch):
        inputs = {
            'enc_input_ids':            batch[0].to(self.cfg.device), 
            'enc_mask_ids':             batch[1].to(self.cfg.device), 
            'dec_table_ids':            batch[2].to(self.cfg.device),
            'dec_table_attention_mask': batch[3].to(self.cfg.device),
            'dec_prompt_lens':          batch[4],
            'list_target_info':         batch[5], 
            'old_tok_to_new_tok_indexs':batch[6],
            'list_arg_slots':           batch[7],
            'list_roles':               batch[8],
            'trigger_enc_token_index':  batch[11],
            'list_arg_2_prompt_slots':  batch[12],
            'cum_event_nums_per_type':  batch[13],
            'list_dec_prompt_ids':      batch[14].to(self.cfg.device),
            'list_len_prompt_ids':      batch[15].to(self.cfg.device),
        }

        return inputs
