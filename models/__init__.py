import sys
sys.path.append("../")
import copy
import logging
logger = logging.getLogger(__name__)

from transformers import RobertaConfig, RobertaTokenizerFast
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from .seq_to_table import Seq2Table
from utils import EXTERNAL_TOKENS
from processors.processor_multiarg import MultiargProcessor


def build_model(args):
    config_class, model_class, tokenizer_class = (RobertaConfig, Seq2Table, RobertaTokenizerFast)
    if args.inference_only:
        config = config_class.from_pretrained(args.inference_model_path)
    else:
        config = config_class.from_pretrained(args.model_name_or_path)

    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.context_representation = args.context_representation

    # length
    config.max_enc_seq_length = args.max_enc_seq_length
    config.max_dec_seq_length= args.max_dec_seq_length
    config.max_prompt_seq_length=args.max_prompt_seq_length
    config.max_span_length = args.max_span_length

    config.bipartite = args.bipartite
    config.matching_method_train = args.matching_method_train

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    if args.inference_only:
        model = model_class.from_pretrained(args.inference_model_path, config=config,
                                            num_prompt_pos=args.num_prompt_pos, num_event_embed=args.num_event_embed)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config,
                                            num_prompt_pos=args.num_prompt_pos, num_event_embed=args.num_event_embed)
        model.reset()

    # Add trigger special tokens and continuous token (maybe in prompt)
    new_token_list = copy.deepcopy(EXTERNAL_TOKENS)
    prompts = MultiargProcessor._read_prompt_group(args.prompt_path)
    for event_type, prompt in prompts.items():
        token_list = prompt.split()
        for token in token_list:
            if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                new_token_list.append(token)
    tokenizer.add_tokens(new_token_list)   
    logger.info("Add tokens: {}".format(new_token_list))
    if args.bert:
        model.bert.resize_token_embeddings(len(tokenizer))
    else:
        model.roberta.resize_token_embeddings(len(tokenizer))

    if args.inference_only:
        optimizer, scheduler = None, None
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'layer_norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'crossattention' in n], 
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate * 1.5
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'crossattention' in n],
                'weight_decay': 0.0,
                'lr': args.learning_rate * 1.5
            },
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'crossattention' not in n], 
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'crossattention' not in n],
                'weight_decay': 0.0,
                'lr': args.learning_rate
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    return model, tokenizer, optimizer, scheduler