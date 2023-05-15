import os
import copy
import jsonlines
import argparse
from itertools import chain

def find_list(sub, full):
    sub_len = len(sub)
    full_len = len(full)
    pos = -1
    for i in range(full_len):
        if full[i: i+sub_len] == sub:
            pos = i
            break

    return pos

def overlap(span1, span2):
    return ((span1[1] >= span2[0] and span1[0] <= span2[1]) or
            (span2[1] >= span1[0] and span2[0] <= span1[1]))

total_arg = 0
total_events = 0
total_instance = 0

def split(in_file, out_file, max_len=250):
    global total_arg
    global total_events
    global total_instance

    num_event_cnter = dict()
    reader = jsonlines.open(in_file)
    writer = jsonlines.open(out_file, mode='w')
    for line in reader:
        entity_dict = {entity['id']:entity for entity in line['entity_mentions']}
        if not line["event_mentions"]:
            continue
        doc_key = line["doc_id"]
        full_text = line['tokens']
        doc_length = len(full_text)
        # print(doc_key)
        
        sents = [[s[0] for s in sent[0]] for sent in line['sentences']]
        sent_lens = [len(sent) for sent in sents]
        assert sum(sent_lens) == doc_length

        events = []
        event_tuples = []
        for i, event in enumerate(line["event_mentions"]):
            event_type = event['event_type']
            trigger = event['trigger']
            event_span = [trigger['sent_idx'], trigger['sent_idx']]
            trigger_sent_idx = trigger['sent_idx']
            trigger = [trigger['start'], trigger['end'], trigger['text']]
            
            args = []
            for arg_info in event['arguments']:
                arg_entity = entity_dict[arg_info['entity_id']]
                arg_sent_idx = arg_entity['sent_idx']
                arg = [arg_entity['start'], arg_entity['end'], arg_info['text'], arg_info['role']]
                args.append(arg)

                event_span = [min(event_span[0], arg_sent_idx), max(event_span[1], arg_sent_idx)]

            event = {'event_type': event_type, 'trigger': trigger, 'args': args}
            events.append(event)

            event_tuple = ([i], event_span)
            event_tuples.append(event_tuple)

        # print(sorted(event_tuples, key=lambda x: x[1]))

        # merging
        num_event = len(event_tuples)
        tmp_tuples = copy.deepcopy(event_tuples)
        merged_indexs = []
        have_merged = True
        while have_merged:
            have_merged = False
            for i in range(num_event):
                if i in merged_indexs:
                    continue
                tuple1 = tmp_tuples[i]
                span1 = tuple1[-1]

                for j in range(i+1, num_event):
                    if j in merged_indexs:
                        continue
                    tuple2 = tmp_tuples[j]
                    span2 = tuple2[-1]
                    
                    if overlap(span1, span2):
                        merge_span = [min(span1[0], span2[0]), max(span1[1], span2[1])]
                        merge_tuple = (tuple1[0] + tuple2[0], merge_span)
                        # if merge_span != -1:
                        tmp_tuples[i] = merge_tuple
                        tuple1 = merge_tuple
                        merged_indexs.append(j)
                        have_merged = True
        
        new_tuples = [tmp_tuples[i] for i in range(num_event) if i not in merged_indexs]
        # print(sorted(new_tuples, key=lambda x: x[1]))
        # print()

        for i, (indice, merge_span) in enumerate(new_tuples):
            start_sent_index = merge_span[0]
            end_sent_index = merge_span[1]
            context_len = sum(sent_lens[start_sent_index: end_sent_index+1])

            if context_len > max_len:
                print('[Exceed max len limit]: %s\t%d' % (doc_key, context_len))
                # print(sorted(event_tuples, key=lambda x: x[1]))
                # print(sorted(new_tuples, key=lambda x: x[1]))
                # print(sent_lens)
                # print(indice, merge_span, context_len)
                # for sent_id in range(start_sent_index, end_sent_index):
                #     print(sent_lens[sent_id])

            # expand the context by appending neighbor sents until exceeding the max len limit
            flag_expand_front = True
            flag_expand_back = True
            while flag_expand_front or flag_expand_back:
                index_front = start_sent_index - 1
                index_back = end_sent_index + 1
                
                if index_front >= 0:
                    if (context_len + sent_lens[index_front]) <= max_len:
                        start_sent_index = index_front
                        context_len += sent_lens[index_front]
                    else:
                        flag_expand_front = False
                else:
                    flag_expand_front = False

                if index_back < len(sents):
                    if (context_len + sent_lens[index_back]) <= max_len:
                        end_sent_index = index_back
                        context_len += sent_lens[index_back]
                    else:
                        flag_expand_back = False
                else:
                    flag_expand_back = False

            context = list(chain(*sents[start_sent_index : end_sent_index+1]))
            assert find_list(context, full_text) >= 0
            
            new_events = []
            for index in indice:
                event = events[index]

                if start_sent_index > 0:
                    offset = sum(sent_lens[0: start_sent_index])
                    trigger = event['trigger']
                    args = event['args']
                    trigger[0] -= offset
                    trigger[1] -= offset
                    for k, arg in enumerate(args):
                        args[k][0] -= offset
                        args[k][1] -= offset
                    event['trigger'] = trigger
                    event['args'] = args

                new_events.append(event)

                total_arg += len(event['args'])
                total_events += 1
            total_instance += 1
            
            sample = {'id': '%d-%s' % (i, doc_key), 'context': context, 'events': new_events, 'len(context)': len(context)}
            writer.write(sample)

            num_event = len(new_events)
            if num_event not in num_event_cnter:
                num_event_cnter[num_event] = 1
            else:
                num_event_cnter[num_event] += 1

    print(num_event_cnter)
    num_events = 0
    for k, v in num_event_cnter.items():
        num_events += k * v
    print('[num_events]:', num_events)
    print()

    return num_event_cnter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='--indir')
    parser.add_argument('--outdir', type=str, required=True, help='--outdir')
    args = parser.parse_args()

    in_dir = getattr(args, 'indir')
    out_dir = getattr(args, 'outdir')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fnames = os.listdir(in_dir)
    list_num_event_cnter = []
    for fn in fnames:
        fpath = os.path.join(in_dir, fn)
        out_fpath = os.path.join(out_dir, fn)

        print('\n', fpath)
        num_event_cnter = split(fpath, out_fpath, max_len=250)
        list_num_event_cnter.append(num_event_cnter)

    print('[total_arg]', total_arg)
    print('[total_events]', total_events)
    print('[total_instance]', total_instance)
    print('[# args per event]', total_arg / total_events)
    print('[# event per context]', total_events / total_instance)
