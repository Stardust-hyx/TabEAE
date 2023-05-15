from glob import glob 
from itertools import chain
import os, json
import argparse
import jsonlines
import copy
import numpy as np

def process_paie_format_single(txt, ann, output_path):
    output = {}
    corpus_text, event_ann = open(txt, 'r'), open(ann, 'r')
    context_list, sents_list, event_list = [], [], []
    corpus, annotation = corpus_text.readlines(), event_ann.readlines()
    #annotation.extend(event_ann2)

    for text_line in corpus:
        sent = []
        for token in text_line.strip('\n').split(' '):
            context_list.append(token)
            sent.append(token)
        sents_list.append(sent)

    trigger_list, event_rec = {}, {}
    

    for ann in annotation:
        if ann.startswith('E'):
            #print(ann)
            e_idx, event_info = ann.strip('\n').split('\t')
            #print(e_idx, event_info)
            #print(event_info)
            trg = event_info.split(' ')[0].split(':')[1]
            event_rec[e_idx] = trg

    for ann in annotation:
        event = {}
        skip = 0
        if ann.startswith('T'):
            trigger = ann.strip('\n').split('\t')
            #print(trigger)
            type, start, end, trg_text = trigger[1].split(' ')[0], int(trigger[1].split(' ')[1]), int(trigger[1].split(' ')[2]), trigger[-1]
            #trigger_list[trigger[0]] = [type, start, end, trg_text]
            #trigger_list[trigger[0]].append(trigger[-1])

            #reindex
            #start, end = trigger[1].split(' ')[1], trigger[1].split(' ')[2]
            #print(start, end, trg_text)
            lens, blanks, spans, new_s, new_e = 0, 0, end-start, -1, -1
            f = False
            for idx, citem in enumerate(context_list):
                if start != 0 or new_s == 0:
                    lens += (len(citem)+1)
                #if f == False:
                #    print(lens, start)
                #    f = True
                if lens == start:
                    new_s = idx if start==0 else idx+1
                    new_e = new_s
                    if start == 0:
                        lens += (len(citem)+1)
                    #print(citem, trg_text,new_s, new_e)
                if new_e != -1 and lens == end + 1:
                        new_e = idx + 1
 
                    #print(trg_text, context_list[new_s:new_e])
            if new_s == -1 or new_e == -1:
                skip += 1
            trigger_list[trigger[0]] = [type, new_s, new_e, trg_text]
            #print(trigger_list)
            #trigger_list[trigger[0]].append(trigger[-1])
            #lens = 
            #for jdx, citem in enumerate(context_list):
                
                
        
        if ann.startswith('E'):
            
            e_idx, event_info = ann.strip('\n').split('\t')
            #print(e_idx, event_info)
            #print(event_info)
            trg = event_info.split(' ')[0].split(':')[1]
            #event_rec[e_idx] = trg

            event['event_type'] = trigger_list[trg][0]
            event['trigger'] = [int(trigger_list[trg][1]), int(trigger_list[trg][2]), trigger_list[trg][3]]
            #print(context_list[event['trigger'][0]:event['trigger'][1]+1], event['trigger'][2])
            event['args'] = []
            #print(event_info)
            for arg_item in event_info.split(' ')[1:]:
                #print(event_info)
                role, trg_index = arg_item.split(':')
                if trg_index.startswith('E'):
                    trg_index = event_rec[trg_index]
                    
                arg_info = [int(trigger_list[trg_index][1]), int(trigger_list[trg_index][2]), trigger_list[trg_index][3], role]
                event['args'].append(arg_info)
                #print(event)
                
            event_list.append(event)
    #print(len(event_list))
    assert(skip==0)
    #print(skip)
                

    #print(trigger_list)
    #output['id'] = txt_file.split('/')[-1].split('-')[1].split('.')[0]
    output['id'] = os.path.split(txt)[1].split('.')[0]
    output['context'] , output['sents']= context_list, sents_list
    output['events'] = event_list
    #print(output)
    outputs = output_path + output['id'] + '.txt'
    output_file = open(outputs, 'w')
    #print(outputs)
    output_file.write(json.dumps(output))
    #return output


def process_all(path, output_path, mode):
    path, output_path = path + '/' + mode, output_path + '/' + mode + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    for corpus_f in glob(path + '/*.txt'):
        idx = os.path.split(corpus_f)[1].split('.')[0]
        ann_f = os.path.join(path, idx+'.ann')
        #print(ann_f)
        process_paie_format_single(corpus_f, ann_f, output_path)

def event_info(path):
    event_count, file_cnt = 0, 0
    arg_nums = dict.fromkeys(range(6), 0)
    print(path)
    for ann_file in glob(path + '/*.ann'):
        file_cnt += 1
        read_f = open(ann_file, 'r')
        lines = read_f.readlines()
        for line in lines:
            if line.startswith('E'):
                split_list = line.strip('\n').split('\t')[1]
                arg_list = split_list.split(' ')
                if len(arg_list) == 5:
                    print(split_list)
                arg_nums[len(arg_list)] += 1
    
    print(file_cnt, arg_nums)


def clean_up(in_dir, out_dir, fn):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fpath = os.path.join(out_dir, fn)
    writer = jsonlines.open(out_fpath, mode='w')

    fnames = os.listdir(in_dir)
    event_type_2_role_2_cnt = dict()
    total_arg = 0
    trigger_as_arg = 0
    for fn in fnames:
        fpath = os.path.join(in_dir, fn)
        reader = jsonlines.open(fpath)
        for line in reader:
            events = line['events']
            events_ = []
            triggers = set()
            for event in events:
                event_type = event['event_type']
                args = event['args']
                args_ = []
                this_role_2_cnt = dict()
                for arg in args:
                    role = arg[-1]
                    if 'Site' in role and role != 'Site':
                        role = 'Site'
                    if 'Theme' in role and role != 'Theme':
                        role = 'Theme'
                    if 'Participant' in role and role != 'Participant':
                        role = 'Participant'
                    if 'Instrument' in role and role != 'Instrument':
                        role = 'Instrument'
                    args_.append(arg[:-1] + [role])
                        
                    if role in this_role_2_cnt:
                        this_role_2_cnt[role] += 1
                    else:
                        this_role_2_cnt[role] = 1

                if event_type in ['Dissociation', 'Cell_division', 'Acetylation', 'Ubiquitination', 'Translation', 'Reproduction']:
                    print(event)
                    continue
                if event_type in ['Localization', 'Positive_regulation', 'Planned_process'] \
                    and 'Theme' in this_role_2_cnt and this_role_2_cnt['Theme'] > 1:
                    print(event)
                    continue
                if event_type == 'Blood_vessel_development' and ('Theme' in this_role_2_cnt or 'FromLoc' in this_role_2_cnt):
                    print(event)
                    continue
                # if event_type == 'Gene_expression' and this_role_2_cnt['Theme'] > 2:
                #     print(event)
                #     continue
                # if event_type == 'Pathway' and 'Participant' in this_role_2_cnt and this_role_2_cnt['Participant'] > 2:
                #     print(event)
                #     continue

                events_.append({'event_type': event_type, 'trigger': event['trigger'], 'args': args_})
                
                if event_type not in event_type_2_role_2_cnt:
                    event_type_2_role_2_cnt[event_type] = this_role_2_cnt
                else:
                    for role, cnt in this_role_2_cnt.items():
                        if role not in event_type_2_role_2_cnt[event_type] or cnt > event_type_2_role_2_cnt[event_type][role]:
                            event_type_2_role_2_cnt[event_type][role] = cnt
                            event_type_2_role_2_cnt[event_type][role] = cnt

                triggers.add((event['trigger'][0], event['trigger'][1]))
            
            for event in events_:
                for arg in event['args']:
                    total_arg += 1
                    if (arg[0], arg[1]) in triggers:
                        trigger_as_arg += 1
            
            new_line = copy.deepcopy(line)
            new_line['events'] = events_
            writer.write(new_line)

    print('\n[Trigger as Arg] %d' % trigger_as_arg)
    print('[Total Arg] \t %d' % total_arg)
    print('Nested Ratio\t %f' % (trigger_as_arg/total_arg))

    return event_type_2_role_2_cnt


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
        doc_key = line["id"]
        full_text = line['context']
        doc_length = len(full_text)
        # print(doc_key)
        
        sents = line['sents']
        sent_lens = [len(sent) for sent in sents]
        cum_lens = np.cumsum(sent_lens)
        assert sum(sent_lens) == doc_length

        events = line["events"]
        event_tuples = []
        if len(events) == 0:
            continue

        for i, event in enumerate(events):
            trigger = event['trigger']
            args = event['args']

            event_span = [trigger[0], trigger[1]]
            for arg in args:
                event_span = [min(event_span[0], arg[0]), max(event_span[1], arg[1])]

            # find the range of sents that the event occupies
            start_sent_index, end_sent_index = -1, -1
            for j, cum_len in enumerate(cum_lens):
                if start_sent_index == -1 and cum_len > event_span[0]:
                    start_sent_index = j
                if end_sent_index == -1 and cum_len >= event_span[1]:
                    end_sent_index = j
            event_span = [start_sent_index, end_sent_index]

            event_tuple = ([i], event_span)
            event_tuples.append(event_tuple)

        event_tuples = sorted(event_tuples, key=lambda x: x[1])
        # if doc_key == 'PMID-11948691':
        #     print(event_tuples)

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
                        # if merge_span[0] != span1[0] or merge_span[1] != span1[1] or merge_span[0] != span2[0] or merge_span[1] != span2[1]:
                        #     print(doc_key)
                        #     print(span1, span2)
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

            new_events = sorted(new_events, key=lambda x: x['trigger'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='--indir')
    parser.add_argument('--outdir', type=str, required=True, help='--outdir')
    args = parser.parse_args()

    input_dir = getattr(args, 'indir') 
    output_dir = getattr(args, 'outdir')
    
    print('[MLEE Preprocess] step 2 starting...\n')

    # Conduct Tokenization
    process_all(input_dir, output_dir, 'test')
    process_all(input_dir, output_dir, 'train') 

    # Aggregate instances and Discard some rare cases
    event_type_2_role_2_cnt = clean_up(in_dir=os.path.join(output_dir, 'train'), out_dir=os.path.join(output_dir, 'data'), fn='train.json')
    print('\n[Train]')
    for event_type, role_2_cnt in event_type_2_role_2_cnt.items():
        print(event_type, '\t', role_2_cnt)
    print()

    event_type_2_role_2_cnt = clean_up(in_dir=os.path.join(output_dir, 'test'), out_dir=os.path.join(output_dir, 'data'), fn='test.json')
    print('\n[Test]')
    for event_type, role_2_cnt in event_type_2_role_2_cnt.items():
        print(event_type, '\t', role_2_cnt)
    print()
    
    # Split document into chunks to avoid exceeding max len limit
    in_dir = os.path.join(output_dir, 'data')
    out_dir = os.path.join(output_dir, 'data_final')
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
