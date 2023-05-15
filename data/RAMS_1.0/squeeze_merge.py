import os
import jsonlines
import argparse
import copy
import numpy as np
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


def squeeze(in_file, max_len=250):

    num_event_cnter = dict()
    url_2_anns = dict()
    reader = jsonlines.open(in_file)
    for line in reader:
        assert len(line["evt_triggers"]) == 1

        sents = line['sentences']
        text = list(chain(*sents))
        event = line["evt_triggers"][0]
        trigger = [event[0], event[1] + 1, ' '.join(text[event[0] : event[1] + 1])]
        event_type = event[2][0][0]

        args = []
        event_span = trigger

        for arg_info in line["gold_evt_links"]:
            assert arg_info[0][0] == event[0] and arg_info[0][1] == event[1]
            arg = arg_info[1]
            arg = [arg[0], arg[1] + 1, ' '.join(text[arg[0] : arg[1] + 1]),
                    arg_info[2].split('arg', maxsplit=1)[-1][2:]]

            args.append(arg)

            event_span = [min(event_span[0], arg[0]), max(event_span[1], arg[1])]

        sent_lens = [len(sent) for sent in sents]
        cum_lens = np.cumsum(sent_lens)

        text_squeezed = text
        if len(text) > max_len:
            # find the range of sents that the event occupies
            start_sent_index, end_sent_index = -1, -1
            for i, cum_len in enumerate(cum_lens):
                if start_sent_index == -1 and cum_len > event_span[0]:
                    start_sent_index = i
                if end_sent_index == -1 and cum_len >= event_span[1]:
                    end_sent_index = i
            event_text_len = cum_lens[end_sent_index] - cum_lens[start_sent_index] + sent_lens[start_sent_index]

            # expand the context by appending neighbor sents until exceeding the max len limit
            flag_expand_front = True
            flag_expand_back = True
            while flag_expand_front or flag_expand_back:
                index_front = start_sent_index - 1
                index_back = end_sent_index + 1
                
                if index_front >= 0:
                    if (event_text_len + sent_lens[index_front]) <= max_len:
                        start_sent_index = index_front
                        event_text_len += sent_lens[index_front]
                    else:
                        flag_expand_front = False
                else:
                    flag_expand_front = False

                if index_back < len(sents):
                    if (event_text_len + sent_lens[index_back]) <= max_len:
                        end_sent_index = index_back
                        event_text_len += sent_lens[index_back]
                    else:
                        flag_expand_back = False
                else:
                    flag_expand_back = False

            if start_sent_index > 0:
                offset = cum_lens[start_sent_index-1]
                trigger[0] = int(trigger[0] - offset)
                trigger[1] = int(trigger[1] - offset)
                for i, arg in enumerate(args):
                    args[i][0] = int(args[i][0] - offset)
                    args[i][1] = int(args[i][1] - offset)

            text_squeezed = list(chain(*sents[start_sent_index : end_sent_index+1]))
            # print('[After]', len(text_squeezed))
            # print()
            sent_lens = [len(sent) for sent in sents[start_sent_index : end_sent_index+1]]
            cum_lens = np.cumsum(sent_lens)

        # calculate uni_span for each annotated event
        # defined as the (start, end) index of the sentence where the trigger occurs 
        # used as the clue for merging
        start_sent_index, end_sent_index = -1, -1
        for i, cum_len in enumerate(cum_lens):
            if start_sent_index == -1 and cum_len > trigger[0]:
                start_sent_index = i
            if end_sent_index == -1 and cum_len > trigger[1]:
                end_sent_index = i
        # start_sent_index = max(start_sent_index-1, 0)
        # end_sent_index= min(end_sent_index+1, len(sents)-1)
        uni_span = (int(cum_lens[start_sent_index]-sent_lens[start_sent_index]), int(cum_lens[end_sent_index]))
        assert (uni_span[1] - uni_span[0]) <= len(text)

        url = line["source_url"]
        event = {'event_type': event_type, 'trigger': trigger, 'args': args}
        ann = {'url': url, 'text': text_squeezed, 'events': [event], 'uni_span': uni_span}
        # print(ann, '\n')
        if url not in url_2_anns:
            url_2_anns[url] = [ann]
        else:
            url_2_anns[url].append(ann)

    for anns in url_2_anns.values():
        num_event = len(anns)
        if num_event not in num_event_cnter:
            num_event_cnter[num_event] = 1
        else:
            num_event_cnter[num_event] += 1

    return url_2_anns


def merge_(ann1, ann2, max_len, do_print):
    text1 = ann1['text']
    text2 = ann2['text']
    len1 = len(text1)
    len2 = len(text2)

    span1 = ann1['uni_span']
    span2 = ann2['uni_span']

    span1_text = text1[span1[0] : span1[1]]
    startOffset_in_text2 = find_list(span1_text, text2)
    endOffset_in_text2 = len2 - startOffset_in_text2 - len(span1_text)

    span2_text = text2[span2[0] : span2[1]]
    startOffset_in_text1 = find_list(span2_text, text1)
    endOffset_in_text1 = len1 - startOffset_in_text1 - len(span2_text)

    if startOffset_in_text2 > span1[0]:
        offset = startOffset_in_text2 - span1[0]
        text = text2[:offset] + text1
        rear_offset = endOffset_in_text2 - (len1 - span1[1])
        if rear_offset > 0:
            if rear_offset != (len2 - span2[1]) - endOffset_in_text1:
                print(text1)
                print(span1_text)
                print()
                print(text2)
                print(span2_text)
                print(len1, span1[1], endOffset_in_text2)
                print(endOffset_in_text1, len2, span2[1])
                exit(0)
            text = text + text2[-rear_offset:]

        events = copy.deepcopy(ann2['events'])
        for event in ann1['events']:
            trigger = event['trigger']
            args = event['args']

            trigger_ = [trigger[0]+offset, trigger[1]+offset, trigger[2]]
            args_ = [[arg[0]+offset, arg[1]+offset, arg[2], arg[3]] for arg in args]
            event_ = {'event_type': event['event_type'], 'trigger': trigger_, 'args': args_}
            events.append(event_)

        span1 = [span1[0]+offset, span1[1]+offset]
        
    else:
        offset = span1[0] - startOffset_in_text2
        text = text1[:offset] + text2
        rear_offset = endOffset_in_text1 - (len2 - span2[1])
        if rear_offset > 0:
            if rear_offset != (len1 - span1[1]) - endOffset_in_text2:
                print(text1)
                print(span1_text)
                print()
                print(text2)
                print(span2_text)
                print(len1, span1[1], endOffset_in_text2)
                print(endOffset_in_text1, len2, span2[1])
                exit(0)
            text = text + text1[-rear_offset:]

        events = copy.deepcopy(ann1['events'])
        for event in ann2['events']:
            trigger = event['trigger']
            args = event['args']

            trigger_ = [trigger[0]+offset, trigger[1]+offset, trigger[2]]
            args_ = [[arg[0]+offset, arg[1]+offset, arg[2], arg[3]] for arg in args]
            event_ = {'event_type': event['event_type'], 'trigger': trigger_, 'args': args_}
            events.append(event_)


        span2 = [span2[0]+offset, span2[1]+offset]
    
    uni_span = [min(span1[0], span2[0]), max(span1[1], span2[1])]
    assert (uni_span[1] - uni_span[0]) <= len(text)
    ann = {'text': text, 'events': events, 'uni_span': uni_span}

    if len(text) > max_len:
        return -1

    return ann

def can_merge(ann1, ann2):
    text1 = ann1['text']
    text2 = ann2['text']
    span1 = ann1['uni_span']
    span2 = ann2['uni_span']
    span1_text = text1[span1[0] : span1[1]]
    span2_text = text2[span2[0] : span2[1]]

    return (find_list(span1_text, text2) != -1) and (find_list(span2_text, text1) != -1)

total_arg = 0
total_events = 0
total_instance = 0

def merge(url_2_anns, out_file, max_len=250):
    num_event_cnter = dict()
    writer = jsonlines.open(out_file, mode='w')
    global total_arg
    global total_events
    global total_instance
        
    for url, anns in url_2_anns.items():
        # merging
        num_ann = len(anns)
        tmp_anns = copy.deepcopy(anns)
        merged_indexs = []
        have_merged = True
        while have_merged:
            have_merged = False
            for i in range(num_ann):
                if i in merged_indexs:
                    continue
                ann1 = tmp_anns[i]

                for j in range(i+1, num_ann):
                    if j in merged_indexs:
                        continue
                    ann2 = tmp_anns[j]
                    
                    if can_merge(ann1, ann2):
                        do_print = False
                        merge_ann = merge_(ann1, ann2, max_len=max_len, do_print=do_print)
                        if merge_ann != -1:
                            tmp_anns[i] = merge_ann
                            ann1 = merge_ann
                            merged_indexs.append(j)
                            have_merged = True
        
        new_anns = [tmp_anns[i] for i in range(num_ann) if i not in merged_indexs]

        # print(url)
        for i, ann in enumerate(new_anns):
            num_event = len(ann['events'])
            if num_event not in num_event_cnter:
                num_event_cnter[num_event] = 1
            else:
                num_event_cnter[num_event] += 1

            id = '%d-%s' % (i, url)
            sample = {'id': id, 'context': ann['text'], 'events': ann['events']}
            total_arg += sum([len(event['args']) for event in ann['events']])
            total_events += len(ann['events'])
            total_instance += 1

            writer.write(sample)
            if len(ann['text']) > max_len:
                print('[Exceed max len limit]: %s %d, %d events' % (id, len(ann['text']), len(ann['events'])))
            # if len(ann['events']) == 3:
            #     print()
            # joint = ' '.join(ann['text'])
            # if not any([joint in text for text in url2texts[url]]) and len(ann['events']) > 1:
            #     print(url)
            #     print(joint)

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
        print(fpath)
        url_2_anns = squeeze(fpath, max_len=250)
        num_event_cnter = merge(url_2_anns, out_fpath, max_len=260)
        list_num_event_cnter.append(num_event_cnter)

    print('[total_arg]', total_arg)
    print('[total_events]', total_events)
    print('[total_instance]', total_instance)
    print('[# args per event]', total_arg / total_events)
    print('[# event per context]', total_events / total_instance)

    