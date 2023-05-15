import os
import jsonlines

global_cnt = 0

total_arg = 0
total_events = 0
total_instance = 0

def convert(in_file, out_file, max_len=250):
    global total_arg
    global total_events
    global total_instance

    global global_cnt
    num_event_cnter = dict()
    reader = jsonlines.open(in_file)
    writer = jsonlines.open(out_file, mode='w')
    for line in reader:
        sent = line["sentence"]
        offset = line["s_start"]
        events = line["event"]

        assert len(sent) < max_len

        if not line['event']:
            continue

        events_ = []
        for event in events:
            event_type = event[0][1]
            start = event[0][0] - offset; end = start+1
            event_trigger = [start, end, " ".join(sent[start:end])]

            event_args = []
            for arg_info in event[1:]:
                start = arg_info[0]-offset; end = arg_info[1]-offset+1
                role = arg_info[2]
                arg = [start, end, " ".join(sent[start:end]), role]
                event_args.append(arg)

            events_.append({'event_type': event_type, 'trigger': event_trigger, 'args': event_args})
            total_arg += len(event_args)
            total_events += 1
        total_instance += 1

        sample = {'id': global_cnt, 'context': sent, 'events': events_}
        writer.write(sample)
        global_cnt += 1

        num_event = len(events_)
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
    in_dir = 'data'
    out_dir = 'data_final'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fnames = os.listdir(in_dir)
    list_num_event_cnter = []
    for fn in fnames:
        fpath = os.path.join(in_dir, fn)
        out_fpath = os.path.join(out_dir, fn)

        print('\n', fpath)
        num_event_cnter = convert(fpath, out_fpath)
        list_num_event_cnter.append(num_event_cnter)

    print('[total_arg]', total_arg)
    print('[total_events]', total_events)
    print('[total_instance]', total_instance)
    print('[# args per event]', total_arg / total_events)
    print('[# event per context]', total_events / total_instance)

