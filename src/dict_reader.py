def text_to_dict(path):
    slot_set = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            slot_set[line.rstrip('\n').rstrip('\r')] = ''

    return slot_set
