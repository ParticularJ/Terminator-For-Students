from collections import defaultdict


test_dict = defaultdict(dict)

test_dict['1']['1'] = 0

for key in test_dict.keys():
    for key_ in test_dict[key].keys():
        print(test_dict[key][key_])
        print(key + '_' + key_)