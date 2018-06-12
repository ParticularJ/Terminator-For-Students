from random import sample
from collections import defaultdict

if __name__ == '__main__':
    stu_dict = {'0': '李兵', '1': '叶超超', '2': '肖琦', '3': '蒋文斌', '4': '卢海峰', '5': '谢恩宁',
                '6': '王恒', '7': '高慧云', '8': '杨梦月', '9': '张雨金'}
    stu_dict_ = defaultdict(lambda: 0)
    for i in range(1000000):
        sample_list = sample(list(stu_dict.keys()), 6)
        for sample_ in sample_list:
            stu_dict_[stu_dict[sample_]] += 1
    result_list = []
    for stu_key in stu_dict_.keys():
        result_list.append((stu_key, stu_dict_[stu_key]))
    sorted_result = sorted(result_list, key=lambda x: x[1], reverse=True)
    for name, count in sorted_result:
        print('{0}: {1}'.format(name, count))