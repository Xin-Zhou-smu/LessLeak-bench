from pathlib import Path
import json
from pathlib import Path
from collections import Counter
import pickle, os

def list_files_in_directory(path):
    return [f.resolve() for f in Path(path).rglob('*') if f.is_file()]

path = './less-leak-data/'
files = list_files_in_directory(path)
def sort_files_by_first_letter(file_list):
    return sorted(file_list, key=lambda x: os.path.basename(x)[0].lower())

sorted_files = sort_files_by_first_letter(files)
for file_ in sorted_files:
    file_ = '/'.join(file_.parts)

    print(file_.split('/')[-1])
    count_label = []
    unique_leaked_data_samples = []
    unique_leaked_data_ids = []
    with open(file_, 'r') as f:
        data = json.load(f)


        for ijk in range(1, len(data)):
            d = data [ijk]
            if 'id' in d[0] and len(list(d[0].keys())) == 1:
                d = d[1:]

            if d[0]['id'].startswith('test_'):
                input_a = d[0]['content']  ## benchmark data
                input_a_id = d[0]['id']
                input_b = d[1]['content']  ## pre-training data

                if len(input_a) == 0:
                    input_a = d[0]['befor_content']
                if len(input_b) == 0:
                    input_b = d[1]['befor_content']
            else:
                input_a = d[1]['content']  ## benchmark data
                input_a_id = d[1]['id']
                input_b = d[0]['content']  ## pre-training data
                if len(input_a) == 0:
                    input_a = d[1]['befor_content']
                if len(input_b) == 0:
                    input_b = d[0]['befor_content']

            try:
                label_ = d[2]['real_dup']
            except:
                try:
                    label_ = d[3]['real_dup']
                except:
                    label_ = d[2][0]['real_dup']

            label_ = str(label_)
            count_label.append(label_)

            if label_ in ['2', '3']:
                if input_a_id  not in unique_leaked_data_ids:
                    unique_leaked_data_ids.append(input_a_id)

    print(len(count_label), Counter(count_label))
    print('unique_leaked_data_ids:', len(unique_leaked_data_ids))
    print()
