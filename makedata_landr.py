# nohup python3 makedata_landr.py > make.log 2>&1 &
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import ast
import random
import os
import json

df = pd.read_csv("/home/khj6051/total_landr_250308.csv")
print('l1 : ', len(df))
df = df[~pd.isna(df['audio_path'])]
df = df[[os.path.exists('/home/khj6051/'+path) for path in df['audio_path'].values]]
df['new_sample_name'] = df['sample_name'] + df['pack_title']
print('l2 : ', len(df))

kv = {
    # '808': 'drums',
    'arp': 'synth',
    'bass': 'bass',
    'brass': 'brass & woodwinds',
    'cello': 'strings',
    'clavinet': 'keys',
    'conga': 'percussion',
    'djembe': 'percussion',
    'drums': 'drums',
    'electric guitar': 'guitar',
    'electric piano': 'keys',
    'flute': 'brass & woodwinds',
    'guitar': 'guitar',
    'keys': 'keys',
    'orchestral': 'strings',
    'organ': 'keys',
    'percussion': 'percussion',
    'piano': 'keys',
    'saxophone': 'brass & woodwinds',
    'strings': 'strings',
    'synth': 'synth',
    'tambourine': 'percussion',
    'trombone': 'brass & woodwinds',
    'trumpet': 'brass & woodwinds',
    'viola': 'strings',
    'violin': 'strings',
    'vocal': 'vocals',
    'vocoder': 'vocals',
    'fx': 'fx'
}

df['instrument'] = df['instrument'].apply(lambda x: str(ast.literal_eval(x) + ['drums']) if '808' in x and 'drums' not in ast.literal_eval(x) and 'bass' not in ast.literal_eval(x) else x)

for k, v in kv.items():
    df['instrument'] = df['instrument'].apply(lambda x: str(ast.literal_eval(x) + [v]) if k in x and v not in ast.literal_eval(x) else x)

# df['instrument'] = df['instrument'].apply(lambda x: str(ast.literal_eval(x) + ['tests']) if 'percussion' in x else x)
# df['instrument'] = df['instrument'].apply(lambda x: str(ast.literal_eval(x) + ['tests']) if 'drums' in x else x)

print("new df1 ", len(df))
df.to_csv('new_total_landr_250308.csv')

inst_types = [
    'vocals',
    'drums',
    'guitar',
    'keys',
    'strings',
    'brass & woodwinds',
    'bass',
    'percussion'
]

# 정규표현식 패턴 생성 (각 악기를 OR 연산으로 묶음)
pattern = "|".join(inst_types)

# instruments 칼럼에서 패턴이 포함된 행만 필터링
df = df[
    (df['instrument'].str.len()>2) & 
    (df['duration']>=4) & 
    (df['is_loop'] == True)
]

df = df[
    df["instrument"].str.contains(pattern, na=False, regex=True) |
    ((df["instrument"].str.contains("synth", na=False)) & (df["duration"] >= 8))
]
target_df = df[~df['tags'].str.contains("songstarters")][~df['tags'].str.contains("multi instrument")]
print("target_df : ", len(target_df))

# instruments 칼럼에서 패턴이 포함된 행만 필터링
df = df[
    df["instrument"].str.contains(pattern, na=False, regex=True) |
    ((df["instrument"].str.contains("synth", na=False)) & (df["duration"] >= 4)) |
    ((df["instrument"].str.contains("fx", na=False)) & (df["duration"] >= 8))
]
back_df = df[~df['tags'].str.contains("songstarters")][~df['tags'].str.contains("multi instrument")]
print("back df : ", len(back_df))
back_df = back_df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'tags', 'audio_path', 'is_loop', 'is_oneshot'])

import ast
import random

inst_types = set(inst_types)

def find_same_packs(url_base, inst, bpm, idx):
    lst = ast.literal_eval(inst)
    lst = [l for l in lst if l in inst_types]

    if 'drums' in lst:
        lst.append("percussion")
    elif 'percussion' in lst:
        lst.append("drums")

    pattern = "|".join(lst)

    res = back_df[idx-500:idx+500]

    res = res.loc[
        (res['pack_title'] == url_base) &
        (res['bpm'] != '--') &
        (res['bpm'] == bpm) &
        (~res['instrument'].str.contains(pattern, na=False, regex=True))
    ]

    tot_len = len(res)
    if tot_len == 0:
        return [], 0

    num_of_add_insts = random.randint(1, 5)
    
    datas = []
    for i in range(num_of_add_insts):
        if len(res) == 0:
            break

        idx = random.randint(0, len(res)-1)
        data2 = res.iloc[idx]
        lst = ast.literal_eval(data2['instrument'])
        lst = [l for l in lst if l in inst_types]
        pattern = "|".join(lst)
        res = res[~res['instrument'].str.contains(pattern, na=False, regex=True)]
        datas.append(data2)
    
    return datas, tot_len


alls = []
for i in tqdm(range(len(target_df))):
    data = target_df.iloc[i]
    cand_list, total_len = find_same_packs(data['pack_title'], data['instrument'], data['bpm'], i)

    if len(cand_list)>0:
        alls.append([
            data['new_sample_name'],
            *[i['new_sample_name'] for i in cand_list]
        ])
        
        if total_len>6:
            for _ in range(total_len//6):
                cand_list, total_len2 = find_same_packs(data['pack_title'], data['instrument'], data['bpm'], i)
                alls.append([
                    data['new_sample_name'],
                    *[i['new_sample_name'] for i in cand_list]
                ])
    
    if i%5000 == 4999:
        with open("data_landr_0314.json", "w", encoding="utf-8") as f:
            json.dump(alls, f, ensure_ascii=False, indent=4)  # ensure_ascii=False → 한글 깨짐 방지
        print(len(alls))

with open("data_landr_0314.json", "w", encoding="utf-8") as f:
    json.dump(alls, f, ensure_ascii=False, indent=4)  # ensure_ascii=False → 한글 깨짐 방지



