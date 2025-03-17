import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import json
import ast
import random
import os

# 데이터 불러오기
df = pd.read_csv("/home/khj6051/total_splice_0309.csv")
# df = df[~pd.isna(df['audio_path'])]
# df = df[[os.path.exists('/home/khj6051/sp/'+path) for path in df['audio_path'].values]]

df['instrument'] = df['instrument'].apply(lambda x: str(ast.literal_eval(x) + ['drums']) if 'percussion' in x and 'drums' not in ast.literal_eval(x) else x)

# 필터링 최적화
df_filtered = df[(df['duration'] >= 4) & (df['is_loop']) & (df['instrument'].str.len().fillna(0) > 2)]

# 악기 필터링 패턴
inst_types = set([
    'vocals',
    'drums',
    'guitar',
    'keys',
    'strings',
    'brass & woodwinds',
    'bass',
    'percussion'
])
pattern = "|".join(inst_types)

# target_df 생성
target_df = df_filtered[
    df_filtered["instrument"].str.contains(pattern, na=False, regex=True) |
    ((df_filtered["instrument"].str.contains("synth", na=False)) & (df_filtered["duration"] >= 8))
]
target_df = target_df[~target_df['tags'].str.contains("songstarter")]

# back_df 생성 및 인덱싱 최적화
back_df = df_filtered[
    df_filtered["instrument"].str.contains(pattern, na=False, regex=True) |
    ((df_filtered["instrument"].str.contains("synth", na=False)) & (df_filtered["duration"] >= 4)) |
    ((df_filtered["instrument"].str.contains("fx", na=False)) & (df_filtered["duration"] >= 8))
].drop(columns=['Unnamed: 0', 'tags', 'audio_path', 'is_loop', 'is_oneshot'])

# back_df를 url_base 기준으로 딕셔너리 변환 (검색 속도 최적화)
back_df_dict = {k: v for k, v in back_df.groupby("url_base")}

# 🔥 find_same_packs 함수 (검색 최적화 + 공유 리스트 업데이트)
def find_same_packs(url_base, inst, bpm, idx):
    if url_base not in back_df_dict:
        return [], 0

    res = back_df_dict[url_base]
    res = res[(res['bpm'] != '--') & (res['bpm'] == bpm)]

    lst = ast.literal_eval(inst)
    lst = [l for l in lst if l in inst_types]

    if 'drums' in lst:
        lst.append("percussion")
    elif 'percussion' in lst:
        lst.append("drums")
    
    pattern = "|".join(lst)

    res = res[~res['instrument'].str.contains(pattern, na=False, regex=True)]

    tot_len = len(res)
    if tot_len == 0:
        return [], 0

    num_of_add_insts = random.randint(1, 5)
    datas = []

    for i in range(num_of_add_insts):
        if len(res) == 0:
            break

        indexs = random.randint(0, len(res) - 1)
        data2 = res.iloc[indexs]
        res = res[~res['instrument'].str.contains("|".join(ast.literal_eval(data2['instrument'])), na=False, regex=True)]
        datas.append(data2)

    return datas, tot_len


# 🔥 멀티프로세싱을 활용한 함수 (진행률 업데이트 포함)
def process_target(i, progress, lock, results):
    data = target_df.iloc[i]
    same_url_list = back_df[back_df['url_base'] == data['url_base']].index.to_list()
    if not same_url_list:
        return

    idx_loc = back_df.index.get_loc(same_url_list[len(same_url_list) // 2])
    cand_list, total_len = find_same_packs(data['url_base'], data['instrument'], data['bpm'], idx_loc)

    with lock:
        if cand_list:
            results.append([data['sample_name'], *[i['sample_name'] for i in cand_list]])

        if total_len > 6:
            for _ in range(total_len // 6):
                cand_list, _ = find_same_packs(data['url_base'], data['instrument'], data['bpm'], idx_loc)
                if cand_list:
                    results.append([data['sample_name'], *[i['sample_name'] for i in cand_list]])

        progress.value += 1  # ✅ 진행률 업데이트


# 🔥 멀티프로세싱 실행 (진행률 출력 포함)
def main():
    num_workers = mp.cpu_count() - 1 # 사용 가능한 CPU 코어 개수
    manager = mp.Manager()
    results = manager.list()
    progress = manager.Value("i", 0)
    lock = manager.Lock()

    pool = mp.Pool(num_workers)
    total_tasks = len(target_df)

    # tqdm과 함께 실행
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        def update_progress(_):
            pbar.update(1)

        for i in range(total_tasks):
            pool.apply_async(process_target, args=(i, progress, lock, results), callback=update_progress)

        pool.close()
        pool.join()

    # 결과 저장
    with open("data_splice_0314.json", "w", encoding="utf-8") as f:
        json.dump(list(results), f, ensure_ascii=False, indent=4)

    print(f"\n✅ 완료! 총 {len(results)}개의 데이터가 저장됨.")

if __name__ == "__main__":
    main()
