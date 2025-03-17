# import json
# import pandas as pd
# from tqdm import tqdm
# import random

# with open("./data_landr_0314.json", "r", encoding="utf-8") as file:
#     data_ld = json.load(file)  # JSON 데이터를 파이썬 객체(딕셔너리 or 리스트)로 변환

# with open("./data_splice_0314.json", "r", encoding="utf-8") as file:
#     data_sp = json.load(file)  # JSON 데이터를 파이썬 객체(딕셔너리 or 리스트)로 변환

# print(len(data_ld))
# print(len(data_sp))

# spdf = pd.read_csv('/home/khj6051/total_splice_250310.csv')
# lddf = pd.read_csv('/home/khj6051/mel_con_sample/new_total_landr_250308.csv')

# sp_dict = {}
# for i in tqdm(range(len(spdf))):
#     d = spdf.iloc[i]
#     sp_dict[d['sample_name']] = ['/home/khj6051/sp/'+d['audio_path'], int(d['duration']), d['tags'], d['instrument']]

# alls = []
# enum = 0
# for lst in tqdm(data_sp):
#     try:
#         target = lst[0]
#         target_data = sp_dict[target]

#         others = lst[1:]

#         if 'drums' in target_data[-1] or 'bass' in target_data[-1]:
#             if random.random() >= 0.5:
#                 continue
        
#         alls.append({
#             'audio_path': target_data[0],
#             'text': target_data[-1],
#             'duration': target_data[1]+1,
#             'tags': target_data[2],
#             'others': [
#                 sp_dict[o] for o in others
#             ]
#         })
#     except Exception as e:
#         enum += 1

# ld_dict = {}
# for i in tqdm(range(len(lddf))):
#     d = lddf.iloc[i]
#     ld_dict[d['sample_name']+d['pack_title']] = ['/home/khj6051/'+d['audio_path'], int(d['duration']), d['tags'], d['instrument']]

# for lst in tqdm(data_ld):
#     try:
#         target = lst[0]
#         target_data = ld_dict[target]

#         others = lst[1:]

#         if 'drums' in target_data[-1] or 'bass' in target_data[-1]:
#             if random.random() >= 0.5:
#                 continue
        
#         alls.append({
#             'audio_path': target_data[0],
#             'text': target_data[-1],
#             'duration': target_data[1]+1,
#             'tags': target_data[2],
#             'others': [
#                 ld_dict[o] for o in others
#             ]
#         })
#     except Exception as e:
#         enum += 1

# alldf = pd.DataFrame(alls)
# alldf.to_csv("total_mixs_0314.csv")
# print(len(alldf))

import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("total_mixs_0314.csv")

# 99:1 비율로 데이터 분할
train_df = df[:-20000]
# train_df = df.sample(frac=0.99, random_state=42)  # 99% 샘플링
test_df = df.drop(train_df.index)  # 나머지 1% 사용

# 분할된 데이터 저장
train_df.to_csv("train_mix.csv", index=False)
test_df.to_csv("valid_mix.csv", index=False)

print(f"Train Data: {len(train_df)} rows")
print(f"Test Data: {len(test_df)} rows")

# import pandas as pd
# import torchaudio
# import multiprocessing as mp
# from tqdm import tqdm

# def get_audio_info(audio_filename):
#     """오디오 파일 정보를 가져오는 함수"""
#     try:
#         ti = torchaudio.info(audio_filename)
#         original_sample_rate = ti.sample_rate
#         duration = ti.num_frames / original_sample_rate
#         total_samples = int(ti.num_frames)
#         return audio_filename, original_sample_rate, duration, total_samples
#     except Exception as e:
#         print(f"Error processing {audio_filename}: {e}")
#         return audio_filename, None, None, None

# def process_audio_files(audio_files, num_workers=mp.cpu_count() - 1):
#     """멀티프로세싱을 이용하여 오디오 파일 정보를 가져옴"""
#     with mp.Pool(num_workers) as pool:
#         results = list(tqdm(pool.imap(get_audio_info, audio_files), total=len(audio_files)))
#     return results

# # CSV 파일 로드
# traind = pd.read_csv('valid_mix.csv')
# audio_files = traind['audio_path'].tolist()
# results = process_audio_files(audio_files)
# results_df = pd.DataFrame(results, columns=['audio_path', 'original_sample_rate', 'duration', 'total_samples'])
# traind = traind.merge(results_df, on='audio_path', how='left')
# # 기존 duration을 새로운 duration으로 업데이트
# traind['duration'] = traind['duration_y']
# traind = traind.drop(columns=['duration_x', 'duration_y'])  # 불필요한 칼럼 삭제

# # 변경된 CSV 저장
# traind.to_csv('valid_mix_with_audio_info.csv', index=False)
# print("Processing complete. Saved as valid_mix_with_audio_info.csv")