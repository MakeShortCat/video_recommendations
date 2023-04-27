import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import ElectraTokenizer, ElectraModel
import re



###################### 파일에 있었던 오류 제거 코드 현재는 csv파일 수정 완료로 쓰이지 않음 ##################
# # 파일이 저장된 폴더 경로
# folder_path = "C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_data/fix"

# # 폴더 내 파일 리스트 가져오기
# file_list = os.listdir(folder_path)

# # 파일 리스트 중 CSV 파일만 선택하여 데이터프레임으로 읽어오기
# df_list = []
# for file_name in file_list:
#     if file_name.endswith(".csv"):
#         file_path = os.path.join(folder_path, file_name)
#         df = pd.read_csv(file_path)
#         df_list.append(df)

# # 모든 데이터프레임을 병합하여 하나의 데이터프레임으로 만들기
# watcha_nfix_df = pd.concat(df_list, axis=0, ignore_index=True)

# watcha_nfix_df.drop('Unnamed: 0', axis=1, inplace=True)

# watcha_nfix_df['title_num'] = watcha_nfix_df['title_num'] + 10000

# watcha_nfix_df.sort_values('title_num').to_csv('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_data/watcha_10000.0.csv')





# 파일이 저장된 폴더 경로
folder_path = "C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_data"

# 폴더 내 파일 리스트 가져오기
file_list = os.listdir(folder_path)

# 파일 리스트 중 CSV 파일만 선택하여 데이터프레임으로 읽어오기
df_list = []
for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df_list.append(df)

# 모든 데이터프레임을 병합하여 하나의 데이터프레임으로 만들기
watcha_merged_df = pd.concat(df_list, axis=0, ignore_index=True)

watcha_merged_df.drop('Unnamed: 0', axis=1, inplace=True)

# 파일이 저장된 폴더 경로
folder_path = "C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/data"

# 폴더 내 파일 리스트 가져오기
file_list = os.listdir(folder_path)

# 파일 리스트 중 CSV 파일만 선택하여 데이터프레임으로 읽어오기
df_list = []
for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df_list.append(df)

# 모든 데이터프레임을 병합하여 하나의 데이터프레임으로 만들기
just_merged_df = pd.concat(df_list, axis=0, ignore_index=True)

just_merged_df.drop('Unnamed: 0', axis=1, inplace=True)

just_merged_df = just_merged_df.drop_duplicates(subset=['title']).dropna().reset_index(drop=True)

# KoELECTRA 모델 및 토크나이저 불러오기
model_name = "monologg/koelectra-base-v3-discriminator"
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraModel.from_pretrained(model_name)

def get_sentence_embedding_mean(sentences, tokenizer, model, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Tokenize sentences
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding='max_length')
    
    # Move inputs to the device
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Move model to the device
    model = model.to(device)

    # Process the sentences in batches
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            input_ids = inputs['input_ids'][i:i+batch_size]
            attention_mask = inputs['attention_mask'][i:i+batch_size]
            token_type_ids = inputs['token_type_ids'][i:i+batch_size]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            embeddings.extend(batch_embeddings)

    return embeddings

# 시놉시스 문장 데이터
sentences = just_merged_df['synopsis']

##################### 문장 임베딩 생성 ####################
# embeddings = [get_sentence_embedding_mean(s, tokenizer, model) for s in sentences]

# np.save("embeddings.npy", embeddings)

# embeddings

# stacked_embeddings = np.stack(embeddings)

# stacked_embeddings = np.squeeze(stacked_embeddings, axis=1)

# np.save('stacked_embeddings.npy', stacked_embeddings)

# 코사인 유사도 계산
# similarity_matrix = cosine_similarity(stacked_embeddings)

# np.save('stacked_similarity_matrix.npy', similarity_matrix)

# 시놉시스에 대한 코사인 유사도 불러오기

stacked_similarity_matrix = np.load('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/stacked_similarity_matrix.npy')

# 타이틀 이름을 인덱스로 변환하는 딕셔너리
title_to_index = dict(zip(just_merged_df['title'], just_merged_df.index))

# 댓글 데이터
comments = watcha_merged_df['comment']

##### replace로 특수문자 /n등 바꿀것
comments = comments.str.replace('\n', ' ')

def remove_special_chars(element):
    return re.sub('[^A-Za-z0-9가-힣\s]+', '', str(element))

comments = comments.apply(remove_special_chars)

comments = comments.replace('', np.nan).dropna()

############# 댓글 유사도 생성 ###############
# 문장 임베딩 생성
# watcha_comment_embeddings = [get_sentence_embedding_mean(s, tokenizer, model) for s in sentences]

# st_watcha_comment_embeddings = np.stack(watcha_comment_embeddings)

# st_watcha_comment_embeddings.shape

# st_watcha_comment_embeddings = np.squeeze(st_watcha_comment_embeddings, axis=1)

# np.save('watcha_comment_embeddings.npy', watcha_comment_embeddings)

# watcha_comment_embeddings = np.load('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_comment_embeddings.npy')

# watcha_comment_embeddings = np.squeeze(watcha_comment_embeddings, axis=1)

# watcha_comment_embeddings.shape

# watcha_comment_similarity_matrix = cosine_similarity(watcha_comment_embeddings)

# watcha_comment_similarity_matrix.shape

# np.save('watcha_comment_similarity_matrix.npy', watcha_comment_similarity_matrix)

# 댓글 유사도 불러오기
watcha_comment_similarity_matrix = np.load('C:/Users/pgs66/Desktop/GoogleDrive/python/OTT_Project/watcha_comment_similarity_matrix.npy')

##### replace로 원본 watcha_merged_df['comment']에서 특수문자 /n등 삭제
watcha_merged_df['comment'] = watcha_merged_df['comment'].str.replace('\n', ' ')

watcha_merged_df['comment'] = watcha_merged_df['comment'].apply(remove_special_chars)

watcha_merged_df['comment'] = watcha_merged_df['comment'].replace('', 'sarwegfhntrewrfdfdqthntdkjyuly8iyt')

watcha_merged_df = watcha_merged_df.drop(watcha_merged_df[watcha_merged_df['comment'] == 'sarwegfhntrewrfdfdqthntdkjyuly8iyt'].index)

watcha_merged_df.reset_index(drop=True, inplace=True)

#### 1. 영화에 가중치 추가 input 영화 이름만 2. 문장 자체를 입력해서 그에 가까운 문장으로 가중치 추가
# 타이틀 이름 justwatch로 통일시키기위해 열 추가

watcha_merged_df['title_num'] = watcha_merged_df['title_num'].astype(int)

def same_title(title_num):
    return just_merged_df['title'].loc[title_num]

watcha_merged_df['same_title'] = watcha_merged_df['title_num'].apply(same_title)

#타이틀별로 인덱스가 몇번인지 딕셔너리 생성
#여기에서 인덱스의 갯수를 세서 총 몇개중 몇개가 있는지 알 수 있도록 함
title_to_index_comment = watcha_merged_df.groupby('same_title').apply(lambda x: x.index.to_list()).to_dict()

comment_nums = {}

for i in title_to_index_comment.keys():
    comment_nums[i] = len(title_to_index_comment[i])

##딕셔너리의 키와 벨류를 바꿔서 인덱스 몇번이 어떤 키인지 생성
title_to_index_comment_idx_dict = watcha_merged_df['same_title'].to_dict()

## 다른 변수들(장르, 연도, 러닝타임) 추가
genres_encoded = just_merged_df['genre'].str.get_dummies(',')

create_year_encoded = just_merged_df['create_year'].str.get_dummies(',')

running_time_encoded = just_merged_df['running_time'].str.get_dummies(',')

# 다른 변수들(장르, 연도, 러닝타임) 유사도 행렬
cosine_sim_genres = cosine_similarity(genres_encoded, genres_encoded)

cosine_sim_create_year = cosine_similarity(create_year_encoded, create_year_encoded)

cosine_sim_running_time = cosine_similarity(running_time_encoded, running_time_encoded)

# 영화 추천해주는 함수 정의
def get_recommendations(title, comment_weight = 0.02, synopsis_weight = 1.2, genre_weight = 0.045, year_weight = 0.001, running_time_weight = 0.001):
    
    try:
        idx = title_to_index[title]
        comment_idx = title_to_index_comment[title]

    except: 
        return '영화 DB에 없습니다. 다른 영화를 선택해 주세요'

    #영화의 유사도에 직접 더하는 방법??

    #영화의 댓글들 중에 비슷한 댓글을 단 영화가 몇개있나?
    # 0.97이 넘어가는 코멘트만 가져오기
    sim_scores_comment = []
    for i in comment_idx:
        sim_scores_comment.append(list(enumerate(watcha_comment_similarity_matrix[i])))

    for i in range(len(comment_idx)):
        sim_scores_comment[i] = list(filter(lambda x: x[1] >= 0.97, sim_scores_comment[i]))

    comment_indices = []
    for i in range(len(comment_idx)):
        comment_indices.extend([idx[0] for idx in sim_scores_comment[i]])

    comment_over_name = []
    for i in comment_indices:
        comment_over_name.append(title_to_index_comment_idx_dict[i])

    unique_values, value_counts = np.unique(comment_over_name, return_counts=True)

    # 리스트에 영화 인덱스 이름과 비슷한 댓글 수 / 총 댓글 수 * 가중치
    unique_values_idx = []
    for i, j in zip(unique_values, value_counts):
        unique_values_idx.append([title_to_index[i], (j/comment_nums[i])*comment_weight])

    # stacked_similarity_matrix = 미리 만들어놓은 시놉시스 유사도 행렬
    stacked_similarity_matrix_1 = np.copy(stacked_similarity_matrix*synopsis_weight)

    ## 다른 변수들(장르, 연도, 러닝타임) 추가
    weighted_cosine_sim  = cosine_sim_genres * genre_weight + cosine_sim_create_year * year_weight + cosine_sim_running_time * running_time_weight

    stacked_similarity_matrix_2 = weighted_cosine_sim + stacked_similarity_matrix_1

    # 유사도에 댓글 가중치를 더해준다
    stacked_similarity_matrix_2[idx][np.array(unique_values_idx)[:,0].astype(int)] += np.array(unique_values_idx)[:,1]

    sum_matrix = list(enumerate(stacked_similarity_matrix_2[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sum_matrix, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[0:20]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    return just_merged_df['title'].iloc[movie_indices]

get_recommendations('Interstelar')

