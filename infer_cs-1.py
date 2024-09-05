import numpy as np
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.regularizers import l1_l2
import pickle
from datetime import datetime
import time
import warnings

# 忽略 TensorFlow 的 UserWarning
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# 載入歷史彩票開獎號碼數據
lottery_data = pd.read_csv('539_results.csv')

# 提取每期開獎號碼
dd = lottery_data[['日期', 'NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values
date = lottery_data[['日期']].values
drawings = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values

print("歷史開獎號碼:\n", dd)

# 創建特徵
features = []

# 1. 對於每一期,統計每個號碼出現的次數作為特徵
for draw in drawings:
    feature = np.bincount(draw, minlength=40)[:39]  # 限制號碼範圍在1到39之間
    features.append(feature)

# 2. 計算最近 n 期每個號碼的平均出現次數作為特徵
window_size = 10
windowed_features = []
for i in range(len(features) - window_size):
    window = features[i:i+window_size]
    windowed_feature = np.mean(window, axis=0)
    windowed_features.append(windowed_feature)

# 3. 將號碼本身也作為特徵
single_draw_features = drawings[window_size:]

# 4. 新增特徵：版路走勢
# 連號特徵：某個號碼組合出現後會在下一期出現同樣號碼的機率
def consecutive_feature(drawings):
    consecutive_features = []
    for i in range(1, len(drawings)):
        consecutive_feature = (drawings[i] == drawings[i-1]).astype(int)
        consecutive_features.append(consecutive_feature)
    return consecutive_features

# 雙生組合特徵：兩個號碼一起同時出現的組合機率
def twin_combination_feature(drawings):
    twin_combination_features = []
    for i in range(len(drawings) - 1):
        twin_combination_feature = ((drawings[i] - drawings[i+1]) == 1).astype(int)
        twin_combination_features.append(twin_combination_feature)
    return twin_combination_features

# 三生組合特徵：三個號碼一起同時出現的組合機率
def triplet_combination_feature(drawings):
    triplet_combination_features = []
    for i in range(len(drawings) - 2):
        diff1 = drawings[i] - drawings[i+1]
        diff2 = drawings[i+1] - drawings[i+2]
        triplet_combination_feature = np.logical_and(diff1 == 1, diff2 == 1).astype(int)
        triplet_combination_features.append(triplet_combination_feature)
    return triplet_combination_features

# 斷龍特徵：斷龍是指某個號碼組合在出現後，下一期該號碼不再出現的情況。斷龍特徵可以透過連號特徵來計算。
def dragon_feature(consecutive_features):
    dragon_features = []
    for consecutive_feature in consecutive_features:
        dragon_feature = 1 - consecutive_feature
        dragon_features.append(dragon_feature)
    return dragon_features

# 計算連號特徵
consecutive_features = consecutive_feature(drawings)
# 計算雙生組合特徵
twin_combination_features = twin_combination_feature(drawings)
# 計算三生組合特徵
triplet_combination_features = triplet_combination_feature(drawings)
# 計算斷龍特徵
dragon_features = dragon_feature(consecutive_features)

# 獲取所有特徵長度的最小值
min_len = min(len(features[window_size:]), len(windowed_features), len(single_draw_features), 
              len(consecutive_features), len(twin_combination_features),
              len(triplet_combination_features), len(dragon_features))

# 對長度較長的特徵陣列進行裁剪
features = features[window_size:][:min_len]
windowed_features = windowed_features[:min_len]
single_draw_features = single_draw_features[:min_len]
consecutive_features = consecutive_features[:min_len]
twin_combination_features = twin_combination_features[:min_len]
triplet_combination_features = triplet_combination_features[:min_len]
dragon_features = dragon_features[:min_len]

# 5. 拼接特徵
X = np.concatenate((features, windowed_features, single_draw_features, consecutive_features, 
                    twin_combination_features, triplet_combination_features, dragon_features), axis=1)

# 5. 標記樣本為1個時間視窗後的中獎號碼
y = drawings[window_size:]

# 將標簽編碼為一個熱向量
y = np.array([to_categorical(label-1, num_classes=39) for label in y])

# 重塑標簽形狀為 (samples, 5, 39)
y = y.reshape(y.shape[0], 5, 39)

# 切分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 列印訓練集輸入特徵向量的形狀
print("訓練集輸入特徵向量形狀:", X_train.shape)

# 定義自定義激活函數
def custom_activation(x):
    return (tf.clip_by_value(x, 0, 1) * 38) + 1

# 將自定義激活函數註冊到TensorFlow中
get_custom_objects().update({'custom_activation': tf.keras.activations.get(custom_activation)})

# 修改模型創建函數，與訓練時保持一致
def create_model(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(512, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.5),
        Dense(128, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.5),
        Dense(64, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.5),
        Dense(5 * 39, activation=custom_activation, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        Reshape((5, 39)),
        Activation('softmax')
    ])
    return model

# 打印模型結構
def print_model_structure(model):
    model.summary()

# 加載模型權重
epoc = int(input(f'載入模型步數: '))
model_weights_path = f'best_checkpoint/best_lottery_model_{epoc}.weights.h5'

if os.path.exists(model_weights_path):
    model = create_model(X.shape[1])
    print("模型結構：")
    print_model_structure(model)
    try:
        model.load_weights(model_weights_path)
        print(f"模型權重 {model_weights_path} 已成功加載")
    except ValueError as e:
        print(f"無法加載權重，錯誤信息: {str(e)}")
        exit()
else:
    print('模型不存在')
    exit()

# 獲取指定日期後一期的日期
def get_next_date(date, lottery_data):    
    next_date = None
    try:
        if date.strip() == "":
            next_date = lottery_data['日期'].iloc[0]            
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            if index < len(lottery_data) - 1:
                next_date = lottery_data['日期'].iloc[index + 1]
    except IndexError:        
        pass
    return next_date

# 獲取指定日期的開獎號碼
def get_numbers(date, lottery_data):    
    numbers = None
    try:
        if date.strip() == "":
            numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[0].values
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[index].values
    except IndexError:
        pass
    return numbers

# 獲取指定日期或第一期的開獎號碼
specified_date = input("請輸入指定日期（格式：YYYY/MM/DD）：")

# 獲取指定日期或第一期的開獎號碼
current_date = specified_date if specified_date.strip() != "" else lottery_data['日期'].iloc[0]
current_numbers = get_numbers(current_date, lottery_data)

# 打開 CSV 檔案以寫入模式
if os.path.exists('infer_cs_results.csv'):
    with open('infer_cs_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        f.truncate()

# 開啟 CSV 檔案以附加模式
def save_csv(current_date, current_numbers, predicted_numbers, matched_numbers):    
    with open('infer_cs_results.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['預測日期', '實際開獎號碼', '預測開獎號碼範圍', '對中的號碼數量', '對中的號碼']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
        
        if csvfile.tell() == 0:
            writer.writeheader()  
        
        writer.writerow({
            '預測日期': current_date, 
            '實際開獎號碼': current_numbers, 
            '預測開獎號碼範圍': predicted_numbers, 
            '對中的號碼數量': len(matched_numbers), 
            '對中的號碼': matched_numbers if matched_numbers else ''  # 如果沒有對中號碼，則留空
        })

# 記錄開始時間
start_time = time.time()

# 預測函數
def predict_next_numbers(model, features, drawings, window_size, top_n):
    last_feature = features[-1]
    windowed_features = features[-window_size:]
    windowed_feature = np.mean(windowed_features, axis=0)
    last_vector = to_vector(drawings[-1])

    last_feature = np.expand_dims(last_feature, axis=0)
    windowed_feature = np.expand_dims(windowed_feature, axis=0)
    last_vector = np.expand_dims(last_vector, axis=0)
    
    input_features = np.concatenate((last_feature, windowed_feature, last_vector, 
                                      np.array(consecutive_features)[-1:], 
                                      np.array(twin_combination_features)[-1:], 
                                      np.array(triplet_combination_features)[-1:], 
                                      np.array(dragon_features)[-1:]), axis=1)

    predicted_probs = model.predict(input_features)[0]
    
    predicted_probs /= np.sum(predicted_probs)
    
    sorted_indices = np.argsort(-predicted_probs).flatten()
    
    top_numbers = [i+1 for i in sorted_indices[:top_n]]
    
    return top_numbers

# 獲取開始日期在資料中的索引
start_date = specified_date
if start_date.strip() == "":
    start_date_index = 0
    print("未指定開始日期，將從第一筆資料開始。")
else:
    start_date_index = np.where(date == start_date)[0][0] if start_date in date else None 

# 總預測期數
if start_date_index is not None:
    draws_predicted = len(date) - start_date_index

# 預測號碼數量
top_n = int(input('預測號碼數量： '))

# 調用預測函數 
if start_date_index is None:
    print(f"找不到指定日期 {start_date} 的開獎號碼或資料錯誤，將不進行預測。")
else:
    for i in range(start_date_index, len(date)):
        def to_vector(numbers):
            vector = np.zeros(5)
            for i, num in enumerate(numbers):
                if 1 <= num <= 39:
                    vector[i] = 1
            return vector

        current_date = date[i][0]
        current_numbers = drawings[i]
        print(f"正在預測日期 {current_date} 的開獎號碼")
        
        predicted_numbers = predict_next_numbers(model, features[:i], drawings[:i], window_size, top_n)
        print(f"預測本期539開獎號碼範圍為: {predicted_numbers}")
        
        matched_numbers = set(predicted_numbers).intersection(set(current_numbers))
        print(f"預測號碼與實際開獎號碼對中的號碼數量為: {len(matched_numbers)}")
        if len(matched_numbers) != 0:
            print(f"對中的號碼為: {matched_numbers}")
            matched_numbers_to_save = matched_numbers
        else:
            print("未對中任何號碼")
            matched_numbers_to_save = set()  # 使用空集合代表沒有對中號碼

        save_csv(current_date, current_numbers, predicted_numbers, matched_numbers_to_save)
        print("------------------------------------------------------")

if start_date_index is not None:
    print(f"已達到資料末端，總預測期數 = {draws_predicted}")

print("結果已保存至 infer_cs_results.csv")

# 記錄結束時間
end_time = time.time()

# 計算執行時間
elapsed_time = end_time - start_time

# 格式化時間輸出
elapsed_formatted = time.strftime("%M:%S", time.gmtime(elapsed_time))

print(f"代碼執行完成，總耗時: {elapsed_formatted}")