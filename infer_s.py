import numpy as np
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import pickle
from datetime import datetime
import time

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

# 4. 組合以上特徵
X = np.concatenate((features[window_size:], windowed_features, single_draw_features), axis=1)

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

# 定義模型架構
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # 添加Input層
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5 * 39, activation=custom_activation),
    Reshape((5, 39)),
    Activation('softmax')  # 使用softmax激活函數
])

# 加載模型權重
epoc = int(input(f'載入模型步數: '))
try:                      # 使用 try，測試內容是否正確
    model_weights_path = f'my_lottery_model_{epoc}.weights.h5'
    new_model = Sequential.from_config(model.get_config())
    new_model.load_weights(model_weights_path)
    print(f"模型權重 {model_weights_path} 已加載")
    # 加載自定義對象
    custom_objects_path = 'custom_objects.pkl'
    with open(custom_objects_path, 'rb') as file:
       custom_objects = pickle.load(file)
    new_model.make_predict_function(custom_objects)  # 重新編譯模型並加載自定義對象
    print(f"自定義對象從 {custom_objects_path} 已加載")
except:                   # 如果 try 的內容發生錯誤，就執行 except 裡的內容
    print('模型不存在 \n盲選')


# 獲取指定日期前一期的日期
def get_previous_date(date, lottery_data):    
    previous_date = None
    try:
        if date.strip() == "":
            previous_date = lottery_data['日期'].iloc[1]            
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            if index > 0:
                previous_date = lottery_data['日期'].iloc[index - 1]
    except IndexError:        
        # 異常發生時不做任何動作，直接跳過
        pass
    return previous_date

# 獲取指定日期前一期的開獎號碼
def get_previous_numbers(date, lottery_data):    
    previous_numbers = None
    try:
        if date.strip() == "":
            previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[1].values
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            if index > 0:
                previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[index - 1].values
    except IndexError:
        # 異常發生時不做任何動作，直接跳過
        pass
    return previous_numbers

# 指定日期
specified_date = input("請輸入指定日期（格式：YYYY/MM/DD）：")

# 獲取指定日期前一期或最後一期的開獎號碼
previous_date = get_previous_date(specified_date, lottery_data)
previous_numbers = get_previous_numbers(specified_date, lottery_data)

# 打開 CSV 檔案以寫入模式
# 如果檔案已經存在，先將其清空
if os.path.exists('infer_s_results.csv'):
    with open('infer_s_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        f.truncate()

# 開啟 CSV 檔案以附加模式
def save_csv(specified_date, specified_date_numbers, top_numbers, matched_numbers):    
    with open('infer_s_results.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        # 定義 CSV 欄位名稱
        fieldnames = ['對獎日期', '開獎號碼', '預測開獎號碼範圍', '對中的號碼數量', '對中的號碼']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
        
        # 如果檔案是新建立的，則寫入標題行
        if csvfile.tell() == 0:
            writer.writeheader()  
        
        # 寫入資料行
        writer.writerow({'對獎日期': specified_date, '開獎號碼': specified_date_numbers, '預測開獎號碼範圍': top_numbers, '對中的號碼數量': len(matched_numbers), '對中的號碼': matched_numbers})

# 記錄開始時間
start_time = time.time()  
     
# 預測號碼數量
top_n = int(input('預測號碼數量： '))

# 預測函數
def predict_next_numbers(model, features, drawings, window_size, top_n):
    # 構造預測輸入特徵向量
    last_feature = features[-1]
    windowed_features = features[-window_size:]
    windowed_feature = np.mean(windowed_features, axis=0)
    input_features = np.concatenate((last_feature, windowed_feature, last_vector))
    
    # 獲取預測概率
    predicted_probs = model.predict(np.array([input_features]))[0]
    
    # 對預測概率進行處理,確保和為1
    predicted_probs /= np.sum(predicted_probs)
    
    # 獲取預測概率從大到小排序的索引
    sorted_indices = np.argsort(-predicted_probs).flatten()
    
    # 獲取前 top_n 個概率最高的號碼
    top_numbers = [i+1 for i in sorted_indices[:top_n]]
    
    # 輸出預測的號碼範圍
    print(f"預測本期539開獎號碼範圍為: {top_numbers}")
    
    # 比對指定日期的開獎號碼
    specified_date_numbers = None
    if specified_date in date:
        index = np.where(date == specified_date)[0][0]
        specified_date_numbers = drawings[index]        
    
    # 進行對獎
    if specified_date_numbers is not None:
        matched_numbers = set(top_numbers).intersection(set(specified_date_numbers))
        print(f"預測號碼與指定日期開獎號碼對中的號碼數量為: {len(matched_numbers)}")
        if(len(matched_numbers)!=0):
          print(f"對中的號碼為: {matched_numbers}\n------------------------------------------------------") 
        else:
          print("未對中任何號碼\n------------------------------------------------------")

    # 保存至 infers_results.csv 檔案中
    if len(matched_numbers) != 0:
        save_csv(specified_date, specified_date_numbers, top_numbers, matched_numbers)
    else:
        save_csv(specified_date, specified_date_numbers, top_numbers, "")

# 獲取開始日期在資料中的索引
start_date = specified_date
if start_date.strip() == "":
    start_date_index = 1 
    print("未指定開始日期，將從第二筆資料開始。")
else:
    start_date_index = np.where(date == start_date)[0][0] if start_date in date else None 

# 總預測期數
if start_date_index is not None:
    draws_predicted = len(date) - start_date_index + 1

# 調用預測函數 
if start_date_index is None:
    print(f"找不到指定日期 {start_date} 的開獎號碼或資料錯誤，將不對獎。")
else:
    for i in range(start_date_index, len(date)):    
       # 將最後一期開獎號碼轉換為向量形式
       def to_vector(numbers):
           vector = np.zeros(5)
           for i, num in enumerate(numbers):
               if 1 <= num <= 39:
                   vector[i] = 1
           return vector

       # 將前一期開獎號碼向量化，使其長度為5
       p_date = date[i-1]
       p_date_numbers = drawings[i-1]
       specified_date = date[i]
       specified_date_numbers = drawings[i]
       last_vector = to_vector(previous_numbers)      
       # print(f"對獎日期前期 {p_date} 的開獎號碼: {p_date_numbers}")
       print(f"正在對獎日期 {specified_date} 的開獎號碼: {specified_date_numbers}")
       predict_next_numbers(model, features, drawings, window_size, top_n)     

if start_date_index is not None:
    print("已達到資料底端，總預測期數 = ", draws_predicted) # 總預測期數

print("結果已保存至 infer_s_result.csv")

# 記錄結束時間
end_time = time.time()

# 計算執行時間
elapsed_time = end_time - start_time

# 格式化時間輸出
elapsed_formatted = time.strftime("%M:%S", time.gmtime(elapsed_time))

print(f"代碼執行完成，總耗時: {elapsed_formatted}")