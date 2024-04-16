import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
import pickle

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

# 将标签编码为一个热向量
y = np.array([to_categorical(label-1, num_classes=40) for label in y])

# 重塑标签形状为 (samples, 5, 40)
y = y.reshape(y.shape[0], 5, 40)

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
    Dense(5 * 40, activation='relu'),
    Reshape((5, 40)),  # 将输出重塑为 (5, 40) 的形状
    Dense(40, activation='softmax')  # 输出维度为39,每个号码是一个39分类问题
])

# 加載模型權重
epoc = int(input(f'載入模型步數: '))
try:                      # 使用 try，測試內容是否正確
    model_weights_path = f'my_lottery_model-t_{epoc}.weights.h5'
    new_model = Sequential.from_config(model.get_config())
    new_model.load_weights(model_weights_path)
    print(f"模型權重 {model_weights_path} 已加載")
    # 加載自定義對象
    custom_objects_path = 'custom_objects-t.pkl'
    with open(custom_objects_path, 'rb') as file:
       custom_objects = pickle.load(file)
    new_model.make_predict_function(custom_objects)  # 重新編譯模型並加載自定義對象
    print(f"自定義對象從 {custom_objects_path} 已加載")
except:                   # 如果 try 的內容發生錯誤，就執行 except 裡的內容
    print('模型不存在 \n盲選')


#預測獎號
# 獲取指定日期前一期的日期
def get_previous_date(date, lottery_data):    
    previous_date = None
    try:
        index = np.where(lottery_data['日期'] == date)[0][0]
        if index > 0:
            previous_date = lottery_data[['日期']].iloc[index - 1].values
    except IndexError:        
        print("找不到指定日期")
    return previous_date

# 獲取指定日期前一期的開獎號碼
def get_previous_numbers(date, lottery_data):    
    previous_numbers = None
    try:
        index = np.where(lottery_data['日期'] == date)[0][0]
        if index > 0:
            previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[index - 1].values
    except IndexError:
        print("將使用最後一期開獎號碼作為參考")
    return previous_numbers

# 獲取指定日期前一期或最後一期的開獎號碼
specified_date = input("請輸入指定日期（格式：YYYY/MM/DD）：")
previous_date = get_previous_date(specified_date, lottery_data)
previous_numbers = get_previous_numbers(specified_date, lottery_data)

if previous_numbers is None:
    previous_numbers = drawings[-1]
    print("使用最後一期開獎號碼作為參考：",date[-1], previous_numbers)
else:
    print("指定日期前一期開獎號碼為：",previous_date, previous_numbers)

# 將最後一期開獎號碼轉換為向量形式
def to_vector(numbers):
    vector = np.zeros(5)
    for i, num in enumerate(numbers):
        if 1 <= num <= 39:
            vector[i] = 1
    return vector

# 將最後一期開獎號碼向量化，使其長度為5
last_vector = to_vector(previous_numbers)

# 預測號碼數量
top_n = int(input('預測號碼數量： '))
     
# 預測函數
window_size = 10  # 与训练时使用的 window_size 值相同

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
    print(f"預測下一期539開獎號碼範圍為: {top_numbers}")
    
    # 比對指定日期的開獎號碼
    specified_date_numbers = None
    if specified_date in date:
        index = np.where(date == specified_date)[0][0]
        specified_date_numbers = drawings[index]
        print(f"指定日期 {specified_date} 的開獎號碼為: {specified_date_numbers}")
    
    # 進行對獎
    if specified_date_numbers is not None:
        matched_numbers = set(top_numbers).intersection(set(specified_date_numbers))
        print(f"預測號碼與指定日期開獎號碼對中的號碼數量為: {len(matched_numbers)}")
        if(len(matched_numbers)!=0):
          print(f"對中的號碼為: {matched_numbers}")

# 調用預測函數
predict_next_numbers(model, features, drawings, window_size, top_n)