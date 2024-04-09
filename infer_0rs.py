import numpy as np
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, Activation, Input
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import get_custom_objects
import pickle
from datetime import datetime

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


# 0.定義損失函數
def custom_loss(y_true, y_pred):
    # 計算每個樣本的交叉熵損失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)  # 返回批次中所有樣本損失的平均值

# 1.將評估指標納入損失函數
def combined_loss(y_true, y_pred):
    # 計算交叉熵損失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # 計算準確率
    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    
    # 組合損失函數,以0.5的權重平衡交叉熵損失和準確率
    combined_loss = 0.5 * cross_entropy - 0.5 * accuracy
    
    return combined_loss

# 2.使用正則化項
def combined_loss_with_regularization(y_true, y_pred):
    # 計算交叉熵損失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # 計算準確率
    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    
    # 組合損失函數,以0.5的權重平衡交叉熵損失和準確率
    combined_loss = 0.5 * cross_entropy - 0.5 * accuracy
    
    # 添加L2正則化項
    reg_loss = tf.reduce_sum(model.losses)
    
    return combined_loss + reg_loss

# 3.動態調整損失函數和評估指標權重
def dynamic_loss(y_true, y_pred, epoch):
    # 計算交叉熵損失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # 計算準確率
    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    
    # 動態調整權重
    loss_weight = max(0.8 - 0.005 * epoch, 0.2)
    acc_weight = 1 - loss_weight
    
    # 組合損失函數
    combined_loss = loss_weight * cross_entropy - acc_weight * accuracy
    
    return combined_loss

# 定義自定義激活函數
def custom_activation(x):
    return (tf.clip_by_value(x, 0, 1) * 38) + 1

# 將自定義激活函數註冊到 TensorFlow 中
get_custom_objects().update({'custom_activation': custom_activation})

# 定義自定義模型
@keras.utils.register_keras_serializable(package='Custom')
class LotteryModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(LotteryModel, self).__init__()
        self.input_dim = input_dim
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(5 * 39)  # 不指定激活函數
        self.reshape = Reshape((5, 39))
        self.custom_activation = Activation(custom_activation)  # 使用自定義激活函數

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.custom_activation(x)  # 使用自定義激活函數
        x = self.reshape(x)
        return x

    def get_config(self):
        config = super(LotteryModel, self).get_config()
        config.update({'input_dim': self.input_dim})
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop('input_dim')
        return cls(input_dim)

# 創建模型實例
input_shape = (X_train.shape[1],)

# 加載模型權重
epoc = int(input(f'載入模型步數: '))
try:
    model_weights_path = f'my_lottery_model-R_{epoc}.weights.h5'
    
    # 創建新的模型實例
    model = LotteryModel(input_shape)
    
    # 加載舊有的模型權重
    model.load_weights(model_weights_path)
    print(f"模型權重 {model_weights_path} 已加載")

except Exception as e:
    print(f'發生錯誤: {e}')
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
        # 異常發生時不做任何動作，直接跳過
        pass
    return previous_date

# 獲取指定日期前一期的開獎號碼
def get_previous_numbers(date, lottery_data):    
    previous_numbers = None
    try:
        index = np.where(lottery_data['日期'] == date)[0][0]
        if index > 0:
            previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[index - 1].values
    except IndexError:
        # 異常發生時不做任何動作，直接跳過
        pass
    return previous_numbers

# 獲取指定日期前一期或最後一期的開獎號碼
specified_date = input("請輸入指定日期（格式：YYYY/MM/DD）：")

# 獲取指定日期前一期或最後一期的開獎號碼
previous_date = get_previous_date(specified_date, lottery_data)
previous_numbers = get_previous_numbers(specified_date, lottery_data)

# 獲取開始日期在資料中的索引
start_date = specified_date
start_date_index = np.where(date == start_date)[0][0] if start_date in date else None 
draws_predicted = len(date) - start_date_index + 1

# 打開 CSV 檔案以寫入模式
# 如果檔案已經存在，先將其清空
if os.path.exists('infer_0rs_results.csv'):
    with open('infer_0rs_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        f.truncate()

# 開啟 CSV 檔案以附加模式
def save_csv(specified_date, specified_date_numbers, top_numbers, matched_numbers):    
    with open('infer_0rs_results.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        # 定義 CSV 欄位名稱
        fieldnames = ['對獎日期', '開獎號碼', '預測開獎號碼範圍', '對中的號碼數量', '對中的號碼']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
        
        # 如果檔案是新建立的，則寫入標題行
        if csvfile.tell() == 0:
            writer.writeheader()  
        
        # 寫入資料行
        writer.writerow({'對獎日期': specified_date, '開獎號碼': specified_date_numbers, '預測開獎號碼範圍': top_numbers, '對中的號碼數量': len(matched_numbers), '對中的號碼': matched_numbers})
        
        
# 循環直到資料底端
while start_date_index is not None and start_date_index < len(date):
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


    # 0.定義損失函數
    def custom_loss(y_true, y_pred):
        # 計算每個樣本的交叉熵損失
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return tf.reduce_mean(cross_entropy)  # 返回批次中所有樣本損失的平均值

    # 1.將評估指標納入損失函數
    def combined_loss(y_true, y_pred):
        # 計算交叉熵損失
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 計算準確率
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        
        # 組合損失函數,以0.5的權重平衡交叉熵損失和準確率
        combined_loss = 0.5 * cross_entropy - 0.5 * accuracy
        
        return combined_loss

    # 2.使用正則化項
    def combined_loss_with_regularization(y_true, y_pred):
        # 計算交叉熵損失
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 計算準確率
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        
        # 組合損失函數,以0.5的權重平衡交叉熵損失和準確率
        combined_loss = 0.5 * cross_entropy - 0.5 * accuracy
        
        # 添加L2正則化項
        reg_loss = tf.reduce_sum(model.losses)
        
        return combined_loss + reg_loss

    # 3.動態調整損失函數和評估指標權重
    def dynamic_loss(y_true, y_pred, epoch):
        # 計算交叉熵損失
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # 計算準確率
        accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
        
        # 動態調整權重
        loss_weight = max(0.8 - 0.005 * epoch, 0.2)
        acc_weight = 1 - loss_weight
        
        # 組合損失函數
        combined_loss = loss_weight * cross_entropy - acc_weight * accuracy
        
        return combined_loss

    # 定義自定義激活函數
    def custom_activation(x):
        return (tf.clip_by_value(x, 0, 1) * 38) + 1

    # 將自定義激活函數註冊到 TensorFlow 中
    get_custom_objects().update({'custom_activation': custom_activation})

    # 定義自定義模型
    @keras.utils.register_keras_serializable(package='Custom')
    class LotteryModel(tf.keras.Model):
        def __init__(self, input_dim):
            super(LotteryModel, self).__init__()
            self.input_dim = input_dim
            self.dense1 = Dense(128, activation='relu')
            self.dense2 = Dense(64, activation='relu')
            self.dense3 = Dense(5 * 39)  # 不指定激活函數
            self.reshape = Reshape((5, 39))
            self.custom_activation = Activation(custom_activation)  # 使用自定義激活函數

        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.custom_activation(x)  # 使用自定義激活函數
            x = self.reshape(x)
            return x

        def get_config(self):
            config = super(LotteryModel, self).get_config()
            config.update({'input_dim': self.input_dim})
            return config

        @classmethod
        def from_config(cls, config):
            input_dim = config.pop('input_dim')
            return cls(input_dim)

    # 創建模型實例
    input_shape = (X_train.shape[1],)  
    
    # 創建新的模型實例    
    model = LotteryModel(input_shape)    

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
        
        return top_numbers
        
    # 比對指定日期的開獎號碼
    specified_date_numbers = None
    if specified_date in date:
        index = np.where(date == specified_date)[0][0]
        specified_date_numbers = drawings[index]        
    
    # 將最後一期開獎號碼轉換為向量形式
    def to_vector(numbers):
       vector = np.zeros(5)
       for i, num in enumerate(numbers):
           if 1 <= num <= 39:
                vector[i] = 1
       return vector
    
    # 調用預測函數
    last_vector = to_vector(previous_numbers)
    p_date = date[start_date_index - 1]
    p_date_numbers = drawings[start_date_index - 1]
    specified_date = date[start_date_index]
    specified_date_numbers = drawings[start_date_index]   
    print(f"對獎日期前期 {p_date} 的開獎號碼: {p_date_numbers}")
    print(f"正在對獎日期 {specified_date} 的開獎號碼: {specified_date_numbers}")
    top_numbers = predict_next_numbers(model, features, drawings, window_size, top_n=10)   
    
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
          
    # 更新指定日期前一期的日期及開獎號碼
    specified_date = date[start_date_index]
    previous_date = specified_date
    previous_numbers = specified_date_numbers 

    # 更新起始日期索引
    start_date_index += 1

if start_date_index is not None:
    print("已達到資料底端，總預測期數 = ", draws_predicted) # 總預測期數
else:
    print("未找到指定日期，無法進行預測。")
    
print("結果已保存至 infer_0rs_result.csv")