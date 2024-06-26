import numpy as np
import pandas as pd
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
    print('模型不存在')
    model = LotteryModel(input_shape)  # 創建新的模型實例  


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

# 調用預測函數
predict_next_numbers(model, features, drawings, window_size, top_n=10)