import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import get_custom_objects
import pickle
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

# 6. 標記樣本為1個時間視窗後的中獎號碼
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

# 將自定義激活函數註冊到 TensorFlow 中
get_custom_objects().update({'custom_activation': tf.keras.activations.get(custom_activation)})

# 定義自定義模型
@keras.utils.register_keras_serializable(package='Custom')
class LotteryModel(tf.keras.Model):
    def __init__(self, input_dim):
        super(LotteryModel, self).__init__()
        self.input_dim = input_dim
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(5 * 40)  # 不指定激活函數
        self.reshape = Reshape((5, 40))
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
    model_weights_path = f'my_lottery_model-rt_{epoc}.weights.h5'
    
    # 創建新的模型實例
    model = LotteryModel(input_shape)
    
    # 加載舊有的模型權重
    model.load_weights(model_weights_path)
    print(f"模型權重 {model_weights_path} 已加載")

except Exception as e:
    print(f'發生錯誤: {e}')
    print('模型不存在，從新訓練模型')
    epoc = 0
    model = LotteryModel(input_shape)  # 創建新的模型實例  

epo = int(input('輸入訓練週期數: '))

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
    
    # 添加L2正則化項
    reg_loss = tf.reduce_sum(model.losses)
    
    return cross_entropy + reg_loss

# 3.動態調整損失函數和評估指標權重
def dynamic_loss(y_true, y_pred):
    # 計算交叉熵損失
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    # 計算準確率
    accuracy = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    
    # 獲取當前訓練步數
    iterations = tf.keras.backend.get_value(model.optimizer.iterations)
    
    # 計算當前epoch
    epoch = iterations // steps_per_epoch
    
    # 動態調整權重
    loss_weight = tf.maximum(0.8 - 0.005 * tf.cast(epoch, tf.float32), 0.2)
    acc_weight = 1.0 - loss_weight
    
    # 組合損失函數
    combined_loss = loss_weight * cross_entropy - acc_weight * accuracy
    
    return combined_loss

steps_per_epoch = len(lottery_data) // 128

# 編譯模型
model.compile(optimizer='adam', loss=combined_loss_with_regularization, metrics=['accuracy'])

# 記錄開始時間
start_time = time.time()  

# 訓練模型並獲取歷史記錄
history = model.fit(X_train, y_train, epochs=epo, batch_size=128, validation_data=(X_test, y_test))

# 獲取訓練過程中的損失值和準確率
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 找出最低驗證集損失值和最高驗證集準確率對應的週期數
min_val_loss_epoch = np.argmin(val_loss) + 1
max_val_acc_epoch = np.argmax(val_acc) + 1

# 列印結果
print(f'最低驗證集損失值: {np.min(val_loss):.4f}, 出現在第 {min_val_loss_epoch} 個週期')
print(f'最高驗證集準確率: {np.max(val_acc):.4f}, 出現在第 {max_val_acc_epoch} 個週期')

# 保存模型權重
ep = epoc + epo
model_weights_path = f'my_lottery_model-rt_{ep}.weights.h5'
model.save_weights(model_weights_path)
print(f"Epoch: {epo} \n模型權重已保存至 {model_weights_path}")

# 保存自定義對象
custom_objects_path = 'custom_objects-rt.pkl'
with open(custom_objects_path, 'wb') as file:
    pickle.dump(get_custom_objects(), file)
print(f"自定義對象已保存至 {custom_objects_path}")

# 保存模型架構為 JSON 文件
model_json = model.to_json()
with open('model_architecture-rt.json', 'w') as json_file:
    json_file.write(model_json)
print(f"模型架構已保存至 model_architecture-rt.json")

# 記錄結束時間
end_time = time.time()

# 計算執行時間
elapsed_time = end_time - start_time

# 格式化時間輸出
elapsed_formatted = time.strftime("%M:%S", time.gmtime(elapsed_time))

print(f"代碼執行完成，總耗時: {elapsed_formatted}")

#預測獎號
# 獲取最後一期開獎號碼
last_numbers = drawings[-1]

# 將最後一期開獎號碼轉換為向量形式
def to_vector(numbers):
    vector = np.zeros(5)
    for i, num in enumerate(numbers):
        if 1 <= num <= 39:
            vector[i] = 1
    return vector

# 將最後一期開獎號碼向量化，使其長度為5
last_vector = to_vector(last_numbers)
print("上一期開獎號碼: ", dd[-1])

def predict_next_numbers(model, features, drawings, window_size, top_n):
    last_feature = features[-1]  # 最後一期的特徵
    windowed_features = features[-window_size:]  # 視窗特徵
    windowed_feature = np.mean(windowed_features, axis=0)  # 計算視窗特徵的平均值
    last_vector = to_vector(drawings[-1])  # 最後一期開獎號碼的向量表示

    # 確保所有特徵都是二維的，以便可以使用 np.concatenate 進行合併
    last_feature = np.expand_dims(last_feature, axis=0)  # 將 last_feature 轉換為二維
    windowed_feature = np.expand_dims(windowed_feature, axis=0)  # 將 windowed_feature 轉換為二維
    last_vector = np.expand_dims(last_vector, axis=0)  # 將 last_vector 轉換為二維
    
    # 將計算出的特徵數組合併為一個特徵向量
    input_features = np.concatenate((last_feature, windowed_feature, last_vector, 
                                      np.array(consecutive_features)[-1:], 
                                      np.array(twin_combination_features)[-1:], 
                                      np.array(triplet_combination_features)[-1:], 
                                      np.array(dragon_features)[-1:]), axis=1)

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