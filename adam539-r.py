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
    epoc = 0
    model = LotteryModel(input_shape)  # 創建新的模型實例  

epo = int(input('輸入訓練迭代次數: '))

# 編譯模型
model.compile(optimizer='adam', loss=combined_loss_with_regularization, metrics=['accuracy'])

# 訓練模型並獲取歷史記錄
history = model.fit(X_train, y_train, epochs=epo, batch_size=32, validation_data=(X_test, y_test))

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
model_weights_path = f'my_lottery_model-R_{ep}.weights.h5'
model.save_weights(model_weights_path)
print(f"Epoch: {epo} \n模型權重已保存至 {model_weights_path}")

# 保存自定義對象
custom_objects_path = 'custom_objects-r.pkl'
with open(custom_objects_path, 'wb') as file:
    pickle.dump(get_custom_objects(), file)
print(f"自定義對象已保存至 {custom_objects_path}")

# 保存模型架構為 JSON 文件
model_json = model.to_json()
with open('model_architecture-r.json', 'w') as json_file:
    json_file.write(model_json)
print(f"模型架構已保存至 model_architecture-r.json")

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