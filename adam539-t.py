import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
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

# 创建特征
features = []

# 1. 对于每一期,统计每个号码出现的次数作为特征
for draw in drawings:
    feature = np.bincount(draw, minlength=40)[:39]  # 限制号码范围在1到39之间
    features.append(feature)

# 2. 计算最近 n 期每个号码的平均出现次数作为特征
window_size = 10
windowed_features = []
for i in range(len(features) - window_size):
    window = features[i:i+window_size]
    windowed_feature = np.mean(window, axis=0)
    windowed_features.append(windowed_feature)

# 3. 将号码本身也作为特征
single_draw_features = drawings[window_size:]

# 4. 组合以上特征
X = np.concatenate((features[window_size:], windowed_features, single_draw_features), axis=1)

# 5. 标记样本为1个时间窗口后的中奖号码
y = drawings[window_size:]

# 将标签编码为一个热向量
y = np.array([to_categorical(label-1, num_classes=40) for label in y])

# 重塑标签形状为 (samples, 5, 40)
y = y.reshape(y.shape[0], 5, 40)

# 切分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集输入特征向量的形状
print("训练集输入特征向量形状:", X_train.shape)

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
    epoc = 0
    print('從新訓練模型')

# 訓練模型
epo = int(input('輸入訓練週期數: '))

# 記錄開始時間
start_time = time.time()   

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
model_weights_path = f'my_lottery_model-t_{ep}.weights.h5'
model.save_weights(model_weights_path)
print(f"Epoch: {epo} \n模型權重已保存至 {model_weights_path}")

# 保存自定義對象
custom_objects_path = 'custom_objects-t.pkl'
with open(custom_objects_path, 'wb') as file:
    pickle.dump(get_custom_objects(), file)
print(f"自定義對象已保存至 {custom_objects_path}")

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

# 将最后一期开奖号码转换为向量形式
def to_vector(numbers):
    vector = np.zeros(5)
    for i, num in enumerate(numbers):
        if 1 <= num <= 39:
            vector[i] = 1
    return vector

# 将最后一期开奖号码向量化，使其长度为5
last_vector = to_vector(last_numbers)
print("上一期开奖号码: ", last_numbers)

window_size = 10  # 与训练时使用的 window_size 值相同

def predict_next_numbers(model, features, drawings, window_size):
    # 構造預測輸入特徵向量
    last_feature = features[-1]
    windowed_features = features[-window_size:]
    windowed_feature = np.mean(windowed_features, axis=0)
    input_features = np.concatenate((last_feature, windowed_feature, last_vector))

    # 打印預測輸入特徵向量的形狀和值
    print("Last_feature: ", len(last_feature), ", Windowed_feature: ", len(windowed_feature), ", Last Vector: ", len(last_vector))
    print("預測輸入特徵向量形狀:", input_features.shape)
    print("預測輸入特徵向量值:\n", input_features)

    # 預測下一期開獎號碼的概率分佈
    predicted_probs = model.predict(np.array([input_features]))[0]
    
    # 從概率分佈中選擇最大的5個概率對應的號碼
    top_indices = np.argsort(predicted_probs).flatten()[-5:][::-1] + 1
    predicted_numbers = [num for num in top_indices if 1 <= num <= 39]

    # 輸出預測結果
    print(f"預測下一期539開獎號碼為: {predicted_numbers}")


# 调用预测函数
predict_next_numbers(model, features, drawings, window_size)