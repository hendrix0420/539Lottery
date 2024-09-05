import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import pickle
import time
import os
import re
import glob
import warnings

# 忽略特定的警告
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# 如果您確實需要對 tf.data 操作啟用 eager execution，取消下面這行的註釋
# tf.data.experimental.enable_debug_mode()

# 移除這行，除非您確實需要它
# tf.config.run_functions_eagerly(True)

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

# 定義帶有預熱的循環學習率調度
class WarmupCyclicLR(LearningRateSchedule):
    def __init__(self, base_lr, max_lr, step_size, warmup_steps=1000):
        super(WarmupCyclicLR, self).__init__()
        self.base_lr = tf.cast(base_lr, dtype=tf.float32)
        self.max_lr = tf.cast(max_lr, dtype=tf.float32)
        self.step_size = tf.cast(step_size, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        
        # 預熱階段
        warmup_lr = self.base_lr + (self.max_lr - self.base_lr) * (step / self.warmup_steps)
        
        # 循環學習率階段
        cycle = tf.floor(1 + (step - self.warmup_steps) / (2 * self.step_size))
        x = tf.abs((step - self.warmup_steps) / self.step_size - 2 * cycle + 1)
        cyclic_lr = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0.0, (1 - x))
        
        # 使用 tf.cond 來選擇適當的學習率
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: cyclic_lr)

# 修改模型創建函數，增加正則化
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

# 創建保存檢查點的目錄
checkpoint1_dir = "checkpoint1"
best_checkpoint_dir = "best_checkpoint"
os.makedirs(checkpoint1_dir, exist_ok=True)
os.makedirs(best_checkpoint_dir, exist_ok=True)

# 定義保存模型的回調函數
class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_freq=100, checkpoint_dir='checkpoint1', initial_epoch=0):
        super(SaveModelCallback, self).__init__()
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.initial_epoch = initial_epoch

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = self.initial_epoch + epoch + 1
        if (current_epoch - self.initial_epoch) % self.save_freq == 0:
            save_path = os.path.join(self.checkpoint_dir, f'my_lottery_model_{current_epoch}.weights.h5')
            self.model.save_weights(save_path)
            print(f"\n模型權重已保存至 {save_path}")

# 詢問用戶是否要載入之前的模型
load_option = input("請選擇要加載的模型類型 (1: 特定步數模型, 2: 最佳模型, 3: 從頭開始訓練): ")

if load_option == '1':
    epoc = int(input('載入模型步數: '))
    model_weights_path = os.path.join('checkpoint1', f'my_lottery_model_{epoc}.weights.h5')
    total_epochs = epoc
elif load_option == '2':
    epoc = input('請輸入最佳模型步數 (直接按Enter使用最新的最佳模型): ')
    if epoc:
        model_weights_path = os.path.join('best_checkpoint', f'best_lottery_model_{epoc}.weights.h5')
    else:
        # 尋找最新的最佳模型
        best_models = glob.glob(os.path.join('best_checkpoint', 'best_lottery_model*.weights.h5'))
        if best_models:
            latest_model = max(best_models, key=os.path.getctime)
            model_weights_path = latest_model
            # 使用正則表達式從文件名中提取步數
            match = re.search(r'best_lottery_model_?(\d+)\.weights\.h5', os.path.basename(latest_model))
            if match:
                epoc = int(match.group(1))
            else:
                print("無法從文件名中提取步數，將從頭開始訓練")
                model_weights_path = None
                epoc = 0
        else:
            model_weights_path = None
    
    if not model_weights_path or not os.path.exists(model_weights_path):
        print(f"找不到指定的最佳模型，將從頭開始訓練")
        model_weights_path = None
        total_epochs = 0
    else:
        total_epochs = int(epoc)
        print(f"找到最佳模型：{model_weights_path}")
else:
    print('將從頭開始訓練')
    model_weights_path = None
    total_epochs = 0

# 創建模型
model = create_model(X_train.shape[1])

# 如果選擇載入模型，則嘗試載入權重
if model_weights_path and os.path.exists(model_weights_path):
    model.load_weights(model_weights_path)
    print(f"已加載模型權重: {model_weights_path}")
    print(f"當前累計訓練步數: {total_epochs}")
else:
    print('未找到模型權重文件或選擇從頭開始訓練')
    total_epochs = 0

# 詢問用戶要訓練的步數
new_epochs = int(input('請輸入要訓練的步數: '))

# 記錄開始時間
start_time = time.time()   

# 創建帶有預熱的循環學習率調度
lr_schedule = WarmupCyclicLR(base_lr=1e-6, max_lr=1e-4, step_size=2000, warmup_steps=1000)

# 使用循環學習率創建優化器
optimizer = Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 定義回調函數
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=500,
    restore_best_weights=True,
    verbose=1
)

# 設置最佳模型檢查點
best_checkpoint_path = os.path.join('best_checkpoint', f'best_lottery_model.weights.h5')
model_checkpoint = ModelCheckpoint(
    filepath=best_checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# 創建保存模型的回調函數
save_model_callback = SaveModelCallback(save_freq=100, checkpoint_dir='checkpoint1', initial_epoch=0)

# 增加批次大小
batch_size = 128  # 或者更大，如256

# 訓練模型
history = model.fit(
    X_train, y_train,
    initial_epoch=total_epochs,
    epochs=total_epochs + new_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint, save_model_callback],
    verbose=1
)

# 更新總訓練步數
total_epochs += new_epochs

# 重命名最佳模型檢查點，加入總步數
if os.path.exists(best_checkpoint_path):
    new_best_checkpoint_path = os.path.join('best_checkpoint', f'best_lottery_model_{total_epochs}.weights.h5')
    os.rename(best_checkpoint_path, new_best_checkpoint_path)
    print(f"最佳模型已保存為: {new_best_checkpoint_path}")

# 評估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"測試集損失: {test_loss:.4f}")
print(f"測試集準確率: {test_accuracy:.4f}")

# 記錄結束時間
end_time = time.time()

# 計算執行時間
elapsed_time = end_time - start_time

# 格式化時間輸出
elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

print(f"代碼執行完成，總耗時: {elapsed_formatted}")
print(f"當前累計訓練步數: {total_epochs}")

# 預測獎號
# 獲取最後一期開獎號碼
last_numbers = drawings[-1]

# 將最後一期開獎號碼轉換為向量形式
def to_vector(numbers):
    vector = np.zeros(39)
    for num in numbers:
        if 1 <= num <= 39:
            vector[num-1] = 1
    return vector

# 將最後一期開獎號碼向量化，使其長度為39
last_vector = to_vector(last_numbers)
print("上一期開獎號碼: ", dd[-1])

def predict_next_numbers(model, features, drawings, window_size, top_n):
    # 構造預測輸入特徵向量
    last_feature = features[-1]
    windowed_features = features[-window_size:]
    windowed_feature = np.mean(windowed_features, axis=0)
    input_features = np.concatenate((last_feature, windowed_feature, last_vector))
    
    # 確保輸入特徵向量的形狀與模型輸入形狀一致
    input_features = input_features[:103].reshape(1, -1)
    
    # 獲取預測概率
    predicted_probs = model.predict(input_features)[0]
    
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