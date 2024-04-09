import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.utils import to_categorical

epo = int(input('输入训练迭代次数: '))

# 载入历史彩票开奖号码数据
lottery_data = pd.read_csv('539_results.csv')

# 提取每期开奖号码
drawings = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values

print("历史开奖号码:\n", drawings)

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

# 训练模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(5 * 40, activation='relu'),
    Reshape((5, 40)),  # 将输出重塑为 (5, 40) 的形状
    Dense(40, activation='softmax')  # 输出维度为39,每个号码是一个39分类问题
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=epo, batch_size=32, validation_data=(X_test, y_test))

# 获取最后一期开奖号码
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
      
    # 构造预测输入特征向量
    last_feature = features[-1]
    windowed_features = features[-window_size:]
    windowed_feature = np.mean(windowed_features, axis=0)
    input_features = np.concatenate((last_feature, windowed_feature, last_vector))
    
    # 打印预测输入特征向量的形状和值
    print("Last_feature: ", len(last_feature), ", Windowed_feature: ", len(windowed_feature), ", Last Vector: ", len(last_vector))
    print("预测输入特征向量形状:", input_features.shape)
    print("预测输入特征向量值:\n", input_features)
    

    # 预测下一期开奖号码
    predicted_probs = model.predict(np.array([input_features]))[0]
    predicted_numbers = [i+1 for i, prob in enumerate(predicted_probs) if prob > 0.5]
    predicted_numbers = [num for num in predicted_numbers if 1 <= num <= 39]

    # 输出预测结果
    print(f"预测下一期539开奖号码为: {predicted_numbers}")

# 调用预测函数
predict_next_numbers(model, features, drawings, window_size)