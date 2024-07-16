import numpy as np
import pandas as pd

# 載入歷史彩票開獎號碼數據
lottery_data = pd.read_csv('539_results.csv')

# 提取每期開獎號碼
dd = lottery_data[['日期', 'NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values
date = lottery_data[['日期']].values
drawings = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values

print("歷史開獎號碼:\n", dd)

import numpy as np
import pandas as pd

# 載入歷史彩票開獎號碼數據
lottery_data = pd.read_csv('539_results.csv')

# 提取每期開獎號碼
dd = lottery_data[['日期', 'NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values
date = lottery_data['日期'].values
drawings = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].values

print("歷史開獎號碼:\n", dd)

# 獲取指定日期前一期的日期
def get_previous_date(date, lottery_data):
    previous_date = None
    try:
        if date.strip() == "":
            previous_date = lottery_data['日期'].iloc[-1]
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            if index > 0:
                previous_date = lottery_data['日期'].iloc[index - 1]
            else:
                previous_date = lottery_data['日期'].iloc[-1]
    except IndexError:
        # 異常發生時不做任何動作，直接跳過
        previous_date = lottery_data['日期'].iloc[-1]
    except KeyError:
        previous_date = lottery_data['日期'].iloc[-1]
    return previous_date

# 獲取指定日期前一期的開獎號碼
def get_previous_numbers(date, lottery_data):
    previous_numbers = None
    try:
        if date.strip() == "":
            previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[-1].values
        else:
            index = np.where(lottery_data['日期'] == date)[0][0]
            if index > 0:
                previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[index - 1].values
            else:
                previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[-1].values
    except IndexError:
        # 異常發生時不做任何動作，直接跳過
        previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[-1].values
    except KeyError:
        previous_numbers = lottery_data[['NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5']].iloc[-1].values
    return previous_numbers

# 獲取開始日期在資料中的索引
specified_date = input("請輸入指定日期（格式：YYYY/MM/DD）：")
start_date = specified_date

if start_date.strip() == "":
    start_date_index = 1 
    print("未指定開始日期，將從第二筆資料開始。")
else:
    start_date_index = np.where(date == start_date)[0][0] if start_date in date else None
    if start_date_index is None:
        start_date_index = len(date) - 1

# 獲取指定日期前一期或最後一期的開獎號碼
previous_date = get_previous_date(start_date, lottery_data)
previous_numbers = get_previous_numbers(start_date, lottery_data)

print(f"前一期的日期：{previous_date}")
print(f"前一期的開獎號碼：{previous_numbers}")

