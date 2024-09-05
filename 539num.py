import requests
from bs4 import BeautifulSoup
import csv
import os
from datetime import datetime, timedelta

def get_latest_date_from_csv(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳過標題行
        dates = []
        for row in reader:
            if row and row[0].strip():  # 確保行不為空且日期欄位有值
                try:
                    dates.append(datetime.strptime(row[0].strip(), '%Y/%m/%d'))
                except ValueError:
                    print(f"警告：無法解析日期 '{row[0]}'，已跳過此行")
    return max(dates) if dates else None

def fetch_and_update_data(url, file_path):
    latest_date = get_latest_date_from_csv(file_path)
    page = 1
    new_data = []
    print('正在擷取歷史資料...')
    
    while True:
        print(f"正在處理第 {page} 頁...")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        draw_dates = soup.find_all('span', style="font-size:18px")
        draw_numbers = soup.find_all('span', id=lambda x: x and x.startswith('ltono'))

        for date, num1, num2, num3, num4, num5 in zip(draw_dates, draw_numbers[::5], draw_numbers[1::5], draw_numbers[2::5], draw_numbers[3::5], draw_numbers[4::5]):
            date_str = date.text.strip()
            try:
                current_date = datetime.strptime(date_str, '%Y/%m/%d')
            except ValueError:
                print(f"警告：無法解析日期 '{date_str}'，已跳過此筆資料")
                continue

            if latest_date and current_date <= latest_date:
                print(f"已達到最新資料（{latest_date.strftime('%Y/%m/%d')}），停止更新。")
                return new_data

            nums = [num1.text, num2.text, num3.text, num4.text, num5.text]
            new_data.append([date_str] + nums)
            print([date_str] + nums)

        if len(draw_dates) < 30 or len(draw_numbers) < 150:       
            break

        page += 1
        url = f"https://www.lotto-8.com/listlto539.asp?indexpage={page}&orderby=new"

    return new_data

# 設定初始網址和檔案路徑
url = "https://www.lotto-8.com/listlto539.asp?indexpage=1&orderby=new"
file_path = '539_results.csv'

# 執行更新
new_data = fetch_and_update_data(url, file_path)

if new_data:
    mode = 'a' if os.path.exists(file_path) else 'w'
    with open(file_path, mode, newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'w':
            writer.writerow(['日期', 'NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5'])
        writer.writerows(reversed(new_data))  # 反轉數據以保持日期順序
    print(f"新增了 {len(new_data)} 筆資料到 {file_path}")
else:
    print("沒有新的資料需要更新。")

print("開獎號碼資料已更新完成")