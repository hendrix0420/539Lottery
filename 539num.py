import requests
from bs4 import BeautifulSoup
import csv

# 設定初始網址
url = "https://www.lotto-8.com/listlto539.asp?indexpage=1&orderby=old"
page = 1

# 創建一個空列表來儲存所有資料
data = []
print('正在擷取歷史資料...')

while True:
    # 發送請求並獲取網頁內容
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 找到包含開獎號碼資料的HTML標籤
    draw_dates = soup.find_all('span', style="font-size:18px")
    draw_numbers = soup.find_all('span', id=lambda x: x and x.startswith('ltono'))

    # 遍歷每個開獎日期和對應的號碼
    for date, num1, num2, num3, num4, num5 in zip(draw_dates, draw_numbers[::5], draw_numbers[1::5], draw_numbers[2::5], draw_numbers[3::5], draw_numbers[4::5]):
        date_str = date.text.strip()
        nums = [num1.text, num2.text, num3.text, num4.text, num5.text]
        data.append([date_str] + nums)
        print([date_str] + nums)

    # 如果找到的開獎號碼數量少於30筆,則認為是最後一頁
    if len(draw_dates) < 30 or len(draw_numbers) < 150:
        # 遍歷最後一頁的開獎號碼資料
        for date, num1, num2, num3, num4, num5 in zip(draw_dates, draw_numbers[::5], draw_numbers[1::5], draw_numbers[2::5], draw_numbers[3::5], draw_numbers[4::5]):
            date_str = date.text.strip()
            nums = [num1.text, num2.text, num3.text, num4.text, num5.text]
            data.append([date_str] + nums)            
        break

    # 更新網址以獲取下一頁
    page += 1
    url = f"https://www.lotto-8.com/listlto539.asp?indexpage={page}&orderby=old"

# 將資料儲存為CSV文件
with open('539_results.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['日期', 'NUM1', 'NUM2', 'NUM3', 'NUM4', 'NUM5'])
    writer.writerows(data)   

print("開獎號碼資料已儲存為539_results.csv") 