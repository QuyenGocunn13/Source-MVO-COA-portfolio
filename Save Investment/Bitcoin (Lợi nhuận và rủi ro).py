import pandas as pd
import numpy as np

# Đọc file CSV
df = pd.read_csv('Bitcoin.csv')

# Chuyển các giá trị trong cột thành số (loại bỏ dấu phẩy)
#df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
df['Change %'] = df['Change %'].replace({'%': '', ',': ''}, regex=True).astype(float) / 100

# Tính lợi nhuận phần trăm hàng ngày từ cột 'Change %'
df['Daily Return'] = df['Change %']

# Tính lợi nhuận trung bình hàng ngày
average_daily_return = df['Daily Return'].mean()

# Annualized Return (Lợi nhuận hàng năm)
annualized_return = (1 + average_daily_return) ** 252 - 1

# Tính độ lệch chuẩn hàng ngày (rủi ro hàng ngày)
daily_volatility = df['Daily Return'].std()

# Annualized Volatility (Độ biến động hàng năm)
annualized_volatility = daily_volatility * np.sqrt(252)

# In kết quả
print(f"Lợi nhuận hàng năm (Annualized Return): {annualized_return * 100:.2f}%")
print(f"Độ biến động hàng năm (Annualized Volatility): {annualized_volatility * 100:.2f}%")

# Lưu kết quả vào file CSV mới (nếu cần)
#df.to_csv('processed_data.csv', index=False)
