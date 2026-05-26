import pandas as pd
import numpy as np

# Nạp bảng tính thô từ hệ thống
df_raw = pd.read_excel(filepath, header=None)

# Thuật toán khử lỗi tự động chuyển đổi ngày tháng của Excel
def fix_date_to_score(val):
    if pd.isna(val): return np.nan
    if hasattr(val, 'day'): return float(val.day)
    return float(val)

# Tính toán ma trận 7 đặc trưng chính (Pandas tự động bỏ qua NaN)
features = pd.DataFrame(index=df.index)
features['GPA_TB']     = df.mean(axis=1)
features['GPA_Std']    = df.std(axis=1).fillna(0)
features['GPA_Min']    = df.min(axis=1)
features['GPA_Max']    = df.max(axis=1)
features['GPA_# NgoThiHien
