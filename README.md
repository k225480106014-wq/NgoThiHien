# 📊 ỨNG DỤNG MACHINE LEARNING PHÂN CỤM NĂNG LỰC HỌC TẬP SINH VIÊN K58KTP

Dự án nghiên cứu này xây dựng một hệ thống phân loại học lực sinh viên dựa trên phương pháp **Học máy không giám sát (Unsupervised Learning)**. Bằng cách áp dụng các thuật toán phân cụm tiên tiến, hệ thống tự động nhận diện cấu trúc phân bố năng lực học tập tự nhiên từ dữ liệu điểm số, khắc phục hoàn toàn tính chủ quan của các phương pháp áp ngưỡng cố định truyền thống.

---
## 🚀 1. Hướng Dẫn Vận Hành Nhanh (Quick Start)

### Yêu cầu môi trường
* Hệ điều hành tích hợp: Ngôn ngữ nền tảng **Python 3.8 trở lên**.
* Cài đặt toàn bộ các thư viện phụ thuộc bằng một câu lệnh duy nhất tại Terminal:
```bash
pip install pandas numpy scikit-learn matplotlib openpyxl xlsxwriter
```
---

Quy trình triển khai

1 Di chuyển file dữ liệu bảng điểm gốc ⁠TỔNG HỢP ĐIỂM K58KTP.xlsx⁠ vào chung thư mục dự án.

2 Kích hoạt bộ xử lý mô hình thông qua câu lệnh:
```
python phan_cum_sinh_vien.py
```
1 Tiến trình hoàn tất, hệ thống tự động khởi tạo thư mục ⁠/ket_qua⁠ chứa các file báo cáo Excel chi tiết và sơ đồ phân bố trực quan.

---
🎯 2. Ý Tưởng Đề Tài & Mục Tiêu Nghiên Cứu
---
Hệ thống được phát triển nhằm thay thế phương pháp gán nhãn điểm cứng truyền thống (như xét duyệt thủ công GPA \ge 3.2 là Giỏi, GPA \ge 2.5 là Khá) vốn mang nặng 
tính chủ quan. Dự án hướng tới việc cho phép thuật toán tự động phân tích xu hướng điểm số thực tế để gom nhóm 71 sinh viên thành 3 vùng năng lực tự nhiên.

Ý nghĩa thực tiễn mang lại:

 Khai phá bức tranh phân hóa học lực khách quan của tập thể lớp K58KTP mà không bị ảnh hưởng bởi độ khó/dễ của đề thi qua từng học kỳ.

 Tách biệt danh sách sinh viên xuất sắc (phục vụ khen thưởng) hoặc nhóm cần lưu ý học vụ (phục vụ cảnh báo) chỉ trong vòng 1 giây mà không cần lọc tay thủ công.

---
🛠️ 3. Pipeline Tiền Xử Lý Dữ Liệu (Data Preprocessing)
---
Hệ thống thực hiện bóc tách bảng điểm gồm 52 môn học thông qua một đường ống xử lý khép kín:

A. Làm sạch dữ liệu thông minh (Data Cleaning)
---
 Khử lỗi định dạng Excel:
 
 Hàm ⁠fix_date_to_score()⁠ tự động rà soát và ép kiểu các ô điểm chẵn bị cấu hình Auto-Format của Excel nhận nhầm thành dữ liệu ngày tháng (ví dụ: chuyển đổi ngày ⁠2026-05-02⁠ quay ngược về giá trị điểm số thực ⁠2.0⁠).

 Bảo toàn dữ liệu trống (NaN): Đối với các ô điểm khuyết thiếu (sinh viên chưa học hoặc hoãn thi), dự án chọn giải pháp giữ nguyên trạng thái NaN thay vì điền bừa số 0. Các hàm thống kê của Pandas được thiết lập để tự động bỏ qua NaN, giúp phản ánh đúng năng lực thực tế dựa trên các môn đã hoàn thành.

 Bộ lọc bản ghi nhiễu: Hệ thống tự động loại bỏ 16 hồ sơ trống (chưa nhập bất kỳ điểm số nào), giữ lại tập dữ liệu chuẩn gồm 55 sinh viên hợp lệ để đưa vào huấn luyện.

B. Trích xuất đặc trưng (Feature Engineering)
---
Để loại bỏ "lời nguyền đa chiều" khi dùng trực tiếp điểm của 52 môn học đơn lẻ, hệ thống cô đọng dữ liệu thành 7 chỉ số đại diện:

 ⁠GPA_TB⁠: Điểm trung bình tích lũy hệ 4.
 
 ⁠GPA_Std⁠: Độ lệch chuẩn (đo lường độ ổn định và giữ vững phong độ điểm số).

 ⁠GPA_Min⁠ / ⁠GPA_Max⁠ / ⁠GPA_Median⁠: Không gian phân vùng điểm đáy, điểm đỉnh và giá trị trung vị học lực.
 
 ⁠Ty_Le_Gioi⁠: Tỷ lệ phần trăm số môn cán mốc điểm giỏi (\ge 3.2).

 Ty_Le_Yeu⁠: Tỷ lệ phần trăm số học phần rơi vào nhóm điểm yếu (< 2.0).

 💡 Note: Toàn bộ không gian đặc trưng sau khi trích xuất được chuẩn hóa đồng bộ bằng bộ công cụ ⁠StandardScaler⁠ (\mu = 0, \sigma = 1) trước khi đưa vào tính toán khoảng cách Euclidean.

---
 ⚙️ 4. Giải Thuật Học Máy Sử Dụng
---

Dự án áp dụng chiến lược phân cụm đa mô hình nhằm kiểm định và đối chiếu chéo kết quả:

1 Thuật toán cốt lõi (K-Means): Thiết lập vùng phân tách cố định K=3, cấu hình tham số chống bẫy cực trị địa phương ⁠n_init=30⁠ cùng giới hạn vòng lặp hội tụ ⁠max_iter=500⁠.

2 Thuật toán đối chứng (Hierarchical Clustering): Áp dụng tiêu chí liên kết Ward Linkage (tối thiểu hóa tổng phương sai trong các cụm được gộp) để đánh giá độ tin cậy và kiểm tra tính giao thoa của các phổ điểm ngoài thực tế.

---
📊 5. Kết Quả Thực Nghiệm & Đo Lường Chất Lượng
---
Hệ thống phân tích thành công tập mẫu chuẩn gồm 55 sinh viên và đưa ra các kết luận định danh:
 
 🟢 Cụm Học Lực Giỏi (19 SV - Chiếm 34.5%): Phong độ học tập dẫn đầu, điểm số có tính ổn định cực cao, điểm trung bình nhóm đạt GPA_{TB} = 3.15.

 🟡 Cụm Học Lực Khá (25 SV - Chiếm 45.5%): Lực lượng nòng cốt giữ thế cân bằng cho tập thể, học lực đại trà đồng đều, đạt GPA_{TB} = 2.58.
 
 🔴 Cụm Cần Hỗ Trợ (Trung bình/Yếu - 11 SV - Chiếm 20.0%): Biên độ dao động điểm số lớn, xuất hiện nhiều học phần dưới ngưỡng an toàn, đạt GPA_{TB} = 2.20.

Chỉ số kiểm định toán học:

 . Silhouette Score (K-Means): Đạt ngưỡng ⁠$0.3222$⁠ (Mức độ phân tách ranh giới giữa các cụm ở mức ổn định).

 . Silhouette Score (Hierarchical): Đạt ngưỡng ⁠$0.4166$⁠ (Cơ chế phân cấp Ward Linkage cho độ tách biệt không gian rất nét).

 . Adjusted Rand Index (ARI): Đạt ngưỡng ⁠$0.3655$⁠ (Độ tương đồng toán học phản ánh đúng tính giao thoa tự nhiên của điểm số thực tế).

<img width="1192" height="957" alt="image" src="https://github.com/user-attachments/assets/ec17c723-8bdc-4887-8fd6-6818e65b477f" />

 ---
 💻 6. Đoạn Mã Nguồn Cốt Lõi
---

Khởi tạo & Trích xuất Vector đặc trưng
---
```
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
features['GPA_Median'] = df.median(axis=1)
features['Ty_Le_Gioi'] = (df >= 3.2).sum(axis=1) / df.notna().sum(axis=1)
features['Ty_Le_Yeu']  = (df < 2.0).sum(axis=1) / df.notna().sum(axis=1)
```
---
Thực thi cấu trúc mô hình phân cụm song song
---
```
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Khớp không gian chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(features.values)

# Khởi chạy Mô hình A: K-Means Clustering
kmeans = KMeans(n_clusters=3, n_init=30, max_iter=500, random_state=42)
labels_km = kmeans.fit_predict(X)

# Khởi chạy Mô hình B: Hierarchical Clustering (Ward)
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X)
