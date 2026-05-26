import os
import sys
import io
import glob
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CẤU HÌNH
# ============================================================================

CONFIG = {
    "n_clusters": 3,
    "excel_file": "TỔNG HỢP ĐIỂM K58KTP.xlsx",
    "output_dir": "ket_qua",
    "cluster_labels": ["Trung bình - Yếu", "Khá", "Giỏi"],
}


# ============================================================================
# ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ============================================================================

def find_excel_file():
    """Tìm file Excel trong thư mục."""
    if CONFIG['excel_file']:
        if os.path.exists(CONFIG['excel_file']):
            return CONFIG['excel_file']
        print(f"Khong tim thay: {CONFIG['excel_file']}")
        sys.exit(1)

    xlsx_files = [f for f in glob.glob("*.xlsx") if not f.startswith("ket_qua")]
    if len(xlsx_files) == 0:
        print("Khong tim thay file .xlsx!")
        sys.exit(1)
    elif len(xlsx_files) == 1:
        return xlsx_files[0]
    else:
        print(f"Tim thay {len(xlsx_files)} file Excel:")
        for i, f in enumerate(xlsx_files):
            print(f"  {i+1}. {f}")
        choice = input("Chon file (nhap so): ").strip()
        try:
            return xlsx_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("Lua chon khong hop le!")
            sys.exit(1)


def fix_date_to_score(val):
    """
    Sửa lỗi Excel tự chuyển điểm thành ngày tháng.
    Ví dụ: điểm 2 → 2026-05-02, điểm 3 → 2026-05-03
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            pass
    if hasattr(val, 'day'):
        return float(val.day)
    if isinstance(val, str):
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', val)
        if match:
            return float(int(match.group(3)))
    return np.nan


def load_data(filepath):
    """
    Đọc file Excel - KHÔNG xử lý NaN.
    Môn nào SV chưa học thì để trống, tính toán bình thường (bỏ qua NaN).
    """
    print(f"\n>>> Doc du lieu: {filepath}")

    df_raw = pd.read_excel(filepath, header=None)
    print(f"    Kich thuoc goc: {df_raw.shape[0]} hang x {df_raw.shape[1]} cot")

    # Tìm vị trí header
    mssv_row = None
    name_row = None
    data_start_row = None

    for r in range(min(10, df_raw.shape[0])):
        row_vals = [str(v).strip() for v in df_raw.iloc[r] if pd.notna(v)]
        if 'MSSV' in row_vals:
            mssv_row = r
        if 'Tên Sinh Viên' in row_vals or 'Họ và Tên' in row_vals or 'Tên SV' in row_vals:
            name_row = r
        if 'STT' in row_vals and ('Mã Môn học' in row_vals or any('Tên' in v and 'Môn' in v for v in row_vals)):
            data_start_row = r + 1

    if mssv_row is None: mssv_row = 1
    if name_row is None: name_row = mssv_row + 1
    if data_start_row is None: data_start_row = name_row + 2

    # Tìm cột bắt đầu SV
    sv_start_col = 3
    for c in range(df_raw.shape[1]):
        val = df_raw.iloc[mssv_row, c]
        if pd.notna(val) and str(val).strip().startswith('K'):
            sv_start_col = c
            break

    # Trích xuất MSSV và tên
    mssv_list = []
    name_list = []
    for c in range(sv_start_col, df_raw.shape[1]):
        mssv = df_raw.iloc[mssv_row, c]
        name = df_raw.iloc[name_row, c]
        if pd.notna(mssv) or pd.notna(name):
            mssv_list.append(str(mssv).strip() if pd.notna(mssv) else f"SV_{c}")
            name_list.append(str(name).strip() if pd.notna(name) else f"SV_{c}")

    n_students = len(mssv_list)
    print(f"    So sinh vien: {n_students}")

    # Trích xuất tên môn
    subject_names = []
    subject_codes = []
    data_rows = []

    for r in range(data_start_row, df_raw.shape[0]):
        subj_name = df_raw.iloc[r, 2]
        if pd.notna(subj_name) and str(subj_name).strip() != '':
            subject_names.append(str(subj_name).strip())
            code = df_raw.iloc[r, 1]
            subject_codes.append(str(code).strip() if pd.notna(code) else '')
            data_rows.append(r)

    n_subjects = len(subject_names)
    print(f"    So mon hoc: {n_subjects}")

    # Trích xuất điểm
    scores = np.full((n_students, n_subjects), np.nan)
    for j, r in enumerate(data_rows):
        for i in range(n_students):
            c = sv_start_col + i
            if c < df_raw.shape[1]:
                scores[i, j] = fix_date_to_score(df_raw.iloc[r, c])

    df_students = pd.DataFrame(scores, columns=subject_names, index=mssv_list)

    # Clip điểm hợp lệ về [0, 4], KHÔNG điền NaN
    df_students = df_students.clip(0, 4)

    # Đếm NaN
    nan_count = df_students.isna().sum().sum()
    total = df_students.shape[0] * df_students.shape[1]
    print(f"    Gia tri thieu (NaN): {nan_count}/{total} ({nan_count/total*100:.1f}%)")
    print(f"    -> KHONG xu ly NaN: bo qua mon chua hoc khi tinh toan")

    # Loại bỏ SV không có điểm nào (toàn NaN)
    valid_mask = df_students.notna().any(axis=1)
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        print(f"    -> Loai bo {n_removed} SV khong co diem nao")
        removed_indices = df_students.index[~valid_mask].tolist()
        removed_names = [name_list[mssv_list.index(idx)] for idx in removed_indices]
        df_students = df_students[valid_mask]
        mssv_list = [m for m, v in zip(mssv_list, valid_mask) if v]
        name_list = [n for n, v in zip(name_list, valid_mask) if v]

    print(f"\n    Du lieu sau xu ly: {df_students.shape[0]} SV x {df_students.shape[1]} mon")

    return df_students, mssv_list, name_list, subject_names


# ============================================================================
# PHÂN CỤM
# ============================================================================

def run_clustering(df_students):
    """Chạy K-Means (chính) + Hierarchical (xác nhận)."""
    print(f"\n{'='*60}")
    print(f"    PHAN CUM SINH VIEN")
    print(f"{'='*60}")

    n_clusters = CONFIG['n_clusters']

    # Trích xuất đặc trưng - pandas tự bỏ qua NaN khi tính mean/std/min/max
    features = pd.DataFrame(index=df_students.index)
    features['GPA_TB'] = df_students.mean(axis=1)         # Trung bình (bỏ qua NaN)
    features['GPA_Std'] = df_students.std(axis=1).fillna(0)  # Độ lệch chuẩn
    features['GPA_Min'] = df_students.min(axis=1)          # Điểm thấp nhất
    features['GPA_Max'] = df_students.max(axis=1)          # Điểm cao nhất
    features['GPA_Median'] = df_students.median(axis=1)    # Trung vị
    features['Ty_Le_Gioi'] = (df_students >= 3.2).sum(axis=1) / df_students.notna().sum(axis=1)
    features['Ty_Le_Yeu'] = (df_students < 2.0).sum(axis=1) / df_students.notna().sum(axis=1)

    print(f"    Dac trung: {list(features.columns)}")

    # Chuẩn hóa
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    # --- K-Means ---
    print(f"\n    K-Means (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=30, max_iter=500, random_state=42)
    labels_km = kmeans.fit_predict(X)
    sil_km = silhouette_score(X, labels_km)
    print(f"    Silhouette Score: {sil_km:.4f}")

    # --- Hierarchical ---
    print(f"\n    Hierarchical Clustering (K={n_clusters}, Ward)...")
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_hc = hc.fit_predict(X)
    sil_hc = silhouette_score(X, labels_hc)
    print(f"    Silhouette Score: {sil_hc:.4f}")

    # --- So sánh ---
    ari = adjusted_rand_score(labels_km, labels_hc)
    print(f"\n    So sanh:")
    print(f"    Adjusted Rand Index: {ari:.4f}")

    print(f"\n    -> Su dung ket qua K-Means lam chinh (Silhouette = {sil_km:.4f})")

    return labels_km, features, sil_km, sil_hc, ari


def assign_cluster_labels(df_students, labels):
    """Gán nhãn Giỏi / Khá / TB-Yếu dựa trên GPA trung bình mỗi cụm."""
    n_clusters = CONFIG['n_clusters']
    label_names = CONFIG['cluster_labels']

    gpa_mean = df_students.mean(axis=1).values

    cluster_gpas = {}
    for c in range(n_clusters):
        cluster_gpas[c] = gpa_mean[labels == c].mean()

    sorted_clusters = sorted(cluster_gpas.keys(), key=lambda c: cluster_gpas[c])
    label_map = {}
    for i, c in enumerate(sorted_clusters):
        label_map[c] = label_names[i] if i < len(label_names) else f"Nhom {i+1}"

    named_labels = [label_map[l] for l in labels]

    print(f"\n    Gan nhan:")
    for c in sorted_clusters:
        count = (labels == c).sum()
        print(f"    Cum {c} -> {label_map[c]} | GPA TB: {cluster_gpas[c]:.2f} | SL: {count} SV")

    return named_labels, label_map, cluster_gpas


# ============================================================================
# XUẤT KẾT QUẢ
# ============================================================================

def export_results(df_students, mssv_list, name_list, named_labels):
    """Xuất 3 file .xlsx theo nhóm + 1 biểu đồ tròn."""
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"    XUAT KET QUA")
    print(f"{'='*60}")

    df_full = pd.DataFrame({
        'MSSV': mssv_list,
        'Họ và Tên': name_list,
        'GPA Trung bình': df_students.mean(axis=1).values.round(2),
        'Điểm cao nhất': df_students.max(axis=1).values.round(2),
        'Điểm thấp nhất': df_students.min(axis=1).values.round(2),
        'Số môn đã học': df_students.notna().sum(axis=1).values,
        'Nhóm': named_labels,
    })

    for col in df_students.columns:
        df_full[col] = df_students[col].values

    df_full = df_full.sort_values('GPA Trung bình', ascending=False)

    # Xuất 3 file theo nhóm
    groups = CONFIG['cluster_labels']
    group_counts = {}
    group_gpas = {}

    for group_name in groups:
        df_group = df_full[df_full['Nhóm'] == group_name].copy()
        df_group = df_group.reset_index(drop=True)
        df_group.index = df_group.index + 1
        df_group.index.name = 'STT'

        count = len(df_group)
        group_counts[group_name] = count
        group_gpas[group_name] = df_group['GPA Trung bình'].mean()

        safe_name = group_name.replace(' ', '_').replace('-', '_')
        filename = os.path.join(output_dir, f'nhom_{safe_name}.xlsx')

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df_group.to_excel(writer, sheet_name=group_name)
            workbook = writer.book
            worksheet = writer.sheets[group_name]

            header_fmt = workbook.add_format({
                'bold': True,
                'bg_color': '#4472C4' if 'Giỏi' in group_name else '#ED7D31' if 'Khá' in group_name else '#A5A5A5',
                'font_color': 'white',
                'border': 1, 'text_wrap': True, 'valign': 'vcenter', 'align': 'center',
            })
            for col_num, col_name in enumerate(['STT'] + list(df_group.columns)):
                worksheet.write(0, col_num, col_name, header_fmt)
            worksheet.set_column(0, 0, 5)
            worksheet.set_column(1, 1, 16)
            worksheet.set_column(2, 2, 25)
            worksheet.set_column(3, 7, 14)

        print(f"    -> {filename} ({count} sinh vien)")

    # Biểu đồ tròn
    print(f"\n    Tao bieu do phan cum...")

    vn_fonts = ['Arial', 'Segoe UI', 'Tahoma', 'Times New Roman', 'Calibri']
    chosen_font = 'DejaVu Sans'
    for font_name in vn_fonts:
        matches = [f for f in fm.findSystemFonts() if font_name.lower().replace(' ', '') in f.lower().replace(' ', '')]
        if matches:
            chosen_font = font_name
            break

    plt.rcParams['font.family'] = chosen_font
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 8))

    labels_list = list(group_counts.keys())
    sizes = list(group_counts.values())
    total = sum(sizes)

    colors = ['#FF6B6B', '#FFD93D', '#6BCB77']
    explode = [0.03, 0.03, 0.06]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct/100.*total))} SV)',
        startangle=90, explode=explode, shadow=True,
        textprops={'fontsize': 13, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        pctdistance=0.6,
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)

    legend_labels = [f'{name}  ({count} SV - {count/total*100:.1f}%)'
                     for name, count in zip(labels_list, sizes)]
    ax.legend(wedges, legend_labels, title="Phân loại",
              loc="center left", bbox_to_anchor=(0.85, 0, 0.5, 1),
              fontsize=11, title_fontsize=12)

    ax.set_title('BIỂU ĐỒ PHÂN CỤM SINH VIÊN\nTheo Năng Lực Học Tập (K-Means, K=3)',
                 fontsize=16, fontweight='bold', pad=20)

    fig.text(0.5, 0.02, f'Tổng số: {total} sinh viên', ha='center', fontsize=11, style='italic', color='gray')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'bieu_do_phan_cum.png')
    fig.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"    -> {chart_path}")
    plt.show()
    plt.close()

    return group_counts, group_gpas


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  PHAN CUM SINH VIEN THEO NANG LUC HOC TAP")
    print("  K-Means + Hierarchical Clustering")
    print("=" * 60)

    filepath = find_excel_file()
    df_students, mssv_list, name_list, subject_names = load_data(filepath)

    labels, features, sil_km, sil_hc, ari = run_clustering(df_students)

    named_labels, label_map, cluster_gpas = assign_cluster_labels(df_students, labels)

    group_counts, group_gpas = export_results(df_students, mssv_list, name_list, named_labels)

    # Tổng kết
    total = sum(group_counts.values())
    print(f"\n{'='*60}")
    print(f"    TONG KET")
    print(f"{'='*60}")
    print(f"    Tong sinh vien: {total}")
    for name, count in group_counts.items():
        gpa = group_gpas[name]
        print(f"    {name:20s}: {count:3d} SV ({count/total*100:5.1f}%) | GPA TB: {gpa:.2f}")
    print(f"\n    Silhouette Score: {sil_km:.4f}")
    print(f"\n    Hoan tat! Ket qua trong: {os.path.abspath(CONFIG['output_dir'])}/")


if __name__ == "__main__":
    main()
