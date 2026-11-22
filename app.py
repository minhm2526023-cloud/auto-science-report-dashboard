import os
import platform
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fpdf import FPDF

# ==============================================================================
# 1. KHỞI TẠO DATASET MẶC ĐỊNH (MÔI TRƯỜNG & NÔNG NGHIỆP ĐBSCL)
# ==============================================================================
DEFAULT_CSV = "dataset_moi_truong_lua.csv"

def get_default_dataset():
    if os.path.exists(DEFAULT_CSV):
        return pd.read_csv(DEFAULT_CSV)
    
    np.random.seed(42)
    n_samples = 300
    df = pd.DataFrame({
        "Nhiet_do_C": np.round(np.random.normal(31.5, 2.2, n_samples), 1),
        "Do_am_Phan_tram": np.round(np.random.normal(82.0, 6.5, n_samples), 1),
        "Do_pH_Dat": np.round(np.random.normal(5.8, 0.5, n_samples), 2),
        "Luong_Phan_Dam_kg": np.round(np.random.normal(110, 20, n_samples), 1),
        "Chi_so_Diep_luc_SPAD": np.round(np.random.normal(38.0, 4.5, n_samples), 1)
    })
    risk_score = (df["Do_am_Phan_tram"] > 83).astype(int) + (df["Luong_Phan_Dam_kg"] > 115).astype(int)
    df["Nguy_co_Benh"] = (risk_score >= 1).astype(int)
    df.to_csv(DEFAULT_CSV, index=False)
    return df

df_default = get_default_dataset()

# ==============================================================================
# 2. CẤU HÌNH FONT TIẾNG VIỆT ĐA NỀN TẢNG (WINDOWS / MACOS / LINUX & COLAB)
# ==============================================================================
def register_unicode_fonts(pdf: FPDF):
    system_os = platform.system()
    if system_os == "Windows":
        font_reg = r"C:\Windows\Fonts\arial.ttf"
        font_bold = r"C:\Windows\Fonts\arialbd.ttf"
    elif system_os == "Darwin":  # macOS
        font_reg = "/System/Library/Fonts/Supplemental/Arial.ttf"
        font_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    else:  # Linux / Google Colab
        font_reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    pdf.add_font("CustomFont", "", font_reg)
    pdf.add_font("CustomFont", "B", font_bold)

# ==============================================================================
# 3. HÀM PHÂN TÍCH DỮ LIỆU, VẼ BIỂU ĐỒ VÀ SINH BÁO CÁO PDF 5 MỤC
# ==============================================================================
def analyze_and_generate_pdf(file_obj, x_col, y_col, target_col):
    # Đọc dữ liệu đầu vào
    if file_obj is not None:
        df = pd.read_csv(file_obj.name)
    else:
        df = df_default.copy()

    # Làm sạch dữ liệu (loại bỏ dòng thiếu giá trị)
    df_clean = df.dropna().copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        raise ValueError("Dataset không chứa cột dữ liệu số để phân tích.")

    if x_col not in num_cols:
        x_col = num_cols[0]
    if y_col not in num_cols:
        y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    if target_col not in df_clean.columns:
        target_col = df_clean.columns[-1]

    # Tính toán thống kê mô tả & tương quan
    desc = df_clean[num_cols].describe().round(2)
    corr_val = df_clean[x_col].corr(df_clean[y_col])

    # Huấn luyện mô hình phân loại Random Forest đơn giản
    model_summary = ""
    try:
        features = [c for c in num_cols if c != target_col]
        X = df_clean[features]
        y = df_clean[target_col].astype(int)
        if len(y.unique()) >= 2 and len(features) >= 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test)) * 100
            model_summary = f"Mô hình phân loại Random Forest dự đoán '{target_col}' đạt độ chính xác: {acc:.2f}%."
        else:
            model_summary = f"Hệ số tương quan Pearson giữa {x_col} và {y_col} đạt r = {corr_val:.3f}."
    except Exception:
        model_summary = f"Hệ số tương quan tuyến tính giữa {x_col} và {y_col} là r = {corr_val:.3f}."

    # Trực quan hóa dữ liệu
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    sns.histplot(df_clean[x_col], kde=True, color="#0284C7", ax=axes[0])
    axes[0].set_title(f"Phân phối biến {x_col}", fontweight="bold", color="#003366")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    sns.scatterplot(data=df_clean, x=x_col, y=y_col, hue=target_col, palette="viridis", ax=axes[1])
    axes[1].set_title(f"Tương quan {x_col} vs {y_col}", fontweight="bold", color="#003366")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = "dashboard_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Khởi tạo và xuất báo cáo PDF 5 mục
    pdf = FPDF()
    pdf.add_page()
    register_unicode_fonts(pdf)

    # Tiêu đề báo cáo
    pdf.set_font("CustomFont", "B", 15)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 8, "BÁO CÁO KHOA HỌC PHÂN TÍCH DỮ LIỆU TỰ ĐỘNG\nMINI-PROJECT: HỆ THỐNG QUẢN LÝ NỘI DUNG SỐ", align="C")
    pdf.ln(3)

    # 1. Giới thiệu đề tài
    pdf.set_font("CustomFont", "B", 11)
    pdf.cell(0, 7, "1. Giới thiệu đề tài", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, "Nghiên cứu xây dựng quy trình tự động hóa phân tích dữ liệu môi trường và nông nghiệp tại Đồng bằng Sông Cửu Long, hỗ trợ đánh giá các yếu tố ảnh hưởng đến nguy cơ phát sinh bệnh trên cây lúa.")
    pdf.ln(2)

    # 2. Mô tả dữ liệu
    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "2. Mô tả dữ liệu", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, f"Bộ dữ liệu sau khi làm sạch gồm {df_clean.shape[0]} mẫu quan sát và {df_clean.shape[1]} thuộc tính. Các biến được chọn phân tích chính gồm: {x_col}, {y_col} và biến mục tiêu {target_col}.")
    pdf.ln(2)

    # 3. Kết quả phân tích
    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "3. Kết quả phân tích thống kê & Mô hình", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    stats_text = (
        f"- Giá trị trung bình của {x_col}: {df_clean[x_col].mean():.2f} (Độ lệch chuẩn: {df_clean[x_col].std():.2f}).\n"
        f"- Giá trị trung bình của {y_col}: {df_clean[y_col].mean():.2f} (Độ lệch chuẩn: {df_clean[y_col].std():.2f}).\n"
        f"- Đánh giá mô hình: {model_summary}"
    )
    pdf
