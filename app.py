%%writefile app.py
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

def register_unicode_fonts(pdf: FPDF):
    system_os = platform.system()
    if system_os == "Windows":
        font_reg, font_bold = r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"
    elif system_os == "Darwin":
        font_reg, font_bold = "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    else:
        font_reg, font_bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdf.add_font("CustomFont", "", font_reg)
    pdf.add_font("CustomFont", "B", font_bold)

def analyze_and_generate_pdf(file_obj, x_col, y_col, target_col):
    df = pd.read_csv(file_obj.name) if file_obj is not None else df_default.copy()
    df_clean = df.dropna().copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if x_col not in num_cols: x_col = num_cols[0]
    if y_col not in num_cols: y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    if target_col not in df_clean.columns: target_col = df_clean.columns[-1]

    desc = df_clean[num_cols].describe().round(2)
    corr_val = df_clean[x_col].corr(df_clean[y_col])

    try:
        features = [c for c in num_cols if c != target_col]
        X, y = df_clean[features], df_clean[target_col].astype(int)
        if len(y.unique()) >= 2 and len(features) >= 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test)) * 100
            model_summary = f"Mô hình phân loại Random Forest dự đoán '{target_col}' đạt độ chính xác: {acc:.2f}%."
        else:
            model_summary = f"Hệ số tương quan Pearson giữa {x_col} và {y_col} đạt r = {corr_val:.3f}."
    except Exception:
        model_summary = f"Hệ số tương quan tuyến tính giữa {x_col} và {y_col} là r = {corr_val:.3f}."

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

    pdf = FPDF()
    pdf.add_page()
    register_unicode_fonts(pdf)

    pdf.set_font("CustomFont", "B", 15)
    pdf.set_text_color(0, 51, 102)
    pdf.multi_cell(0, 8, "BÁO CÁO KHOA HỌC PHÂN TÍCH DỮ LIỆU TỰ ĐỘNG\nMINI-PROJECT: HỆ THỐNG QUẢN LÝ NỘI DUNG SỐ", align="C")
    pdf.ln(3)

    pdf.set_font("CustomFont", "B", 11)
    pdf.cell(0, 7, "1. Giới thiệu đề tài", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, "Nghiên cứu xây dựng quy trình tự động hóa phân tích dữ liệu môi trường và nông nghiệp tại Đồng bằng Sông Cửu Long, hỗ trợ đánh giá các yếu tố ảnh hưởng đến nguy cơ phát sinh bệnh trên cây lúa.")
    pdf.ln(2)

    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "2. Mô tả dữ liệu", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, f"Bộ dữ liệu sau khi làm sạch gồm {df_clean.shape[0]} mẫu quan sát và {df_clean.shape[1]} thuộc tính. Các biến được chọn phân tích chính gồm: {x_col}, {y_col} và biến mục tiêu {target_col}.")
    pdf.ln(2)

    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "3. Kết quả phân tích thống kê & Mô hình", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, f"- Trung bình {x_col}: {df_clean[x_col].mean():.2f} (Độ lệch chuẩn: {df_clean[x_col].std():.2f}).\n- Trung bình {y_col}: {df_clean[y_col].mean():.2f} (Độ lệch chuẩn: {df_clean[y_col].std():.2f}).\n- Đánh giá mô hình: {model_summary}")
    pdf.ln(2)

    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 7, "4. Biểu đồ minh họa trực quan", ln=True)
    pdf.image(chart_path, x=15, w=180)
    pdf.ln(2)

    pdf.set_font("CustomFont", "B", 11)
    pdf.set_text_color(27, 94, 32)
    pdf.cell(0, 7, "5. Kết luận & Đề xuất", ln=True)
    pdf.set_font("CustomFont", "", 10)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 5.5, "Hệ thống Dashboard tự động giúp rút ngắn thời gian xử lý số liệu thực nghiệm và xuất báo cáo chuẩn hóa. Đề xuất mở rộng tích hợp cảm biến IoT thời gian thực để cảnh báo sớm dịch bệnh cho nông dân.")

    pdf_path = "Bao_cao_khoa_hoc_tu_dong.pdf"
    pdf.output(pdf_path)
    return chart_path, desc.reset_index(), model_summary, pdf_path

def update_dropdowns(file_obj):
    cols = df_default.columns.tolist() if file_obj is None else pd.read_csv(file_obj.name).columns.tolist()
    return (
        gr.Dropdown(choices=cols, value=cols[0]),
        gr.Dropdown(choices=cols, value=cols[1] if len(cols) > 1 else cols[0]),
        gr.Dropdown(choices=cols, value=cols[-1])
    )

cols_list = df_default.columns.tolist()

with gr.Blocks(title="Dashboard Phân tích & Sinh Báo cáo Khoa học") as demo:
    gr.Markdown("# 📊 MINI-PROJECT: DASHBOARD PHÂN TÍCH DỮ LIỆU & SINH BÁO CÁO PDF TỰ ĐỘNG")
    gr.Markdown("Chọn biến phân tích từ bộ dữ liệu mặc định (hoặc tải lên tệp `.csv` của bạn) để xem trực quan hóa và tải báo cáo khoa học.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Tải lên Dataset (.csv) - Tùy chọn", file_types=[".csv"])
            x_dropdown = gr.Dropdown(choices=cols_list, value=cols_list[1], label="Chọn biến trục X (Phân phối)")
            y_dropdown = gr.Dropdown(choices=cols_list, value=cols_list[3], label="Chọn biến trục Y (Tương quan)")
            target_dropdown = gr.Dropdown(choices=cols_list, value=cols_list[-1], label="Chọn biến mục tiêu (Phân loại)")
            run_btn = gr.Button("🚀 Phân tích & Xuất Báo cáo PDF", variant="primary")

        with gr.Column(scale=2):
            plot_out = gr.Image(label="Biểu đồ trực quan hóa dữ liệu")
            model_out = gr.Textbox(label="Kết quả mô hình & Tương quan")
            pdf_out = gr.File(label="📥 Tải xuống Báo cáo Khoa học (.pdf)")

    table_out = gr.Dataframe(label="Bảng thống kê mô tả dữ liệu (Pandas Describe)")

    file_input.change(fn=update_dropdowns, inputs=[file_input], outputs=[x_dropdown, y_dropdown, target_dropdown])
    run_btn.click(fn=analyze_and_generate_pdf, inputs=[file_input, x_dropdown, y_dropdown, target_dropdown], outputs=[plot_out, table_out, model_out, pdf_out])

if __name__ == "__main__":
    demo.launch(share=True)
