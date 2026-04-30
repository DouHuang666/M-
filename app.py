import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import requests
import json
import hashlib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

from forecast_models import select_best_model, forecast_future, weighted_moving_average, double_moving_average
from inventory_optim import optimize_RQ, optimize_tS
from chat_ai import chat_response, compute_inventory_for_sku, check_alerts, get_slow_moving_skus, generate_health_report

# ==================== 页面配置 ====================
st.set_page_config(page_title="智能库存决策系统", layout="wide", page_icon="📊")

# 添加滚动位置保存与恢复的 JS
st.markdown("""
<script>
// 保存滚动位置
window.addEventListener('beforeunload', function() {
    sessionStorage.setItem('scrollPos', window.scrollY);
});
// 恢复滚动位置
window.addEventListener('load', function() {
    var scrollPos = sessionStorage.getItem('scrollPos');
    if (scrollPos !== null) {
        window.scrollTo(0, parseInt(scrollPos));
        sessionStorage.removeItem('scrollPos');
    }
});
</script>
""", unsafe_allow_html=True)

# ==================== 全局样式 ====================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        width: 220px !important;
        min-width: 0 !important;
        max-width: 280px !important;
        transition: width 0.2s ease;
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        visibility: hidden;
    }
    .stApp { background-color: #F5F7FA; }
    .main-content { margin-top: 1rem; padding: 0 2rem 2rem 2rem; flex: 1; }
    .stApp > header { width: 100%; }
    section.main > div { width: 100%; }
    [data-testid="stSidebar"] * { color: #1F2937 !important; }
    [data-testid="stSidebarNav"] li a {
        font-size: 20px !important;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 6px;
    }
    [data-testid="stSidebarNav"] li a:hover { background-color: #F9FAFB; color: #2A6DFF !important; }
    [data-testid="stSidebarNav"] li a[aria-current="page"] { background-color: #E9F0FF; color: #2A6DFF !important; font-weight: 600; }
    
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 18px;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E5E7EB;
    }
    .card-title i {
        margin-right: 10px;
        color: #2A6DFF;
    }
    
    /* Dashboard 快速开始卡片专用样式 ———— 统一高度 */
    .quick-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.2s ease;
        border: 1px solid #E5E7EB;
        height: 170px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .quick-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.02);
        border-color: #2A6DFF;
    }
    .quick-icon {
        font-size: 2.5rem;
        color: #2A6DFF;
        margin-bottom: 0.75rem;
    }
    .quick-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #1F2937;
    }
    .quick-desc {
        font-size: 0.85rem;
        color: #6B7280;
        line-height: 1.4;
    }
    
    /* 紧凑卡片样式（与库存概览“整体库存指标”卡片高度一致） */
    .compact-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        padding: 12px !important;
        margin-bottom: 12px !important;
    }
    .compact-card .card-title {
        font-size: 16px;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E5E7EB;
    }
    .compact-card .card-title i {
        margin-right: 8px;
        color: #2A6DFF;
    }
    
    /* 企业级引导卡片 */
    .guide-card {
        background: linear-gradient(135deg, #F8FAFF 0%, #FFFFFF 100%);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .guide-title {
        font-size: 1rem;
        font-weight: 500;
        color: #374151;
        margin-bottom: 0.25rem;
    }
    .guide-desc {
        font-size: 1rem;
        color: #6B7280;
    }
    
    /* 系统封面大标题 */
    .cover-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .cover-subtitle {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* ========== 数据上传折叠窗极致紧凑化 ========== */
    .compact-expander {
        padding: 0 !important;
    }
    .compact-expander .streamlit-expanderContent {
        padding: 0 !important;
    }
    .compact-expander .stMarkdown h4 {
        margin: 0 0 0.1rem 0 !important;
        font-size: 1rem !important;
    }
    .compact-expander .stMarkdown hr {
        margin: 0.1rem 0 !important;
    }
    .compact-expander .stFileUploader {
        margin-bottom: 0.1rem !important;
    }
    .compact-expander .stDownloadButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    .compact-expander .stColumns {
        gap: 0.2rem !important;
        margin-bottom: 0 !important;
    }
    .compact-expander .stMarkdown p {
        margin-bottom: 0.1rem !important;
    }
    /* 缩小上传区域内部拖拽框的上下边距 */
    .compact-expander div[data-testid="stFileUploadDropzone"] {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        min-height: 60px !important;
    }
    .compact-expander .stFileUploader > div:first-child {
        margin-bottom: 0 !important;
    }
    /* 压缩按钮间的空隙 */
    .compact-expander .stButton button {
        margin: 0 !important;
        padding: 0.2rem 0.5rem !important;
    }
    
    [data-testid="stSidebar"] h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stCheckbox {
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stSlider {
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        margin-bottom: 0.25rem !important;
    }
    
    .custom-table-container {
        overflow-x: auto;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background-color: #FFFFFF;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        background-color: #FFFFFF;
    }
    .custom-table th {
        background-color: #1E3A8A;
        color: white !important;
        padding: 12px 16px;
        text-align: center;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #2C3E50;
        white-space: nowrap;
    }
    .custom-table td {
        padding: 12px 16px;
        text-align: center;
        vertical-align: middle;
        border-bottom: 1px solid #F3F4F6;
        background-color: #FFFFFF;
        color: #1F2937;
        white-space: nowrap;
    }
    .custom-table tbody tr:nth-child(even) td {
        background-color: #F5F8FF;
    }
    .custom-table tbody tr:hover td {
        background-color: #eef4fa;
    }
    .custom-table-container::-webkit-scrollbar {
        height: 6px;
        width: 6px;
    }
    .custom-table-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .custom-table-container::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    .custom-table-container::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    .class-label {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        text-align: center;
    }
    .class-AX { background-color: #DBEAFE; color: #1E40AF; }
    .class-AY { background-color: #CCFBF1; color: #0F766E; }
    .class-BX { background-color: #D1FAE5; color: #047857; }
    .class-BY { background-color: #D1FAE5; color: #047857; }
    .class-CX { background-color: #F3E8FF; color: #6B21A8; }
    .class-CY { background-color: #FEF3C7; color: #B45309; }
    .class-CZ { background-color: #F3E8FF; color: #6B21A8; }
    
    .status-normal { background-color: #ECFDF5; color: #10B981; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; display: inline-block; }
    .status-warning { background-color: #FEF3C7; color: #F59E0B; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; display: inline-block; }
    .status-emergency { background-color: #FEF2F2; color: #EF4444; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; display: inline-block; }
    
    .highlight-number { font-weight: bold; color: #2A6DFF; }
    .warning-days { color: #EF4444 !important; font-weight: bold; background-color: #FEF2F2; border-radius: 4px; padding: 2px 6px; display: inline-block; }
    
    .custom-info { background-color: #EFF6FF; border-left: 4px solid #3B82F6; border-radius: 8px; padding: 12px 16px; margin: 1rem 0; color: #1F2937; }
    .custom-warning { background-color: #FEF3C7; border-left: 4px solid #F59E0B; border-radius: 8px; padding: 12px 16px; margin: 1rem 0; color: #1F2937; }
    .custom-success { background-color: #ECFDF5; border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 16px; margin: 1rem 0; color: #1F2937; }
    .fa, .fas, .far { margin-right: 0.3rem; }
    .metric-card { background-color: #F9FAFB; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #E5E7EB; }
    .metric-value { font-size: 28px; font-weight: bold; color: #1F2937; }
    .metric-label { font-size: 14px; color: #6B7280; margin-top: 8px; }
    .chart-title { font-size: 14px !important; font-weight: 600 !important; color: #1F2937 !important; margin-bottom: 8px !important; }
    .param-card { background-color: #FFFFFF; border-radius: 8px; padding: 12px; margin: 4px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #E5E7EB; }
    .param-name { font-size: 12px; color: #6B7280; margin-bottom: 4px; }
    .param-value { font-size: 16px; font-weight: bold; color: #1F2937; }
    .status-card { background-color: #F9FAFB; border-radius: 8px; padding: 12px; margin: 4px; display: flex; align-items: center; gap: 8px; border: 1px solid #E5E7EB; flex: 1; min-height: 130px; }
    .status-card-warning { background-color: #FEF2F2; border-left: 4px solid #EF4444; }
    .status-card-success { background-color: #ECFDF5; border-left: 4px solid #10B981; }
    .status-card-caution { background-color: #FEF3C7; border-left: 4px solid #F59E0B; }
    .status-card-info { background-color: #EFF6FF; border-left: 4px solid #3B82F6; }
    .status-icon { font-size: 20px; }
    .status-text { font-size: 14px; font-weight: 500; color: #1F2937; }
    .status-detail { font-size: 12px; color: #6B7280; }
    .sku-title { font-size: 20px; font-weight: bold; color: #1F2937; margin-bottom: 20px; }
    
    /* 低库存预警标题与卡片标题统一字号 */
    .low-stock-title {
        font-size: 16px;
        font-weight: bold;
        margin: 0 0 6px 0;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

# ==================== 表格渲染函数（完全保持原始版本） ====================
def render_dataframe_as_custom_table(df, add_index=True, special_columns=None, format_dict=None):
    if df.empty:
        return "<div>暂无数据</div>"
    df_display = df.copy()
    if format_dict:
        for col, fmt in format_dict.items():
            if col in df_display.columns:
                try:
                    df_display[col] = df_display[col].apply(lambda x: fmt.format(float(x)) if pd.notna(x) else x)
                except:
                    pass
    headers = []
    if add_index:
        headers.append("序号")
    headers.extend(df_display.columns.tolist())
    rows_html = []
    for i, (_, row) in enumerate(df_display.iterrows(), start=1):
        cells = []
        if add_index:
            cells.append(f'<td style="text-align: center;">{i}</td>')
        for col in df_display.columns:
            val = row[col]
            cell_content = str(val) if pd.notna(val) else ""
            if special_columns and col in special_columns:
                stype = special_columns[col]
                if stype == 'capsule_class':
                    cls = f"class-{val}" if val in ['AX','AY','BX','BY','CX','CY','CZ'] else ""
                    cell_content = f'<span class="class-label {cls}">{val}</span>' if cls else val
                elif stype == 'urgency':
                    if val == "正常":
                        cell_content = '<span class="status-normal">正常</span>'
                    elif val == "注意":
                        cell_content = '<span class="status-warning">注意</span>'
                    elif val == "紧急":
                        cell_content = '<span class="status-emergency">紧急</span>'
                elif stype == 'highlight_number':
                    try:
                        num = float(val)
                        cell_content = f'<span class="highlight-number">{num:.2f}</span>' if isinstance(val, (int, float)) else val
                    except:
                        pass
                elif stype == 'days_warning':
                    try:
                        days = float(val)
                        if days < 7:
                            cell_content = f'<span class="warning-days">{days:.0f}</span>'
                        else:
                            cell_content = f'{days:.0f}'
                    except:
                        cell_content = val
            elif col == "ABC-XYZ分类" and val in ['AX','AY','BX','BY','CX','CY','CZ']:
                cell_content = f'<span class="class-label class-{val}">{val}</span>'
            cells.append(f'<td style="text-align: center;">{cell_content}</td>')
        rows_html.append(f'<tr>{"".join(cells)}</tr>')
    table_html = f"""
    <div class="custom-table-container">
        <table class="custom-table">
            <thead><tr>{"".join(f'<th>{h}</th>' for h in headers)}</thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
    </div>
    """
    return table_html

# ==================== 辅助函数 ====================
def get_in_out_fee(weight_kg):
    if weight_kg <= 2:
        in_fee = 0.15
    elif weight_kg <= 5:
        in_fee = 0.24
    elif weight_kg <= 10:
        in_fee = 0.30
    elif weight_kg <= 15:
        in_fee = 0.36
    elif weight_kg <= 20:
        in_fee = 0.48
    else:
        in_fee = 0.60
    lb = weight_kg * 2.20462
    if lb <= 1:
        out_fee = 0.80
    elif lb <= 2.2:
        out_fee = 1.20
    elif lb <= 4.4:
        out_fee = 1.50
    elif lb <= 11:
        out_fee = 2.00
    elif lb <= 23:
        out_fee = 3.00
    elif lb <= 44:
        out_fee = 5.00
    elif lb <= 66:
        out_fee = 7.00
    elif lb <= 88:
        out_fee = 10.00
    elif lb <= 150:
        out_fee = 15.00
    elif lb <= 194:
        out_fee = 20.00
    else:
        out_fee = 25.00
    return in_fee, out_fee

def compute_abc_xyz_classification(inventory_df, history_df, sku_params, a_ratio=0.7, b_ratio=0.25):
    try:
        years = [idx.split('-')[0] for idx in history_df.index if isinstance(idx, str)]
        if '2025' in years:
            history_2025 = history_df[history_df.index.str.startswith('2025-')]
        else:
            history_2025 = history_df
    except:
        history_2025 = history_df
    results = []
    for _, row in inventory_df.iterrows():
        sku = row['SKU']
        name = row['品名']
        stock = row['库存']
        unit_price = row['单价(美元)']
        if sku in history_2025.columns:
            sales = history_2025[sku].dropna().tolist()
        else:
            sales = []
        annual_quantity = sum(sales)
        annual_value = annual_quantity * unit_price
        if len(sales) > 0:
            mu = np.mean(sales)
            sigma = np.std(sales, ddof=1) if len(sales) > 1 else 0
            cv = sigma / mu if mu != 0 else 0
        else:
            cv = 0
        if cv < 0.25:
            xyz = 'X'
        elif cv < 0.5:
            xyz = 'Y'
        else:
            xyz = 'Z'
        results.append({
            'SKU': sku, '品名': name, '库存': stock, '单价(美元)': unit_price,
            '年销量': annual_quantity, '年销售额': annual_value, 'CV': cv, 'XYZ': xyz
        })
    df = pd.DataFrame(results)

    df_sorted = df.sort_values('年销售额', ascending=False).reset_index(drop=True)
    total_value = df_sorted['年销售额'].sum()
    if total_value == 0:
        df_sorted['ABC'] = 'C'
    else:
        cumsum = 0.0
        a_end_idx = 0
        for i, val in enumerate(df_sorted['年销售额']):
            cumsum += val
            if cumsum / total_value >= a_ratio:
                a_end_idx = i + 1
                break
        else:
            a_end_idx = len(df_sorted)
        rest_df = df_sorted.iloc[a_end_idx:].copy()
        if len(rest_df) > 0 and b_ratio > 0:
            cumsum = 0.0
            b_end_idx = 0
            for i, val in enumerate(rest_df['年销售额']):
                cumsum += val
                if cumsum / total_value >= b_ratio:
                    b_end_idx = i + 1
                    break
            else:
                b_end_idx = len(rest_df)
        else:
            b_end_idx = 0
        abc_labels = ['A'] * a_end_idx + ['B'] * b_end_idx + ['C'] * (len(df_sorted) - a_end_idx - b_end_idx)
        df_sorted['ABC'] = abc_labels
        df_sorted = df_sorted.sort_index()
    df_abc = df_sorted.copy()
    df_abc['累计占比'] = df_abc['年销售额'].cumsum() / total_value if total_value > 0 else 0
    df_abc['累计占比'] = df_abc['累计占比'].apply(lambda x: f"{x*100:.1f}%")
    abc_result = df_abc[['SKU', '品名', '年销售额', '累计占比', 'ABC']].copy()

    xyz_result = df[['SKU', '品名', '年销量', 'CV', 'XYZ']].copy()

    df_combined = df_abc[['SKU', '品名', 'ABC']].merge(xyz_result[['SKU', 'XYZ']], on='SKU')
    df_combined['ABC-XYZ分类'] = df_combined['ABC'] + df_combined['XYZ']
    final = df_combined.merge(df[['SKU', '库存', '单价(美元)']], on='SKU')
    final = final[['SKU', '品名', '库存', '单价(美元)', 'ABC', 'XYZ', 'ABC-XYZ分类']]
    return abc_result, xyz_result, final

def get_history_hash(sku):
    hist = st.session_state.history_data[sku].dropna().tolist()
    hist_str = json.dumps(hist)
    return hashlib.md5(hist_str.encode()).hexdigest()

def get_forecast_cache_key(sku, horizon):
    hist_hash = get_history_hash(sku)
    return f"{sku}_{horizon}_{hist_hash}"

def get_forecast_result(sku, horizon):
    if "forecast_cache" not in st.session_state:
        st.session_state.forecast_cache = {}
    cache = st.session_state.forecast_cache
    key = get_forecast_cache_key(sku, horizon)
    if key in cache:
        return cache[key]
    classification_df = get_classification_df()
    sku_class_row = classification_df[classification_df["SKU"] == sku]
    if sku_class_row.empty:
        combined_class = "CX"
    else:
        combined_class = sku_class_row["ABC-XYZ分类"].iloc[0]
    history = st.session_state.history_data[sku].dropna().tolist()
    if len(history) < 24:
        best_model_name = "简单移动平均法"
        model_info = {"window": 3}
        val_rmse = np.inf
        val_mape = np.inf
        all_models = {}
        future_preds = forecast_future(combined_class, history, horizon, best_model_name, model_info)
    else:
        best_model_name, _, val_rmse, val_mape, model_info, all_models = select_best_model(combined_class, history, horizon)
        future_preds = forecast_future(combined_class, history, horizon, best_model_name, model_info)
    result = (best_model_name, model_info, future_preds, val_rmse, val_mape, all_models)
    cache[key] = result
    return result

def clear_forecast_cache():
    if "forecast_cache" in st.session_state:
        st.session_state.forecast_cache.clear()

def get_classification_df():
    if "classification_df" not in st.session_state or st.session_state.classification_df.empty:
        if st.session_state.inventory_data.empty or st.session_state.history_data.empty:
            return pd.DataFrame()
        a = st.session_state.get("abc_ratio_a", 0.6)
        b = st.session_state.get("abc_ratio_b", 0.25)
        _, _, df = compute_abc_xyz_classification(
            st.session_state.inventory_data,
            st.session_state.history_data,
            st.session_state.sku_params,
            a_ratio=a,
            b_ratio=b
        )
        st.session_state.classification_df = df
    return st.session_state.classification_df

def clear_all_caches():
    clear_forecast_cache()
    if "classification_df" in st.session_state:
        del st.session_state.classification_df

# ==================== 顶部栏 ====================
def render_top_bar():
    current = st.session_state.get("current_page", "主页")
    if current in ["首页", "主页"]:
        left, right = st.columns([1, 1])
        with left:
            pass
    else:
        left, right = st.columns([1, 1])
        with left:
            st.markdown('<p style="font-size: 1.5rem; font-weight: bold; margin: 0;">智能库存决策系统</p>', unsafe_allow_html=True)
    with right:
        spacer, btn_col = st.columns([1, 1])
        with btn_col:
            with st.popover("智能助手", use_container_width=True):
                if st.session_state.deepseek_api_key is None:
                    st.markdown("### <i class='fas fa-key'></i> 首次使用需配置 DeepSeek API Key", unsafe_allow_html=True)
                    st.caption("请填入您的 DeepSeek API Key（可从 platform.deepseek.com 获取）")
                    api_key_input = st.text_input("API Key", type="password", key="api_key_input")
                    if st.button("保存 Key", key="save_api_key"):
                        if api_key_input.strip():
                            st.session_state.deepseek_api_key = api_key_input.strip()
                            st.success("Key 已保存，现在可以开始对话！")
                            st.rerun()
                        else:
                            st.error("请输入有效的 API Key")
                else:
                    st.markdown(f"✅ 已配置 API Key（以 `{st.session_state.deepseek_api_key[:6]}...` 结尾）")
                    col_reset, _ = st.columns([1, 3])
                    with col_reset:
                        if st.button("清除 Key", key="reset_api_key"):
                            st.session_state.deepseek_api_key = None
                            st.session_state.chat_history = []
                            st.rerun()
                    st.markdown("---")
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []
                    for msg in st.session_state.chat_history:
                        if msg.startswith("你:"):
                            with st.chat_message("user"):
                                st.write(msg[2:].strip())
                        elif msg.startswith("助手:"):
                            with st.chat_message("assistant"):
                                st.write(msg[3:].strip())
                    user_query = st.chat_input("请输入问题...")
                    if user_query:
                        st.session_state.chat_history.append(f"你: {user_query}")
                        with st.spinner("AI 思考中..."):
                            response = chat_response(user_query, st.session_state)
                        st.session_state.chat_history.append(f"助手: {response}")
                        st.rerun()

# ==================== 侧边栏参数渲染 ====================
def render_sidebar_by_page(page_name):
    # 库存概览页面不显示任何全局参数（完全清空侧边栏参数区域）
    if page_name == "库存概览":
        return
    if page_name in ["主页", "产品分类"]:
        return
    with st.sidebar:
        st.markdown("### <i class='fas fa-sliders-h'></i> 全局参数设置", unsafe_allow_html=True)
        st.markdown("---")
        new_horizon = st.slider(
            "预测步长", 
            min_value=1, max_value=12, 
            value=st.session_state.common_params["forecast_horizon"], 
            step=1,
            help="预测未来几个月的需求量。步长越大，预测周期越长，但误差可能增大。"
        )
        if new_horizon != st.session_state.common_params["forecast_horizon"]:
            st.session_state.common_params["forecast_horizon"] = new_horizon
            clear_forecast_cache()
            st.rerun()
        if page_name == "需求预测":
            return
        if page_name in ["库存概览", "库存建议"]:
            current_service_level = st.session_state.common_params["service_level"]
            new_service_level = st.number_input(
                "服务水平",
                min_value=0.50,
                max_value=0.9999,
                value=current_service_level,
                step=0.01,
                format="%.4f",
                help="库存满足需求的比例。例如95%表示在100次订货中，有95次不缺货。数值越高，安全库存越大。"
            )
            if abs(new_service_level - current_service_level) > 1e-6:
                st.session_state.common_params["service_level"] = new_service_level
            new_leadtime_mean = st.slider(
                "提前期均值（天）", 
                min_value=20, max_value=80, 
                value=st.session_state.common_params["leadtime_mean"], 
                step=1,
                help="从下达采购订单到货物上架的平均天数。跨境物流通常较长（47天左右）。"
            )
            if new_leadtime_mean != st.session_state.common_params["leadtime_mean"]:
                st.session_state.common_params["leadtime_mean"] = new_leadtime_mean
            new_leadtime_std = st.number_input(
                "提前期标准差（天）", 
                value=st.session_state.common_params["leadtime_std"], 
                step=0.5,
                format="%.1f",
                help="提前期的波动程度，越大表示交货时间越不稳定。"
            )
            if new_leadtime_std != st.session_state.common_params["leadtime_std"]:
                st.session_state.common_params["leadtime_std"] = new_leadtime_std
            st.markdown("---")
            with st.expander("物流费率"):
                new_ocean_freight = st.number_input("头程海运费（美元/箱）", value=st.session_state.common_params["ocean_freight"], step=100)
                new_land_freight = st.number_input("陆运费（美元/箱）", value=st.session_state.common_params["land_freight"], step=50)
                if new_ocean_freight != st.session_state.common_params["ocean_freight"]:
                    st.session_state.common_params["ocean_freight"] = new_ocean_freight
                if new_land_freight != st.session_state.common_params["land_freight"]:
                    st.session_state.common_params["land_freight"] = new_land_freight
            with st.expander("仓储与清关费率"):
                new_holding_fee = st.number_input("仓储费率（美元/立方米/天）", value=st.session_state.common_params["holding_fee_rate_g"], step=0.01, format="%.3f")
                new_port_fee = st.number_input("港口维护费率", value=st.session_state.common_params["port_fee_rate"], step=0.0001, format="%.5f")
                new_handling_fee = st.number_input("货物处理费率", value=st.session_state.common_params["handling_fee_rate"], step=0.0001, format="%.5f")
                new_clearance_base = st.number_input("清关基本服务费（美元/次）", value=st.session_state.common_params["clearance_base_fee"], step=50)
                if new_holding_fee != st.session_state.common_params["holding_fee_rate_g"]:
                    st.session_state.common_params["holding_fee_rate_g"] = new_holding_fee
                if new_port_fee != st.session_state.common_params["port_fee_rate"]:
                    st.session_state.common_params["port_fee_rate"] = new_port_fee
                if new_handling_fee != st.session_state.common_params["handling_fee_rate"]:
                    st.session_state.common_params["handling_fee_rate"] = new_handling_fee
                if new_clearance_base != st.session_state.common_params["clearance_base_fee"]:
                    st.session_state.common_params["clearance_base_fee"] = new_clearance_base

# ==================== 初始化 session_state ====================
if "inventory_data" not in st.session_state:
    st.session_state.inventory_data = pd.DataFrame(columns=["SKU", "品名", "库存", "单价(美元)"])

if "history_data" not in st.session_state:
    st.session_state.history_data = pd.DataFrame()

if "common_params" not in st.session_state:
    st.session_state.common_params = {
        "service_level": 0.95,
        "forecast_horizon": 3,
        "leadtime_mean": 47,
        "leadtime_std": 1.0,
        "ocean_freight": 2200,
        "land_freight": 200,
        "holding_fee_rate_g": 0.20,
        "port_fee_rate": 0.00125,
        "handling_fee_rate": 0.003464,
        "clearance_base_fee": 300
    }

DEFAULT_SKU_PARAMS = {
    "128-2412-BK": {"combined_class": "AX","volume_m3": 0.30,"weight_kg": 25,"purchase_price": 70,"shortage_cost": 30,"fixed_order_cost": 400,"container_capacity": 193,"tariff_rate": 0.55,"in_fee": 0.60,"out_fee": 7.00},
    "123-2412-BK": {"combined_class": "AY","volume_m3": 0.28,"weight_kg": 22,"purchase_price": 65,"shortage_cost": 30,"fixed_order_cost": 400,"container_capacity": 207,"tariff_rate": 0.55,"in_fee": 0.60,"out_fee": 7.00},
    "121-2412-BK": {"combined_class": "CZ","volume_m3": 0.45,"weight_kg": 35,"purchase_price": 58,"shortage_cost": 30,"fixed_order_cost": 400,"container_capacity": 128,"tariff_rate": 0.55,"in_fee": 0.60,"out_fee": 7.00},
    "131-08B": {"combined_class": "BY","volume_m3": 0.12,"weight_kg": 3,"purchase_price": 55,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 400,"tariff_rate": 0.35,"in_fee": 0.24,"out_fee": 2.00},
    "132-10B": {"combined_class": "BX","volume_m3": 0.10,"weight_kg": 2.5,"purchase_price": 50,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 480,"tariff_rate": 0.35,"in_fee": 0.24,"out_fee": 2.00},
    "133-24B": {"combined_class": "CY","volume_m3": 0.035,"weight_kg": 1.8,"purchase_price": 35,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 1370,"tariff_rate": 0.35,"in_fee": 0.24,"out_fee": 2.00},
    "141-18B": {"combined_class": "AX","volume_m3": 0.020,"weight_kg": 1.5,"purchase_price": 35,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 2500,"tariff_rate": 0.25,"in_fee": 0.15,"out_fee": 0.80},
    "142-09B": {"combined_class": "CX","volume_m3": 0.010,"weight_kg": 1.0,"purchase_price": 27,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 5000,"tariff_rate": 0.25,"in_fee": 0.15,"out_fee": 0.80},
    "143-06B": {"combined_class": "CX","volume_m3": 0.008,"weight_kg": 0.5,"purchase_price": 23,"shortage_cost": 20,"fixed_order_cost": 400,"container_capacity": 6250,"tariff_rate": 0.25,"in_fee": 0.15,"out_fee": 0.80}
}

if "sku_params" not in st.session_state:
    st.session_state.sku_params = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = None

if "demand_view_mode" not in st.session_state:
    st.session_state.demand_view_mode = "summary"
if "selected_sku" not in st.session_state:
    st.session_state.selected_sku = ""
if "inventory_view_mode" not in st.session_state:
    st.session_state.inventory_view_mode = "summary"
if "manual_forecast_enabled" not in st.session_state:
    st.session_state.manual_forecast_enabled = False
if "manual_model_name" not in st.session_state:
    st.session_state.manual_model_name = None
if "manual_params_dict" not in st.session_state:
    st.session_state.manual_params_dict = {}
if "manual_model_desc" not in st.session_state:
    st.session_state.manual_model_desc = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "主页"
if "abc_ratio_a" not in st.session_state:
    st.session_state.abc_ratio_a = 0.60
if "abc_ratio_b" not in st.session_state:
    st.session_state.abc_ratio_b = 0.25
if "xyz_threshold_x" not in st.session_state:
    st.session_state.xyz_threshold_x = 0.25
if "xyz_threshold_yz" not in st.session_state:
    st.session_state.xyz_threshold_yz = 0.50

# ==================== 页面1：主页 ====================
def home_page():
    st.session_state.current_page = "主页"
    render_top_bar()
    render_sidebar_by_page(st.session_state.current_page)
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown('''
        <div style="text-align: left; margin-bottom: 1rem;">
            <div class="cover-title">智能库存决策系统</div>
            <div class="cover-subtitle">Inventory Decision Intelligence</div>
        </div>
    ''', unsafe_allow_html=True)
    
    has_data = (not st.session_state.inventory_data.empty) and (not st.session_state.history_data.empty)
    
    if not has_data:
        st.markdown("""
        <div class="guide-card">
            <div class="guide-title">开始使用</div>
            <div class="guide-desc">请从左侧菜单进入「产品分类」上传您的库存数据与历史销量数据，系统将自动进行 ABC‑XYZ 分类并提供库存决策建议。</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="margin: 1.5rem 0 1rem 0;">快速开始</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon"><i class="fas fa-tags"></i></div>
            <div class="quick-title">产品分类</div>
            <div class="quick-desc">上传库存与销量文件，自动进行 ABC‑XYZ 分类</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon"><i class="fas fa-chart-line"></i></div>
            <div class="quick-title">需求预测</div>
            <div class="quick-desc">查看推荐模型与预测结果</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon"><i class="fas fa-eye"></i></div>
            <div class="quick-title">库存概览</div>
            <div class="quick-desc">监控库存健康度与周转</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="quick-card">
            <div class="quick-icon"><i class="fas fa-clipboard-list"></i></div>
            <div class="quick-title">库存建议</div>
            <div class="quick-desc">获取补货策略与参数</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 页面2：产品分类 ====================
def classification_page():
    st.session_state.current_page = "产品分类"
    render_top_bar()
    render_sidebar_by_page(st.session_state.current_page)
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.markdown("## 产品分类")

    has_inventory = not st.session_state.inventory_data.empty
    has_history = not st.session_state.history_data.empty
    if has_inventory:
        st.success(f"✅ 当前已加载库存数据：{len(st.session_state.inventory_data)} 个 SKU")
    if has_history:
        st.success(f"✅ 当前已加载历史销量数据：{len(st.session_state.history_data.columns)} 个 SKU")

    with st.expander("数据上传", expanded=False):
        # 应用紧凑样式
        st.markdown('<div class="compact-expander">', unsafe_allow_html=True)
        st.markdown("#### 第一步：下载模板", unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            stock_template = pd.DataFrame({
                "SKU": ["128-2412-BK", "141-18B"],
                "品名": ["黑色基础款围栏", "智能自动喂食碗"],
                "库存": [817, 2463],
                "单价(美元)": [419.93, 209.93],
                "体积(m³)": [0.30, 0.02],
                "重量(kg)": [25, 1.5],
                "采购单价(美元)": [70, 35],
                "缺货成本(美元/件)": [30, 20],
                "固定订货成本(美元/次)": [400, 400],
                "集装箱容量(件/箱)": [193, 2500],
                "关税税率": [0.55, 0.25],
                "入仓费(美元/件)": [0.60, 0.15],
                "出仓费(美元/件)": [7.00, 0.80]
            })
            stock_buffer = io.BytesIO()
            stock_template.to_excel(stock_buffer, index=False)
            stock_buffer.seek(0)
            st.download_button("下载库存模板 (Excel)", data=stock_buffer, file_name="库存模板.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with col_dl2:
            sales_template = pd.DataFrame({"日期": ["2024-01-01", "2024-02-01"]})
            sales_template["128-2412-BK"] = [3500, 3200]
            sales_template["141-18B"] = [2200, 2100]
            sales_template["示例SKU"] = [100, 120]
            sales_buffer = io.BytesIO()
            sales_template.to_excel(sales_buffer, index=False)
            sales_buffer.seek(0)
            st.download_button("下载销量模板 (Excel)", data=sales_buffer, file_name="销量模板.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        st.markdown("---")
        st.markdown("#### 第二步：上传数据", unsafe_allow_html=True)
        st.markdown("**库存数据**")
        stock_file = st.file_uploader("选择库存文件 (CSV/Excel)", type=["csv", "xlsx"], key="stock_upload")
        if stock_file:
            try:
                if stock_file.name.endswith(".csv"):
                    new_stock = pd.read_csv(stock_file)
                else:
                    new_stock = pd.read_excel(stock_file)
                required_cols = ["SKU", "库存"]
                missing = [col for col in required_cols if col not in new_stock.columns]
                if missing:
                    st.error(f"文件缺少必需列: {missing}。当前列名: {list(new_stock.columns)}")
                else:
                    if "品名" not in new_stock.columns:
                        new_stock["品名"] = new_stock["SKU"]
                    base_cols = ["SKU", "品名", "库存"]
                    if "单价(美元)" in new_stock.columns:
                        base_cols.append("单价(美元)")
                    else:
                        new_stock["单价(美元)"] = 0.0
                        base_cols.append("单价(美元)")
                    inventory_df = new_stock[base_cols].copy()
                    st.session_state.inventory_data = inventory_df

                    new_sku_params = {}
                    for _, row in new_stock.iterrows():
                        sku = row["SKU"]
                        default = DEFAULT_SKU_PARAMS.get(sku, {
                            "combined_class": "CX",
                            "volume_m3": 0.01,
                            "weight_kg": 1.0,
                            "purchase_price": 10,
                            "shortage_cost": 20,
                            "fixed_order_cost": 400,
                            "container_capacity": 5000,
                            "tariff_rate": 0.25,
                            "in_fee": 0.15,
                            "out_fee": 0.80
                        })
                        params = default.copy()
                        if "体积(m³)" in new_stock.columns:
                            try:
                                params["volume_m3"] = float(row["体积(m³)"])
                            except:
                                pass
                        if "重量(kg)" in new_stock.columns:
                            try:
                                params["weight_kg"] = float(row["重量(kg)"])
                                in_fee, out_fee = get_in_out_fee(params["weight_kg"])
                                params["in_fee"] = in_fee
                                params["out_fee"] = out_fee
                            except:
                                pass
                        if "采购单价(美元)" in new_stock.columns:
                            try:
                                params["purchase_price"] = float(row["采购单价(美元)"])
                            except:
                                pass
                        if "缺货成本(美元/件)" in new_stock.columns:
                            try:
                                params["shortage_cost"] = float(row["缺货成本(美元/件)"])
                            except:
                                pass
                        if "固定订货成本(美元/次)" in new_stock.columns:
                            try:
                                params["fixed_order_cost"] = float(row["固定订货成本(美元/次)"])
                            except:
                                pass
                        if "集装箱容量(件/箱)" in new_stock.columns:
                            try:
                                params["container_capacity"] = float(row["集装箱容量(件/箱)"])
                            except:
                                pass
                        if "关税税率" in new_stock.columns:
                            try:
                                params["tariff_rate"] = float(row["关税税率"])
                            except:
                                pass
                        if "入仓费(美元/件)" in new_stock.columns:
                            try:
                                params["in_fee"] = float(row["入仓费(美元/件)"])
                            except:
                                pass
                        if "出仓费(美元/件)" in new_stock.columns:
                            try:
                                params["out_fee"] = float(row["出仓费(美元/件)"])
                            except:
                                pass
                        new_sku_params[sku] = params
                    st.session_state.sku_params = new_sku_params
                    clear_all_caches()
                    st.success("✅ 库存数据已更新")
            except Exception as e:
                st.error(f"读取库存文件失败: {e}")
        
        st.markdown("---")
        st.markdown("**历史销量数据**")
        st.caption("文件第一列必须是日期列（格式如 2024-01-01），其他列为 SKU 销量。")
        sales_file = st.file_uploader("选择销量文件 (CSV/Excel)", type=["csv", "xlsx"], key="sales_upload")
        if sales_file:
            try:
                if sales_file.name.endswith(".csv"):
                    new_sales = pd.read_csv(sales_file)
                else:
                    new_sales = pd.read_excel(sales_file)
                date_col = new_sales.columns[0]
                new_sales[date_col] = pd.to_datetime(new_sales[date_col])
                new_sales[date_col] = new_sales[date_col].dt.strftime('%Y-%m')
                new_sales.set_index(date_col, inplace=True)
                st.session_state.history_data = new_sales
                clear_all_caches()
                st.success("✅ 销量数据已更新")
            except Exception as e:
                st.error(f"读取销量文件失败: {e}\n请确保第一列为日期列，格式如 2024-01-01。")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("数据预览", expanded=False):
        st.markdown("#### 库存数据预览")
        if not st.session_state.inventory_data.empty:
            st.dataframe(st.session_state.inventory_data, use_container_width=True, hide_index=True)
        else:
            st.info("尚未上传库存数据，请在上方上传。")
        
        st.markdown("#### 历史销量数据预览")
        if not st.session_state.history_data.empty:
            history_df = st.session_state.history_data.sort_index().reset_index()
            history_df.rename(columns={'index': '日期'}, inplace=True)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚未上传历史销量数据，请在上方上传。")

    has_inventory = not st.session_state.inventory_data.empty
    has_history = not st.session_state.history_data.empty
    if has_inventory and has_history:
        try:
            a_ratio = st.session_state.abc_ratio_a
            b_ratio = st.session_state.abc_ratio_b

            abc_result, xyz_result, combined_result = compute_abc_xyz_classification(
                st.session_state.inventory_data,
                st.session_state.history_data,
                st.session_state.sku_params,
                a_ratio=a_ratio,
                b_ratio=b_ratio
            )
            st.session_state.classification_df = combined_result

            abc_counts = combined_result["ABC"].value_counts().reset_index()
            abc_counts.columns = ["分类", "数量"]
            total_abc = abc_counts["数量"].sum()
            xyz_counts = combined_result["XYZ"].value_counts().reset_index()
            xyz_counts.columns = ["分类", "数量"]
            total_xyz = xyz_counts["数量"].sum()

            tab_abc, tab_xyz, tab_combined = st.tabs(["ABC 分类", "XYZ 分类", "ABC-XYZ 组合分类"])
            
            with tab_abc:
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    pareto_df = abc_result[["SKU", "年销售额"]].copy()
                    total_sales = pareto_df["年销售额"].sum()
                    pareto_df["累计占比"] = pareto_df["年销售额"].cumsum() / total_sales * 100
                    pareto_df["序号"] = range(1, len(pareto_df) + 1)
                    cum_vals = pareto_df["累计占比"].values
                    a_boundary_idx = None
                    ab_boundary_idx = None
                    for i, cum in enumerate(cum_vals):
                        if a_boundary_idx is None and cum >= a_ratio * 100:
                            a_boundary_idx = i + 1
                        if ab_boundary_idx is None and cum >= (a_ratio + b_ratio) * 100:
                            ab_boundary_idx = i + 1
                            break
                    bars = alt.Chart(pareto_df).mark_bar(color="#3B82F6", opacity=0.7).encode(
                        x=alt.X("序号:O", title="SKU (按年销售额降序)", axis=alt.Axis(labelAngle=0, tickCount=min(20, len(pareto_df)))),
                        y=alt.Y("年销售额:Q", title="年销售额 (美元)"),
                        tooltip=["SKU", "年销售额"]
                    )
                    line = alt.Chart(pareto_df).mark_line(color="#4B5563", strokeWidth=2, point=True).encode(
                        x="序号:O",
                        y=alt.Y("累计占比:Q", title="累计销售额占比 (%)", axis=alt.Axis(titleColor="#4B5563")),
                        tooltip=["累计占比"]
                    )
                    chart_pareto = alt.layer(bars, line).resolve_scale(y='independent').properties(
                        width="container",
                        height=460, 
                        title="销售额帕累托分析 (柱状图: 年销售额, 折线: 累计占比)"
                    )
                    if a_boundary_idx is not None:
                        rule_a = alt.Chart(pd.DataFrame({'x': [a_boundary_idx]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x:O')
                        chart_pareto = chart_pareto + rule_a
                    if ab_boundary_idx is not None and ab_boundary_idx != a_boundary_idx:
                        rule_ab = alt.Chart(pd.DataFrame({'x': [ab_boundary_idx]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x:O')
                        chart_pareto = chart_pareto + rule_ab
                    st.altair_chart(chart_pareto, use_container_width=True)
                
                with col_right:
                    color_abc = alt.Scale(domain=["A", "B", "C"], range=["#1E3A8A", "#60A5FA", "#93C5FD"])
                    chart_abc = alt.Chart(abc_counts).mark_arc().encode(
                        theta=alt.Theta(field="数量", type="quantitative"),
                        color=alt.Color(field="分类", type="nominal", scale=color_abc),
                        tooltip=["分类", "数量"]
                    ).properties(width=180, height=180, title="SKU 数量占比")
                    st.altair_chart(chart_abc, use_container_width=True)
                    
                    with st.form(key="abc_form"):
                        new_a = st.slider(
                            "A 类金额占比 (%)",
                            min_value=50, max_value=90, value=int(a_ratio*100), step=1,
                            key="abc_a_slider"
                        ) / 100.0
                        new_b = st.slider(
                            "B 类金额占比 (%)",
                            min_value=10, max_value=40, value=int(b_ratio*100), step=1,
                            key="abc_b_slider"
                        ) / 100.0
                        submitted = st.form_submit_button("应用并重新分类")
                        if submitted:
                            if new_a + new_b > 1.0:
                                st.warning("A类与B类金额占比之和不能超过100%，请重新调整。")
                            else:
                                st.session_state.abc_ratio_a = new_a
                                st.session_state.abc_ratio_b = new_b
                                clear_all_caches()
                                st.rerun()
                    c_ratio = 1 - st.session_state.abc_ratio_a - st.session_state.abc_ratio_b
                    st.markdown(f'<span style="font-size: 14px;">C 类金额占比：{c_ratio*100:.1f}%（调整后自动重新分类）</span>', unsafe_allow_html=True)
                
                abc_table = render_dataframe_as_custom_table(abc_result, add_index=True, format_dict={"年销售额": "{:.2f}"})
                st.markdown(abc_table, unsafe_allow_html=True)
            
            with tab_xyz:
                x_thresh = st.session_state.xyz_threshold_x
                yz_thresh = st.session_state.xyz_threshold_yz
                
                df_xyz_raw = xyz_result.copy()
                def classify_xyz(cv):
                    if cv < x_thresh:
                        return 'X'
                    elif cv < yz_thresh:
                        return 'Y'
                    else:
                        return 'Z'
                df_xyz_raw['new_XYZ'] = df_xyz_raw['CV'].apply(classify_xyz)
                
                new_xyz_counts = df_xyz_raw['new_XYZ'].value_counts().reset_index()
                new_xyz_counts.columns = ['分类', '数量']
                order_map = {'X':0, 'Y':1, 'Z':2}
                new_xyz_counts['order'] = new_xyz_counts['分类'].map(order_map)
                new_xyz_counts = new_xyz_counts.sort_values('order').drop('order', axis=1)
                
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    scatter_df = df_xyz_raw[['SKU', '品名', '年销量', 'CV', 'new_XYZ']].copy().dropna(subset=['年销量', 'CV'])
                    color_scale = alt.Scale(domain=['X', 'Y', 'Z'], range=['#1E3A8A', '#F59E0B', '#EF4444'])
                    points = alt.Chart(scatter_df).mark_circle(size=60).encode(
                        x=alt.X('年销量:Q', title='年销量（件）', scale=alt.Scale(type='linear')),
                        y=alt.Y('CV:Q', title='变异系数 (CV)', scale=alt.Scale(zero=True, domain=[0, max(1, scatter_df['CV'].max()*1.1)])),
                        color=alt.Color('new_XYZ:N', scale=color_scale, legend=alt.Legend(title='XYZ 分类')),
                        tooltip=['SKU', '品名', '年销量', 'CV', 'new_XYZ']
                    )
                    rule_x = alt.Chart(pd.DataFrame({'y': [x_thresh]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
                    rule_yz = alt.Chart(pd.DataFrame({'y': [yz_thresh]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q')
                    chart_scatter = (points + rule_x + rule_yz).properties(
                        width='container',
                        height=460,  
                        title='波动性-销量散点图（颜色按当前阈值重新分类）'
                    ).interactive()
                    st.altair_chart(chart_scatter, use_container_width=True)
                
                with col_right:
                    color_xyz = alt.Scale(domain=['X', 'Y', 'Z'], range=['#1E3A8A', '#F59E0B', '#EF4444'])
                    chart_xyz_pie = alt.Chart(new_xyz_counts).mark_arc().encode(
                        theta=alt.Theta(field='数量', type='quantitative'),
                        color=alt.Color(field='分类', type='nominal', scale=color_xyz),
                        tooltip=['分类', '数量']
                    ).properties(width=180, height=180, title='SKU 数量占比（按当前阈值）')
                    st.altair_chart(chart_xyz_pie, use_container_width=True)
                    
                    with st.form(key="xyz_form"):
                        new_x_thresh = st.slider(
                            "X/Y 阈值 (CV < ? 为 X 类)",
                            min_value=0.0, max_value=1.0, value=float(x_thresh), step=0.01,
                            key="xyz_x_slider"
                        )
                        new_yz_thresh = st.slider(
                            "Y/Z 阈值 (CV >= ? 为 Z 类)",
                            min_value=0.0, max_value=1.0, value=float(yz_thresh), step=0.01,
                            key="xyz_yz_slider"
                        )
                        submitted_xyz = st.form_submit_button("应用并重新分类")
                        if submitted_xyz:
                            if new_x_thresh > new_yz_thresh:
                                st.warning("X/Y 阈值应小于 Y/Z 阈值，请重新调整。")
                            else:
                                st.session_state.xyz_threshold_x = new_x_thresh
                                st.session_state.xyz_threshold_yz = new_yz_thresh
                                st.rerun()
                    st.markdown(f'<span style="font-size: 14px;">当前分类规则：X: CV < {st.session_state.xyz_threshold_x:.2f} &nbsp;&nbsp; Y: {st.session_state.xyz_threshold_x:.2f} ≤ CV < {st.session_state.xyz_threshold_yz:.2f} &nbsp;&nbsp; Z: CV ≥ {st.session_state.xyz_threshold_yz:.2f}</span>', unsafe_allow_html=True)
                
                xyz_display = df_xyz_raw[['SKU', '品名', '年销量', 'CV', 'new_XYZ']].rename(columns={'new_XYZ': 'XYZ'})
                xyz_display['CV'] = xyz_display['CV'].apply(lambda x: f"{x:.3f}")
                xyz_table = render_dataframe_as_custom_table(xyz_display, add_index=True, format_dict={"年销量": "{:.0f}"})
                st.markdown(xyz_table, unsafe_allow_html=True)
            
            with tab_combined:
                combined_table = render_dataframe_as_custom_table(combined_result, add_index=True, special_columns={"ABC-XYZ分类": "capsule_class"})
                st.markdown(combined_table, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"计算分类时出错: {e}。请检查数据格式是否正确。")
    else:
        missing = []
        if not has_inventory:
            missing.append("库存数据")
        if not has_history:
            missing.append("历史销量数据")
        st.warning(f"⚠️ 当前缺少：{'、'.join(missing)}。请同时上传库存数据和历史销量数据，系统将自动进行 ABC‑XYZ 分类。")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 页面3：需求预测 ====================
def demand_forecast_page():
    st.session_state.current_page = "需求预测"
    render_top_bar()
    render_sidebar_by_page(st.session_state.current_page)
    def calculate_mape(actual, pred):
        actual = np.array(actual)
        pred = np.array(pred)
        mask = actual != 0
        if np.sum(mask) == 0:
            return 100.0
        return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

    def manual_forecast(model_name, params_dict, train, horizon):
        if model_name == "二次指数平滑法":
            alpha = params_dict.get("alpha", 0.5)
            s1 = [train[0]]
            s2 = [train[0]]
            for t in range(1, len(train)):
                s1_t = alpha * train[t] + (1 - alpha) * s1[-1]
                s2_t = alpha * s1_t + (1 - alpha) * s2[-1]
                s1.append(s1_t)
                s2.append(s2_t)
            a_last = 2 * s1[-1] - s2[-1]
            b_last = (alpha / (1 - alpha)) * (s1[-1] - s2[-1]) if alpha < 1 else 0
            preds = [a_last + b_last * (k+1) for k in range(horizon)]
            return preds, f"α={alpha:.2f}"
        elif model_name == "ARIMA(自预测)":
            p = params_dict.get("p", 2)
            d = params_dict.get("d", 1)
            q = params_dict.get("q", 1)
            from statsmodels.tsa.arima.model import ARIMA
            history = list(train)
            preds = []
            for _ in range(horizon):
                model = ARIMA(history, order=(p, d, q))
                fit = model.fit()
                pred = fit.forecast(steps=1)[0]
                preds.append(pred)
                history.append(pred)
            return preds, f"ARIMA({p},{d},{q})"
        elif model_name == "Holt-Winters乘法":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            seasonal = params_dict.get("seasonal", 12)
            model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=seasonal, initialization_method='estimated')
            fit = model.fit(optimized=True, use_brute=True)
            preds = fit.forecast(horizon)
            return preds.tolist(), f"季节周期{seasonal}"
        elif model_name == "Holt-Winters加法":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            seasonal = params_dict.get("seasonal", 12)
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal, initialization_method='estimated')
            fit = model.fit(optimized=True, use_brute=True)
            preds = fit.forecast(horizon)
            return preds.tolist(), f"季节周期{seasonal}"
        elif model_name == "加权移动平均法":
            window = params_dict.get("window", 3)
            preds = weighted_moving_average(train, horizon, window=window, recursive=True)
            return preds, f"N={window}"
        elif model_name == "二次移动平均法":
            N = params_dict.get("N", 3)
            preds = double_moving_average(train, horizon, N)
            return preds, f"N={N}"
        else:
            window = params_dict.get("window", 3)
            preds = []
            hist = list(train)
            for _ in range(horizon):
                if len(hist) >= window:
                    pred = int(round(np.mean(hist[-window:])))
                else:
                    pred = int(round(np.mean(hist)))
                preds.append(pred)
                hist.append(pred)
            return preds, f"N={window}"

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    if st.session_state.history_data.empty or st.session_state.inventory_data.empty:
        st.warning("请先在「产品分类」页面中同时上传库存数据和历史销量数据。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    classification_df = get_classification_df()
    if classification_df.empty:
        st.warning("分类数据尚未生成，请返回「产品分类」页面确保数据完整。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("## 需求预测")
    tab1, tab2 = st.tabs(["汇总视图", "详细视图"])

    with tab1:
        st.session_state.demand_view_mode = "summary"
        horizon = st.session_state.common_params["forecast_horizon"]
        summary_data = []
        for sku in st.session_state.history_data.columns:
            if sku not in st.session_state.sku_params:
                continue
            sku_class_row = classification_df[classification_df["SKU"] == sku]
            if sku_class_row.empty:
                continue
            combined_class = sku_class_row["ABC-XYZ分类"].iloc[0]
            history = st.session_state.history_data[sku].dropna().tolist()
            if len(history) < 24:
                continue
            with st.spinner(f"加载 {sku} 预测..."):
                best_model_name, _, future_preds, best_rmse, _, _ = get_forecast_result(sku, horizon)
            total_pred = sum(future_preds)
            product_name = st.session_state.inventory_data[st.session_state.inventory_data["SKU"] == sku]["品名"].iloc[0] if sku in st.session_state.inventory_data["SKU"].values else sku
            summary_data.append({
                "SKU": sku,
                "品名": product_name,
                "ABC-XYZ分类": combined_class,
                "最优模型": "ARIMA" if best_model_name == "ARIMA(自预测)" else best_model_name,
                "验证集RMSE(件)": f"{best_rmse:.1f}",
                "预测需求量": f"{total_pred:.1f}"
            })
        summary_df = pd.DataFrame(summary_data)
        if summary_df.empty:
            st.info("暂无需求预测数据（请确保每个SKU的历史数据至少24个月）")
        else:
            st.markdown('<div class="compact-card"><div class="card-title">模型表现汇总</div>', unsafe_allow_html=True)
            table_html = render_dataframe_as_custom_table(summary_df, add_index=True)
            st.markdown(table_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.session_state.demand_view_mode = "detail"
        with st.sidebar:
            st.markdown("---")
            st.markdown("### <i class='fas fa-cog'></i> 手动指定预测模型", unsafe_allow_html=True)
            if st.session_state.selected_sku and st.session_state.selected_sku in classification_df["SKU"].values:
                sku_for_manual = st.session_state.selected_sku
            else:
                sku_for_manual = classification_df["SKU"].iloc[0] if not classification_df.empty else None
            if sku_for_manual:
                sku_class_row = classification_df[classification_df["SKU"] == sku_for_manual]
                combined_class_for_manual = sku_class_row["ABC-XYZ分类"].iloc[0]
                use_manual = st.checkbox(
                    "启用手动模型选择（关闭则使用系统自动最优模型）",
                    key="manual_model_checkbox_sidebar",
                    help="勾选后，您可以手动选择预测模型及参数，系统将使用您指定的模型进行预测（仅在详细视图下对当前选中的单个SKU生效）。"
                )
                if use_manual:
                    if combined_class_for_manual in ['AX', 'AY', 'BX']:
                        available_models = ["二次指数平滑法", "简单移动平均法", "二次移动平均法", "加权移动平均法"]
                    else:
                        available_models = ["二次指数平滑法", "ARIMA(自预测)", "Holt-Winters乘法", "Holt-Winters加法", "简单移动平均法", "二次移动平均法", "加权移动平均法"]
                    selected_model = st.selectbox(
                        "选择预测模型",
                        available_models,
                        key="manual_model_select_sidebar",
                        help="不同模型适用于不同数据特征：二次指数平滑法适合有趋势无季节性；ARIMA适合复杂时间序列；Holt-Winters适合有季节性数据。"
                    )
                    params_dict = {}
                    if selected_model == "二次指数平滑法":
                        alpha = st.slider("平滑系数 α", 0.05, 0.95, 0.50, 0.05, key="manual_alpha_sidebar", help="α越大，近期数据权重越高。")
                        params_dict["alpha"] = alpha
                        manual_model_desc = f"α={alpha:.2f}"
                    elif selected_model == "ARIMA(自预测)":
                        p = st.slider("自回归阶数 p", 0, 3, 2, 1, key="manual_p_sidebar", help="使用最近p个历史值预测。")
                        d = st.slider("差分阶数 d", 0, 2, 1, 1, key="manual_d_sidebar", help="使序列平稳所需的差分次数。")
                        q = st.slider("移动平均阶数 q", 0, 3, 1, 1, key="manual_q_sidebar", help="预测误差的影响周期。")
                        params_dict["p"] = p
                        params_dict["d"] = d
                        params_dict["q"] = q
                        manual_model_desc = f"ARIMA({p},{d},{q})"
                    elif selected_model in ["Holt-Winters乘法", "Holt-Winters加法"]:
                        seasonal = st.slider("季节周期长度（月）", 1, 24, 12, 1, key="manual_seasonal_sidebar", help="季节性周期，例如12表示年度季节性。")
                        params_dict["seasonal"] = seasonal
                        manual_model_desc = f"季节周期{seasonal}"
                    elif selected_model in ["简单移动平均法", "加权移动平均法"]:
                        window = st.slider("移动窗口期数 N", 3, 12, 3, 1, key="manual_window_sidebar", help="计算平均值所用的历史数据点数。")
                        params_dict["window"] = window
                        manual_model_desc = f"N={window}"
                    elif selected_model == "二次移动平均法":
                        N = st.slider("移动平均期数 N", 2, 5, 3, 1, key="manual_double_ma_N_sidebar", help="移动平均窗口大小。")
                        params_dict["N"] = N
                        manual_model_desc = f"N={N}"
                    st.session_state.manual_forecast_enabled = True
                    st.session_state.manual_model_name = selected_model
                    st.session_state.manual_params_dict = params_dict
                    st.session_state.manual_model_desc = manual_model_desc
                else:
                    st.session_state.manual_forecast_enabled = False
            else:
                st.info("请先在右侧选择 SKU")

        all_skus = st.session_state.inventory_data["SKU"].tolist()
        sku_display = [f"{sku} - {st.session_state.inventory_data[st.session_state.inventory_data['SKU']==sku]['品名'].values[0]}" for sku in all_skus]
        if not sku_display:
            st.warning("无 SKU 数据")
            st.stop()
        current_index = all_skus.index(st.session_state.selected_sku) if st.session_state.selected_sku in all_skus else 0
        selected_display = st.selectbox("选择SKU", sku_display, index=current_index)
        selected_sku = selected_display.split(" - ")[0]
        if selected_sku != st.session_state.selected_sku:
            st.session_state.selected_sku = selected_sku
            st.rerun()
        sku = st.session_state.selected_sku
        sku_class_row = classification_df[classification_df["SKU"] == sku]
        if sku_class_row.empty:
            st.error(f"未找到SKU {sku} 的分类信息")
            st.stop()
        combined_class = sku_class_row["ABC-XYZ分类"].iloc[0]
        st.markdown(f'<div class="sku-title">SKU {sku} 需求预测详情</div>', unsafe_allow_html=True)
        history_df = st.session_state.history_data
        if sku in history_df.columns:
            history = history_df[sku].dropna().tolist()
            horizon = st.session_state.common_params["forecast_horizon"]
            best_model_name, model_info, future_preds, val_rmse, val_mape, all_models = get_forecast_result(sku, horizon)
        else:
            st.error(f"SKU {sku} 无历史数据")
            st.stop()

        with st.expander("模型对比", expanded=False):
            compare_data = []
            for name, info in all_models.items():
                display_name = "ARIMA模型" if name == "ARIMA(自预测)" else name
                compare_data.append({"预测模型": display_name, "参数": info['params'], "RMSE (件)": f"{info['rmse']:.1f}", "MAPE (%)": f"{info['mape']:.1f}"})
            compare_df = pd.DataFrame(compare_data)
            table_html = render_dataframe_as_custom_table(compare_df, add_index=False)
            st.markdown(table_html, unsafe_allow_html=True)
            best_display_name = "ARIMA模型" if best_model_name == "ARIMA(自预测)" else best_model_name
            st.success(f"系统最优模型：{best_display_name} (滚动验证平均RMSE={val_rmse:.1f}件, MAPE={val_mape:.1f}%)")

        st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

        if st.session_state.get("manual_forecast_enabled", False) and st.session_state.get("manual_model_name") is not None:
            manual_preds, _ = manual_forecast(st.session_state.manual_model_name, st.session_state.manual_params_dict, history, horizon)
            future_preds = manual_preds
            pred_source = f"手动模型（{st.session_state.manual_model_name}，{st.session_state.manual_model_desc}）"
        else:
            pred_source = f"系统最优模型（{best_display_name}）"
        if len(history) > 0:
            last_date_str = st.session_state.history_data.index[-1]
            last_date = datetime.strptime(last_date_str, "%Y-%m")
        else:
            last_date = datetime(2025, 12, 1)
        date_list = []
        for i in range(horizon):
            next_date = last_date + relativedelta(months=i+1)
            date_list.append(next_date.strftime("%Y-%m"))
        pred_df = pd.DataFrame({"日期": date_list, "预测需求量（件）": [f"{p:.1f}" for p in future_preds]})
        st.write(f"**当前使用：{pred_source}**")
        pred_table_html = render_dataframe_as_custom_table(pred_df, add_index=False)
        st.markdown(pred_table_html, unsafe_allow_html=True)
        hist_indices = list(range(len(history)))
        pred_indices = list(range(len(history), len(history) + len(future_preds)))
        hist_df = pd.DataFrame({"月份序号": hist_indices, "销量（件）": history, "类型": "历史销量"})
        pred_df_plot = pd.DataFrame({"月份序号": pred_indices, "销量（件）": future_preds, "类型": "预测值"})
        plot_df = pd.concat([hist_df, pred_df_plot], ignore_index=True)
        line_color = alt.Scale(domain=["历史销量", "预测值"], range=["#3B82F6", "#4B5563"])
        chart = alt.Chart(plot_df).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("月份序号:Q", title="月份序号", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("销量（件）:Q", title="销量（件）"),
            color=alt.Color("类型:N", scale=line_color, legend=alt.Legend(title="")),
            tooltip=["月份序号", "销量（件）", "类型"]
        ).properties(height=400, width="container")
        st.altair_chart(chart, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 页面4：库存概览 ====================
def inventory_overview_page():
    st.session_state.current_page = "库存概览"
    render_top_bar()
    render_sidebar_by_page(st.session_state.current_page)
    
    st.markdown("""
    <style>
    .compact-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        padding: 12px !important;
        margin-bottom: 12px !important;
    }
    .compact-card .card-title {
        font-size: 16px;
        font-weight: bold;
        color: #1F2937;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E5E7EB;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    if st.session_state.history_data.empty or st.session_state.inventory_data.empty:
        st.warning("请先在「产品分类」页面中同时上传库存数据和历史销量数据。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    classification_df = get_classification_df()
    if classification_df.empty:
        st.warning("分类数据尚未生成，请返回「产品分类」页面确保数据完整。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("## 库存概览")
    
    # 整体库存指标卡片
    st.markdown('<div class="compact-card"><div class="card-title">整体库存指标</div>', unsafe_allow_html=True)
    total_stock_units = classification_df["库存"].sum()
    total_stock_value = (classification_df["库存"] * classification_df["单价(美元)"]).sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_stock_units:,.0f}</div><div class="metric-label">总库存数量 (件)</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${total_stock_value:,.0f}</div><div class="metric-label">总库存金额 (美元)</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(classification_df)}</div><div class="metric-label">SKU 数量</div></div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 各 SKU 当前库存水平
    st.markdown('<div class="compact-card"><div class="card-title">各 SKU 当前库存水平</div>', unsafe_allow_html=True)
    stock_data = st.session_state.inventory_data[["SKU", "库存"]].copy()
    chart_stock = alt.Chart(stock_data).mark_bar(color="#3B82F6").encode(
        x=alt.X("SKU:N", title="SKU", axis=alt.Axis(labelAngle=0, labelFontSize=10)),
        y=alt.Y("库存:Q", title="库存数量 (件)"),
        tooltip=["SKU", "库存"]
    ).properties(height=300)
    st.altair_chart(chart_stock, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 库存周转天数表格
    st.markdown('<div class="compact-card"><div class="card-title">库存周转天数估算</div>', unsafe_allow_html=True)
    avg_monthly = {}
    for sku in st.session_state.inventory_data["SKU"]:
        if sku in st.session_state.history_data.columns:
            sales_series = st.session_state.history_data[sku].dropna()
            avg_monthly[sku] = sales_series.mean() if len(sales_series) > 0 else 0
        else:
            avg_monthly[sku] = 0
    turnover_data = []
    for _, row in st.session_state.inventory_data.iterrows():
        sku = row["SKU"]
        stock = row["库存"]
        avg_daily = avg_monthly.get(sku, 0) / 30
        days = stock / avg_daily if avg_daily > 0 else np.inf
        turnover_data.append({"SKU": sku, "品名": row["品名"], "周转天数": f"{days:.0f}" if days != np.inf else "∞"})
    turnover_df = pd.DataFrame(turnover_data)
    turnover_html = render_dataframe_as_custom_table(turnover_df, add_index=True, special_columns={"周转天数": "days_warning"})
    st.markdown(turnover_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 低库存预警
    low_stock = st.session_state.inventory_data[st.session_state.inventory_data["库存"] < 100]
    if not low_stock.empty:
        st.markdown('<p class="low-stock-title">低库存预警（库存 < 100件）</p>', unsafe_allow_html=True)
        low_stock_display = low_stock[["SKU", "品名", "库存"]].copy()
        low_stock_html = render_dataframe_as_custom_table(low_stock_display, add_index=True)
        st.markdown(low_stock_html, unsafe_allow_html=True)
    else:
        st.markdown('<p class="low-stock-title">库存健康</p>', unsafe_allow_html=True)
        st.markdown('<div class="custom-success">无低库存商品</div>', unsafe_allow_html=True)

    # 专业散点图
    st.markdown('<div class="compact-card"><div class="card-title">库存金额与周转天数分析</div>', unsafe_allow_html=True)
    
    scatter_data = []
    for _, row in st.session_state.inventory_data.iterrows():
        sku = row["SKU"]
        avg_daily = avg_monthly.get(sku, 0) / 30
        if avg_daily > 0:
            days = row["库存"] / avg_daily
        else:
            days = np.inf
        if days == np.inf or days > 2000:
            continue
        stock_value = row["库存"] * row["单价(美元)"] if "单价(美元)" in row else 0
        if sku in st.session_state.history_data.columns:
            annual_qty = st.session_state.history_data[sku].dropna().sum()
        else:
            annual_qty = 0
        abc = classification_df[classification_df["SKU"] == sku]["ABC"].iloc[0] if sku in classification_df["SKU"].values else "C"
        scatter_data.append({
            "SKU": sku,
            "品名": row["品名"],
            "周转天数": days,
            "库存金额(美元)": stock_value,
            "年销量(件)": annual_qty,
            "ABC分类": abc
        })
    
    if scatter_data:
        scatter_df = pd.DataFrame(scatter_data)
        scatter_chart = alt.Chart(scatter_df).mark_circle().encode(
            x=alt.X("周转天数:Q", title="库存周转天数（天）", 
                    scale=alt.Scale(domain=[0, scatter_df["周转天数"].max() * 1.05]),
                    axis=alt.Axis(title="库存周转天数（天）", format="d")),
            y=alt.Y("库存金额(美元):Q", title="库存金额（美元）", scale=alt.Scale(type="log")),
            size=alt.Size("年销量(件):Q", title="年销量（件）", legend=alt.Legend(title="年销量")),
            color=alt.Color("ABC分类:N", scale=alt.Scale(domain=["A","B","C"], range=["#1E3A8A", "#60A5FA", "#93C5FD"]), 
                            title="ABC分类"),
            tooltip=["SKU", "品名", "周转天数", "库存金额(美元)", "年销量(件)", "ABC分类"]
        ).properties(height=400, width="container").interactive()
        st.altair_chart(scatter_chart, use_container_width=True)
        st.caption("💡 **说明**：每个点代表一个SKU。横轴“库存周转天数” = 当前库存 ÷ 日均销量，表示现有库存可支撑销售的天数。"
                   "纵轴为对数刻度。点越大表示年销量越高，颜色表示ABC分类。"
                   "将鼠标悬停在点上可查看详细信息。")
    else:
        st.info("暂无足够数据绘制散点图（请确保SKU有历史销量且周转天数有限）")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 策略匹配表格
    st.markdown('<div class="compact-card"><div class="card-title">策略匹配（根据 ABC-XYZ 分类）</div>', unsafe_allow_html=True)
    strategy_match = []
    for _, row in classification_df.iterrows():
        sku = row["SKU"]
        combined = row["ABC-XYZ分类"]
        if combined in ['AX', 'AY', 'BX']:
            strategy = "(R,Q) 连续检查"
        else:
            strategy = "(T,S) 周期性检查"
        strategy_match.append({"SKU": sku, "品名": row["品名"], "ABC-XYZ分类": combined, "匹配策略": strategy})
    match_df = pd.DataFrame(strategy_match)
    match_html = render_dataframe_as_custom_table(match_df, add_index=True, special_columns={"ABC-XYZ分类": "capsule_class"})
    st.markdown(match_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 页面5：库存建议 ====================
def inventory_advice_page():
    st.session_state.current_page = "库存建议"
    render_top_bar()
    render_sidebar_by_page(st.session_state.current_page)
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if st.session_state.history_data.empty or st.session_state.inventory_data.empty:
        st.warning("请先在「产品分类」页面中同时上传库存数据和历史销量数据。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    classification_df = get_classification_df()
    if classification_df.empty:
        st.warning("分类数据尚未生成，请返回「产品分类」页面确保数据完整。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("## 库存建议")

    tab1, tab2 = st.tabs(["汇总视图", "详细视图"])

    with tab1:
        st.markdown('<div class="compact-card"><div class="card-title">库存策略汇总</div>', unsafe_allow_html=True)
        common_dict = st.session_state.common_params
        sku_params_dict = st.session_state.sku_params
        inventory_df = st.session_state.inventory_data
        history_df = st.session_state.history_data
        horizon = common_dict["forecast_horizon"]

        summary_data = []
        for sku in inventory_df["SKU"]:
            params = sku_params_dict.get(sku)
            if not params:
                continue
            sku_class_row = classification_df[classification_df["SKU"] == sku]
            if sku_class_row.empty:
                continue
            combined_class = sku_class_row["ABC-XYZ分类"].iloc[0]
            if sku not in history_df.columns:
                continue

            _, _, future_preds, _, _, _ = get_forecast_result(sku, horizon)
            D = sum(future_preds)
            T_days = horizon * 30

            if sku in history_df.columns:
                hist_series = history_df[sku].dropna()
                if len(hist_series) >= 12:
                    monthly_std = np.std(hist_series, ddof=1)
                    sigma_d = monthly_std / np.sqrt(30)
                else:
                    sigma_d = 1.0
            else:
                sigma_d = 1.0

            leadtime_mean = common_dict["leadtime_mean"]
            leadtime_std = common_dict["leadtime_std"]
            shortage_cost = params["shortage_cost"]
            fixed_order_cost = params["fixed_order_cost"]
            clearance_base_fee = common_dict["clearance_base_fee"]
            container_cap = params["container_capacity"]
            ocean_freight = common_dict["ocean_freight"]
            land_freight = common_dict["land_freight"]
            purchase_price = params["purchase_price"]
            tariff_rate = params["tariff_rate"]
            port_fee_rate = common_dict["port_fee_rate"]
            handling_fee_rate = common_dict["handling_fee_rate"]
            in_fee = params["in_fee"]
            out_fee = params["out_fee"]
            holding_fee_rate_g = common_dict["holding_fee_rate_g"]
            volume_m3 = params["volume_m3"]
            C2 = holding_fee_rate_g * T_days * volume_m3
            inout_fee = in_fee + out_fee

            if combined_class in ['AX', 'AY', 'BX']:
                best_rq = optimize_RQ(
                    D=D, sigma_d=sigma_d, mu=leadtime_mean, sigma_L=leadtime_std,
                    service_level=common_dict["service_level"], C_fixed=fixed_order_cost,
                    customs_fixed_fee=clearance_base_fee, C2=C2,
                    container_capacity=container_cap,
                    trans_cost_per_container=ocean_freight + land_freight,
                    customs_var_per_unit=(tariff_rate + port_fee_rate + handling_fee_rate) * purchase_price,
                    inout_fee=inout_fee, purchase_price=purchase_price,
                    c_s=shortage_cost, T_days=T_days
                )
                R, safety, _, Q, n, total_cost_daily = best_rq
                strategy = "(R,Q)"
                param1 = f"R={R:.0f}"
                param2 = f"Q={Q:.0f}"
                current_stock = inventory_df[inventory_df["SKU"]==sku]["库存"].values[0]
                urgency = "紧急" if current_stock < R else ("注意" if current_stock < safety else "正常")
            else:
                z_service = norm.ppf(common_dict["service_level"])
                best_ts, _ = optimize_tS(
                    D=D, sigma_d=sigma_d, mu=leadtime_mean, sigma_L=leadtime_std,
                    rho=shortage_cost, C_fixed=fixed_order_cost,
                    customs_fixed_fee=clearance_base_fee, C2=C2,
                    container_capacity=container_cap,
                    trans_cost_per_container=ocean_freight+land_freight,
                    customs_var_per_unit=(tariff_rate+port_fee_rate+handling_fee_rate)*purchase_price,
                    inout_fee=inout_fee, purchase_price=purchase_price,
                    z_service=z_service, T_days=T_days
                )
                t, Q_actual, safety, S_target, _, n, total_cost_daily = best_ts
                strategy = "(t,S)"
                param1 = f"t={t}"
                param2 = f"S={S_target:.0f}"
                current_stock = inventory_df[inventory_df["SKU"]==sku]["库存"].values[0]
                urgency = "紧急" if current_stock < S_target else ("注意" if current_stock < safety else "正常")

            summary_data.append({
                "SKU": sku,
                "品名": inventory_df[inventory_df["SKU"]==sku]["品名"].values[0],
                "当前库存": current_stock,
                "策略": strategy,
                "参数1": param1,
                "参数2": param2,
                "安全库存": f"{safety:.1f}",
                "补货紧急程度": urgency,
                "日平均成本(美元)": f"{total_cost_daily:.0f}"
            })
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            table_html = render_dataframe_as_custom_table(summary_df, add_index=True, special_columns={"补货紧急程度": "urgency"})
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("暂无库存策略数据")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        left_col, right_content = st.columns([1, 6])
        with left_col:
            if not classification_df.empty and st.session_state.selected_sku in classification_df["SKU"].values:
                sku = st.session_state.selected_sku
            else:
                sku = classification_df["SKU"].iloc[0] if not classification_df.empty else None
            if sku is None:
                st.warning("无 SKU 数据")
                st.stop()
            params = st.session_state.sku_params.get(sku)
            if params:
                st.markdown('<div class="compact-card"><div class="card-title">SKU 参数</div>', unsafe_allow_html=True)
                new_purchase_price = st.number_input("采购单价（美元/件）", value=float(params["purchase_price"]), step=5.0)
                new_shortage_cost = st.number_input("单位缺货成本（美元/件）", value=float(params["shortage_cost"]), step=1.0, min_value=0.0)
                new_fixed_order_cost = st.number_input("固定订货成本（美元/次）", value=float(params["fixed_order_cost"]), step=50.0)
                new_container_cap = st.number_input("集装箱容量（件/箱）", value=int(params["container_capacity"]), step=10)
                new_tariff_rate = st.number_input("关税税率", value=float(params["tariff_rate"]), step=0.01, format="%.4f")
                new_volume_m3 = st.number_input("单件体积（m³）", value=float(params["volume_m3"]), step=0.01)
                new_weight_kg = st.number_input("单件重量（kg）", value=float(params["weight_kg"]), step=0.5)
                st.caption(f"入仓费: {params['in_fee']} 美元/件, 出仓费: {params['out_fee']} 美元/件 (根据重量自动匹配)")
                if st.button("保存修改", key="save_sku_params_detail"):
                    st.session_state.sku_params[sku]["purchase_price"] = new_purchase_price
                    st.session_state.sku_params[sku]["shortage_cost"] = new_shortage_cost
                    st.session_state.sku_params[sku]["fixed_order_cost"] = new_fixed_order_cost
                    st.session_state.sku_params[sku]["container_capacity"] = new_container_cap
                    st.session_state.sku_params[sku]["tariff_rate"] = new_tariff_rate
                    st.session_state.sku_params[sku]["volume_m3"] = new_volume_m3
                    st.session_state.sku_params[sku]["weight_kg"] = new_weight_kg
                    in_fee, out_fee = get_in_out_fee(new_weight_kg)
                    st.session_state.sku_params[sku]["in_fee"] = in_fee
                    st.session_state.sku_params[sku]["out_fee"] = out_fee
                    st.success("参数已保存")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        with right_content:
            all_skus = st.session_state.inventory_data["SKU"].tolist()
            sku_display = [f"{sku} - {st.session_state.inventory_data[st.session_state.inventory_data['SKU']==sku]['品名'].values[0]}" for sku in all_skus]
            if not sku_display:
                st.warning("无 SKU 数据")
                st.stop()
            current_index = all_skus.index(st.session_state.selected_sku) if st.session_state.selected_sku in all_skus else 0
            selected_display = st.selectbox("选择SKU", sku_display, index=current_index)
            selected_sku = selected_display.split(" - ")[0]
            if selected_sku != st.session_state.selected_sku:
                st.session_state.selected_sku = selected_sku
                st.rerun()
            sku = st.session_state.selected_sku
            params = st.session_state.sku_params.get(sku)
            if not params:
                st.error(f"未找到SKU {sku} 的元数据")
                st.stop()
            sku_class_row = classification_df[classification_df["SKU"] == sku]
            if sku_class_row.empty:
                st.error(f"未找到SKU {sku} 的分类信息")
                st.stop()
            combined_class = sku_class_row["ABC-XYZ分类"].iloc[0]
            st.markdown(f'<div class="sku-title">SKU {sku} 库存策略详情</div>', unsafe_allow_html=True)
            common = st.session_state.common_params
            history_df = st.session_state.history_data
            if sku in history_df.columns:
                history = history_df[sku].dropna().tolist()
                horizon = common["forecast_horizon"]
                _, _, future_preds, _, _, _ = get_forecast_result(sku, horizon)
            else:
                st.error(f"SKU {sku} 无历史数据")
                st.stop()
            D = sum(future_preds)
            T_days = common["forecast_horizon"] * 30

            if sku in history_df.columns:
                hist_series = history_df[sku].dropna()
                if len(hist_series) >= 12:
                    monthly_std = np.std(hist_series, ddof=1)
                    sigma_d = monthly_std / np.sqrt(30)
                else:
                    sigma_d = 1.0
            else:
                sigma_d = 1.0

            inout_fee = params["in_fee"] + params["out_fee"]
            C2 = common["holding_fee_rate_g"] * T_days * params["volume_m3"]
            if combined_class in ['AX', 'AY', 'BX']:
                best_rq = optimize_RQ(
                    D=D, sigma_d=sigma_d, mu=common["leadtime_mean"], sigma_L=common["leadtime_std"],
                    service_level=common["service_level"], C_fixed=params["fixed_order_cost"],
                    customs_fixed_fee=common["clearance_base_fee"], C2=C2,
                    container_capacity=params["container_capacity"],
                    trans_cost_per_container=common["ocean_freight"]+common["land_freight"],
                    customs_var_per_unit=(params["tariff_rate"]+common["port_fee_rate"]+common["handling_fee_rate"])*params["purchase_price"],
                    inout_fee=inout_fee, purchase_price=params["purchase_price"],
                    c_s=params["shortage_cost"], T_days=T_days
                )
                R, safety, E_short, Q, n, total_cost_daily = best_rq
                opt_result = {'strategy': '(R,Q) 连续检查', 'R': R, 'Q': Q, 'safety_stock': safety, 'order_times': n, 'total_cost': total_cost_daily, 'E_short': E_short}
            else:
                z_service = norm.ppf(common["service_level"])
                best_ts, _ = optimize_tS(
                    D=D, sigma_d=sigma_d, mu=common["leadtime_mean"], sigma_L=common["leadtime_std"],
                    rho=params["shortage_cost"], C_fixed=params["fixed_order_cost"],
                    customs_fixed_fee=common["clearance_base_fee"], C2=C2,
                    container_capacity=params["container_capacity"],
                    trans_cost_per_container=common["ocean_freight"]+common["land_freight"],
                    customs_var_per_unit=(params["tariff_rate"]+common["port_fee_rate"]+common["handling_fee_rate"])*params["purchase_price"],
                    inout_fee=inout_fee, purchase_price=params["purchase_price"],
                    z_service=z_service, T_days=T_days
                )
                t, Q_actual, safety, S_target, E_short, n, total_cost_daily = best_ts
                opt_result = {'strategy': '(t,S) 周期性检查', 't': t, 'S_target': S_target, 'safety_stock': safety, 'order_times': n, 'total_cost': total_cost_daily, 'E_short': E_short}
            st.markdown('<div class="compact-card"><div class="card-title">库存策略最优解</div>', unsafe_allow_html=True)
            opt = opt_result
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="param-card"><div class="param-name">策略类型</div><div class="param-value">{opt['strategy']}</div></div>""", unsafe_allow_html=True)
            with col2:
                if opt['strategy'] == '(R,Q) 连续检查':
                    st.markdown(f"""<div class="param-card"><div class="param-name">订货点 R</div><div class="param-value">{opt['R']:.0f} 件</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="param-card"><div class="param-name">检查周期 t</div><div class="param-value">{opt['t']} 天</div></div>""", unsafe_allow_html=True)
            with col3:
                if opt['strategy'] == '(R,Q) 连续检查':
                    st.markdown(f"""<div class="param-card"><div class="param-name">订货批量 Q</div><div class="param-value">{opt['Q']:.0f} 件</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="param-card"><div class="param-name">最高库存 S</div><div class="param-value">{opt['S_target']:.0f} 件</div></div>""", unsafe_allow_html=True)
            col4, col5 = st.columns(2)
            with col4:
                safety_val = opt['safety_stock']
                safety_display = f"{int(safety_val)} 件" if abs(safety_val - round(safety_val)) < 1e-6 else f"{safety_val:.1f} 件"
                st.markdown(f"""<div class="param-card"><div class="param-name">安全库存</div><div class="param-value">{safety_display}</div></div>""", unsafe_allow_html=True)
            with col5:
                st.markdown(f"""<div class="param-card"><div class="param-name">日平均成本</div><div class="param-value">${opt['total_cost']:.0f}</div></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="compact-card"><div class="card-title">辅助决策分析</div>', unsafe_allow_html=True)
            current_stock = st.session_state.inventory_data[st.session_state.inventory_data["SKU"]==sku]["库存"].values[0]
            if combined_class in ['AX', 'AY', 'BX']:
                R_opt = opt['R']
                Q_opt = opt['Q']
                safety_opt = opt['safety_stock']
                avg_inventory = Q_opt / 2 + safety_opt
                if current_stock < R_opt:
                    status = "不足"
                    status_class = "status-card-warning"
                    advice = f"当前库存为 {current_stock} 件，低于订货点 R={R_opt:.0f} 件，建议立即下单"
                    suggested_order = Q_opt
                elif current_stock < safety_opt:
                    status = "紧张"
                    status_class = "status-card-caution"
                    advice = f"当前库存为 {current_stock} 件，低于安全库存 {safety_opt:.0f} 件，建议关注"
                    suggested_order = Q_opt
                else:
                    status = "充足"
                    status_class = "status-card-success"
                    advice = f"当前库存为 {current_stock} 件，库存充足，无需立即补货"
                    suggested_order = 0
            else:
                S_target_opt = opt['S_target']
                safety_opt = opt['safety_stock']
                avg_inventory = (S_target_opt + safety_opt) / 2
                if current_stock < S_target_opt:
                    status = "不足"
                    status_class = "status-card-warning"
                    advice = f"当前库存为 {current_stock} 件，低于目标库存 S={S_target_opt:.0f} 件，建议补货至目标库存"
                    suggested_order = max(0, S_target_opt - current_stock)
                elif current_stock < safety_opt:
                    status = "紧张"
                    status_class = "status-card-caution"
                    advice = f"当前库存为 {current_stock} 件，低于安全库存 {safety_opt:.0f} 件，建议关注"
                    suggested_order = max(0, S_target_opt - current_stock)
                else:
                    status = "充足"
                    status_class = "status-card-success"
                    advice = f"当前库存为 {current_stock} 件，库存充足，无需立即补货"
                    suggested_order = 0
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                icon_class = "fa-check-circle" if status == "充足" else "fa-triangle-exclamation"
                st.markdown(f"""<div class="{status_class} status-card"><div class="status-icon"><i class="fas {icon_class}"></i></div><div><div class="status-text">库存{status}</div><div class="status-detail">{advice}</div></div></div>""", unsafe_allow_html=True)
            with col_b:
                if suggested_order > 0:
                    avg_daily_demand = D / (common['forecast_horizon'] * 30)
                    days_cover = suggested_order / avg_daily_demand if avg_daily_demand > 0 else 0
                    st.markdown(f"""<div class="status-card-info status-card"><div class="status-icon"><i class="fas fa-box"></i></div><div><div class="status-text">建议补货量</div><div class="status-detail">{suggested_order:.0f} 件（预计可售 {days_cover:.1f} 天）</div></div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="status-card-info status-card"><div class="status-icon"><i class="fas fa-check-circle"></i></div><div><div class="status-text">建议补货量</div><div class="status-detail">暂不补货，库存充足</div></div></div>""", unsafe_allow_html=True)
            with col_c:
                annual_demand = D / (common['forecast_horizon'] * 30) * 365
                turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0
                three_month_demand = D / (common['forecast_horizon'] * 30) * 90
                warning_text = "呆滞预警：当前库存超过未来3个月需求" if current_stock > three_month_demand else "库存水平正常，无明显呆滞风险"
                st.markdown(f"""<div class="status-card-info status-card"><div class="status-icon"><i class="fas fa-chart-line"></i></div><div><div class="status-text">预计年库存周转率</div><div class="status-detail">{turnover:.2f} 次/年</div><div class="status-detail" style="font-size:10px;">{warning_text}</div></div></div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== 主入口 ====================
# 注意：不再调用 render_top_bar()，因为已经在每个页面函数内部调用
pg = st.navigation([
    st.Page(home_page, title="主页", icon=":material/home:"),
    st.Page(classification_page, title="产品分类", icon=":material/category:"),
    st.Page(demand_forecast_page, title="需求预测", icon=":material/trending_up:"),
    st.Page(inventory_overview_page, title="库存概览", icon=":material/inventory:"),
    st.Page(inventory_advice_page, title="库存建议", icon=":material/recommend:"),
])
pg.run()
