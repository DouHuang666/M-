import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import hashlib
from scipy.stats import norm

# 导入主程序中的预测和优化模块（确保这些模块在 PYTHONPATH 中）
from forecast_models import select_best_model, forecast_future, weighted_moving_average, double_moving_average
from inventory_optim import optimize_RQ, optimize_tS

# ======================== 辅助函数（与主程序保持一致，但只使用传入的 session_state）========================
def get_history_hash(sku, history_data):
    hist = history_data[sku].dropna().tolist()
    hist_str = json.dumps(hist)
    return hashlib.md5(hist_str.encode()).hexdigest()

def get_forecast_cache_key(sku, horizon, history_data):
    hist_hash = get_history_hash(sku, history_data)
    return f"{sku}_{horizon}_{hist_hash}"

def get_forecast_result(sku, horizon, session_state):
    """从缓存获取或计算预测结果（与主程序逻辑一致）"""
    if "forecast_cache" not in session_state:
        session_state.forecast_cache = {}
    cache = session_state.forecast_cache
    key = get_forecast_cache_key(sku, horizon, session_state.history_data)
    if key in cache:
        return cache[key]

    # 获取分类
    classification_df = session_state.get("classification_df")
    if classification_df is None or classification_df.empty:
        combined_class = "CX"
    else:
        row = classification_df[classification_df["SKU"] == sku]
        combined_class = row["ABC-XYZ分类"].iloc[0] if not row.empty else "CX"

    history = session_state.history_data[sku].dropna().tolist()
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

def compute_inventory_for_sku(sku, session_state):
    """计算单个SKU的库存策略（与主程序库存建议详细视图逻辑一致）"""
    common_params = session_state.common_params
    sku_params = session_state.sku_params
    inventory_data = session_state.inventory_data
    history_data = session_state.history_data

    params = sku_params.get(sku)
    if not params:
        return None

    # 获取分类
    classification_df = session_state.get("classification_df")
    if classification_df is None or classification_df.empty:
        combined_class = "CX"
    else:
        row = classification_df[classification_df["SKU"] == sku]
        combined_class = row["ABC-XYZ分类"].iloc[0] if not row.empty else "CX"

    horizon = common_params["forecast_horizon"]
    _, _, future_preds, _, _, _ = get_forecast_result(sku, horizon, session_state)
    D = sum(future_preds)          # 预测期内总需求
    T_days = horizon * 30

    # 需求标准差
    if sku in history_data.columns:
        hist_series = history_data[sku].dropna()
        if len(hist_series) >= 12:
            monthly_std = np.std(hist_series, ddof=1)
            sigma_d = monthly_std / np.sqrt(30)
        else:
            sigma_d = 1.0
    else:
        sigma_d = 1.0

    leadtime_mean = common_params["leadtime_mean"]
    leadtime_std = common_params["leadtime_std"]
    shortage_cost = params["shortage_cost"]
    fixed_order_cost = params["fixed_order_cost"]
    clearance_base_fee = common_params["clearance_base_fee"]
    container_cap = params["container_capacity"]
    ocean_freight = common_params["ocean_freight"]
    land_freight = common_params["land_freight"]
    purchase_price = params["purchase_price"]
    tariff_rate = params["tariff_rate"]
    port_fee_rate = common_params["port_fee_rate"]
    handling_fee_rate = common_params["handling_fee_rate"]
    in_fee = params["in_fee"]
    out_fee = params["out_fee"]
    holding_fee_rate_g = common_params["holding_fee_rate_g"]
    volume_m3 = params["volume_m3"]
    C2 = holding_fee_rate_g * T_days * volume_m3
    inout_fee = in_fee + out_fee

    current_stock = inventory_data[inventory_data["SKU"] == sku]["库存"].values[0]

    if combined_class in ['AX', 'AY', 'BX']:
        best_rq = optimize_RQ(
            D=D, sigma_d=sigma_d, mu=leadtime_mean, sigma_L=leadtime_std,
            service_level=common_params["service_level"], C_fixed=fixed_order_cost,
            customs_fixed_fee=clearance_base_fee, C2=C2,
            container_capacity=container_cap,
            trans_cost_per_container=ocean_freight + land_freight,
            customs_var_per_unit=(tariff_rate + port_fee_rate + handling_fee_rate) * purchase_price,
            inout_fee=inout_fee, purchase_price=purchase_price,
            c_s=shortage_cost, T_days=T_days
        )
        R, safety, _, Q, n, total_cost_daily = best_rq
        return {
            'strategy': '(R,Q)',
            'R': R,
            'Q': Q,
            'safety_stock': safety,
            'current_stock': current_stock,
            'total_cost': total_cost_daily * 365   # 年成本
        }
    else:
        z_service = norm.ppf(common_params["service_level"])
        best_ts, _ = optimize_tS(
            D=D, sigma_d=sigma_d, mu=leadtime_mean, sigma_L=leadtime_std,
            rho=shortage_cost, C_fixed=fixed_order_cost,
            customs_fixed_fee=clearance_base_fee, C2=C2,
            container_capacity=container_cap,
            trans_cost_per_container=ocean_freight + land_freight,
            customs_var_per_unit=(tariff_rate + port_fee_rate + handling_fee_rate) * purchase_price,
            inout_fee=inout_fee, purchase_price=purchase_price,
            z_service=z_service, T_days=T_days
        )
        t, Q_actual, safety, S_target, _, n, total_cost_daily = best_ts
        return {
            'strategy': '(t,S)',
            't': t,
            'S_target': S_target,
            'safety_stock': safety,
            'current_stock': current_stock,
            'total_cost': total_cost_daily * 365
        }

def check_alerts(session_state):
    """返回低库存和呆滞预警列表（文本格式）"""
    classification_df = session_state.get("classification_df")
    if classification_df is None or classification_df.empty:
        return "暂无分类数据，无法进行预警分析。"

    alerts = []
    # 低库存预警：当前库存低于订货点(R)或目标库存(S)
    for sku in classification_df["SKU"]:
        result = compute_inventory_for_sku(sku, session_state)
        if result:
            if result['strategy'] == '(R,Q)' and result['current_stock'] < result['R']:
                alerts.append(f"⚠️ {sku}：当前库存{result['current_stock']}件，低于订货点R={result['R']:.0f}件")
            elif result['strategy'] == '(t,S)' and result['current_stock'] < result['S_target']:
                alerts.append(f"⚠️ {sku}：当前库存{result['current_stock']}件，低于目标库存S={result['S_target']:.0f}件")

    # 呆滞预警：库存超过未来3个月预测需求
    horizon = session_state.common_params["forecast_horizon"]
    for sku in classification_df["SKU"]:
        try:
            _, _, preds, _, _, _ = get_forecast_result(sku, horizon, session_state)
            three_month_demand = sum(preds)
            stock = session_state.inventory_data[session_state.inventory_data["SKU"] == sku]["库存"].values[0]
            if stock > three_month_demand:
                alerts.append(f"📦 {sku}：当前库存{stock}件，超过未来3个月预测需求{three_month_demand:.0f}件，存在呆滞风险")
        except:
            continue

    if not alerts:
        return "✅ 当前无库存预警，所有SKU库存水平健康。"
    return "\n".join(alerts)

def get_slow_moving_skus(session_state):
    """获取周转天数超过90天的呆滞商品列表"""
    classification_df = session_state.get("classification_df")
    if classification_df is None or classification_df.empty:
        return "暂无分类数据。"

    slow = []
    history_data = session_state.history_data
    inventory_data = session_state.inventory_data
    for sku in classification_df["SKU"]:
        # 计算日均销量
        if sku in history_data.columns:
            sales = history_data[sku].dropna()
            avg_daily = sales.mean() / 30 if len(sales) > 0 else 0
        else:
            avg_daily = 0
        if avg_daily > 0:
            stock = inventory_data[inventory_data["SKU"] == sku]["库存"].values[0]
            days = stock / avg_daily
            if days > 90:
                slow.append(f"• {sku}：周转天数 {days:.0f} 天")
    if not slow:
        return "✅ 无呆滞商品（周转天数均≤90天）。"
    return "呆滞商品（周转 > 90天）：\n" + "\n".join(slow)

def generate_health_report(session_state):
    """生成库存健康报告摘要"""
    alerts = check_alerts(session_state)
    slow = get_slow_moving_skus(session_state)
    report = f"**库存健康报告**\n\n**预警信息**：\n{alerts}\n\n**呆滞商品**：\n{slow}"
    return report

def clear_forecast_cache(session_state):
    """只清除预测缓存，保留分类缓存（分类与服务水平/提前期无关）"""
    if "forecast_cache" in session_state:
        session_state.forecast_cache.clear()

# ======================== DeepSeek 工具定义 ========================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_inventory_alerts",
            "description": "获取当前所有库存预警（低库存、呆滞风险）",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_slow_moving_skus",
            "description": "获取周转天数超过90天的呆滞商品列表",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_health_report",
            "description": "生成库存健康报告（低库存与呆滞SKU摘要）",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sku_strategy",
            "description": "查询指定SKU的当前库存策略（包括订货点、目标库存、安全库存等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "SKU编号，例如 128-2412-BK"}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sku_classification",
            "description": "查询指定SKU的ABC分类、XYZ分类及组合分类（ABC-XYZ）",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "SKU编号"}
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_service_level",
            "description": "修改全局服务水平（0.80~0.9999），并自动重新计算所有库存策略",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_level": {"type": "number", "description": "服务水平，例如0.95表示95%"}
                },
                "required": ["new_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_leadtime",
            "description": "修改全局提前期均值（20~80天），并自动重新计算所有库存策略",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_leadtime": {"type": "integer", "description": "提前期天数，例如50"}
                },
                "required": ["new_leadtime"]
            }
        }
    }
]

def execute_tool(tool_name, arguments, session_state):
    if tool_name == "get_inventory_alerts":
        return check_alerts(session_state)
    elif tool_name == "get_slow_moving_skus":
        return get_slow_moving_skus(session_state)
    elif tool_name == "get_health_report":
        return generate_health_report(session_state)
    elif tool_name == "get_sku_strategy":
        sku = arguments.get("sku")
        if sku not in session_state.sku_params:
            return f"未找到SKU：{sku}，请检查编号是否正确（例如：128-2412-BK）。"
        result = compute_inventory_for_sku(sku, session_state)
        if not result:
            return f"无法计算SKU {sku} 的策略，请检查数据。"
        if result['strategy'] == '(R,Q)':
            return (f"SKU {sku} 当前策略：\n"
                    f"• 策略类型: (R,Q)连续检查\n"
                    f"• 订货点 R: {result['R']:.0f}件\n"
                    f"• 订货批量 Q: {result['Q']:.0f}件\n"
                    f"• 安全库存: {result['safety_stock']:.0f}件\n"
                    f"• 当前库存: {result['current_stock']}件\n"
                    f"• 年总成本: ${result['total_cost']:.0f}")
        else:
            return (f"SKU {sku} 当前策略：\n"
                    f"• 策略类型: (t,S)周期性检查\n"
                    f"• 检查周期 t: {result['t']}天\n"
                    f"• 目标库存 S: {result['S_target']:.0f}件\n"
                    f"• 安全库存: {result['safety_stock']:.0f}件\n"
                    f"• 当前库存: {result['current_stock']}件\n"
                    f"• 年总成本: ${result['total_cost']:.0f}")
    elif tool_name == "get_sku_classification":
        sku = arguments.get("sku")
        classification_df = session_state.get("classification_df")
        if classification_df is None or classification_df.empty:
            return "分类数据尚未生成，请先在「产品分类」页面上传数据并生成分类。"
        row = classification_df[classification_df["SKU"] == sku]
        if row.empty:
            return f"未找到SKU：{sku}"
        abc = row["ABC"].iloc[0]
        xyz = row["XYZ"].iloc[0]
        combined = row["ABC-XYZ分类"].iloc[0]
        return (f"SKU {sku} 的分类结果：\n"
                f"• ABC分类：{abc}\n"
                f"• XYZ分类：{xyz}\n"
                f"• 组合分类：{combined}\n"
                f"• 推荐策略：{'连续检查 (R,Q)' if combined in ['AX','AY','BX'] else '周期性检查 (t,S)'}")
    elif tool_name == "modify_service_level":
        new_sl = arguments.get("new_level")
        if not (0.80 <= new_sl <= 0.9999):
            return "服务水平必须在0.80到0.9999之间。"
        session_state.common_params["service_level"] = new_sl
        clear_forecast_cache(session_state)   # 只清除预测缓存
        return f"已将服务水平修改为 {new_sl*100:.2f}%。预测缓存已清空，后续查询将基于新参数重新计算。"
    elif tool_name == "modify_leadtime":
        new_lt = arguments.get("new_leadtime")
        if not (20 <= new_lt <= 80):
            return "提前期必须在20到80天之间。"
        session_state.common_params["leadtime_mean"] = new_lt
        clear_forecast_cache(session_state)
        return f"已将提前期修改为 {new_lt} 天。预测缓存已清空，后续查询将基于新参数重新计算。"
    else:
        return f"未知工具: {tool_name}"

# ======================== DeepSeek API 调用 ========================
def call_deepseek_with_tools(messages, session_state):
    # 优先从 session_state 获取 API Key
    api_key = session_state.get("deepseek_api_key")
    # 如果 session_state 中没有，尝试从 st.secrets 读取（Streamlit Cloud 部署）
    if not api_key:
        try:
            import streamlit as st
            secret_key = st.secrets.get("DEEPSEEK_API_KEY")
            if secret_key and isinstance(secret_key, str) and secret_key.startswith("sk-"):
                api_key = secret_key
                # 可选：同时保存到 session_state，避免重复读取 secrets
                session_state.deepseek_api_key = api_key
        except Exception:
            pass
    if not api_key:
        return "❌ 未配置 DeepSeek API Key。请点击右上角智能助手，在弹出的窗口中输入您的 API Key。或者在 Streamlit Cloud 的 Secrets 中设置 DEEPSEEK_API_KEY。"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 精简的系统提示，不包含硬编码知识，只强调必须调用函数
    system_content = (
        "你是库存决策支持系统的智能助手。\n"
        "你必须调用提供的函数来获取实时数据，不要编造任何数据。\n"
        "可用函数包括：\n"
        "- get_sku_classification：查询SKU的分类（ABC/XYZ）\n"
        "- get_sku_strategy：查询SKU的库存策略（订货点、安全库存等）\n"
        "- get_inventory_alerts：获取低库存和呆滞预警\n"
        "- get_health_report：获取健康报告\n"
        "- modify_service_level / modify_leadtime：修改全局参数\n"
        "当用户询问具体SKU或预警信息时，必须调用对应函数。回答要简洁、专业。"
    )

    full_messages = [{"role": "system", "content": system_content}] + messages

    while True:
        payload = {
            "model": "deepseek-chat",
            "messages": full_messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.7,
            "stream": False
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            assistant_msg = data["choices"][0]["message"]
            full_messages.append(assistant_msg)

            if assistant_msg.get("tool_calls"):
                for tool_call in assistant_msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    tool_result = execute_tool(tool_name, arguments, session_state)
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
                continue
            else:
                return assistant_msg["content"]
        except Exception as e:
            return f"调用 DeepSeek API 出错：{str(e)}。请检查网络或API Key。"

def chat_response(user_input, session_state):
    if "ai_conversation_history" not in session_state:
        session_state.ai_conversation_history = []
    session_state.ai_conversation_history.append({"role": "user", "content": user_input})
    answer = call_deepseek_with_tools(session_state.ai_conversation_history, session_state)
    session_state.ai_conversation_history.append({"role": "assistant", "content": answer})
    if len(session_state.ai_conversation_history) > 20:
        session_state.ai_conversation_history = session_state.ai_conversation_history[-20:]
    return answer

def render_api_key_config():
    st.markdown("### 🔑 配置 DeepSeek API Key")
    api_key_input = st.text_input("API Key", type="password", placeholder="sk-...", key="api_key_input_widget")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存并开始对话"):
            if api_key_input.startswith("sk-"):
                st.session_state.deepseek_api_key = api_key_input
                if "ai_conversation_history" in st.session_state:
                    st.session_state.ai_conversation_history = []
                st.success("API Key 已保存！")
                st.rerun()
            else:
                st.error("API Key 格式错误，应以 'sk-' 开头")
    with col2:
        if st.button("清除已保存的 Key"):
            if "deepseek_api_key" in st.session_state:
                del st.session_state.deepseek_api_key
            st.success("已清除")
            st.rerun()

def render_floating_ai_assistant():
    if "deepseek_api_key" not in st.session_state:
        st.session_state.deepseek_api_key = None
    if not st.session_state.deepseek_api_key:
        render_api_key_config()
        return
    st.markdown("### 🤖 AI 库存助手")
    if "ai_conversation_history" in st.session_state:
        for msg in st.session_state.ai_conversation_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    user_question = st.chat_input("例如：128-2412-BK 属于什么类别？")
    if user_question:
        with st.spinner("AI 思考中..."):
            answer = chat_response(user_question, st.session_state)
        st.rerun()
