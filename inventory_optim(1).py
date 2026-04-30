import numpy as np
from scipy.stats import norm

def optimize_RQ(D, sigma_d, mu, sigma_L, service_level,
                C_fixed, customs_fixed_fee, C2, container_capacity,
                trans_cost_per_container, customs_var_per_unit,
                inout_fee, purchase_price, c_s, T_days):
    """
    求解 (R,Q) 连续检查策略
    返回：(R, SS, E_short, Q_opt, n, total_cost_daily)
    """
    d_bar = D / T_days
    sigma_LTD = np.sqrt(mu * sigma_d**2 + (d_bar * sigma_L)**2)
    z = norm.ppf(service_level)
    SS = z * sigma_LTD
    R = d_bar * mu + SS
    E_short = sigma_LTD * norm.pdf(z) - (R - d_bar * mu) * (1 - norm.cdf(z))
    fixed_cost_total = C_fixed + customs_fixed_fee + c_s * E_short
    if fixed_cost_total <= 0:
        Q_eoq = container_capacity
    else:
        Q_eoq = np.sqrt(2 * D * fixed_cost_total / C2)
    if Q_eoq < 1e-6:
        Q_opt = container_capacity
    else:
        k_low = max(1, int(np.floor(Q_eoq / container_capacity)))
        k_high = k_low + 1
        Q_low = k_low * container_capacity
        Q_high = k_high * container_capacity
        def total_cost_without_constant(Q):
            n = D / Q
            order_cost_fixed = n * C_fixed
            holding_cost = C2 * (Q / 2 + SS)
            transport_and_var = D * (trans_cost_per_container / container_capacity + customs_var_per_unit)
            customs_fixed_cost = n * customs_fixed_fee
            shortage_cost = n * c_s * E_short
            return order_cost_fixed + holding_cost + transport_and_var + customs_fixed_cost + shortage_cost
        cost_low = total_cost_without_constant(Q_low)
        cost_high = total_cost_without_constant(Q_high)
        Q_opt = Q_low if cost_low <= cost_high else Q_high
    n = D / Q_opt
    order_cost_fixed_actual = n * C_fixed
    holding_cost_actual = C2 * (Q_opt / 2 + SS)
    transport_cost_actual = n * (trans_cost_per_container / container_capacity) * Q_opt
    var_cost_actual = n * customs_var_per_unit * Q_opt
    customs_fixed_actual = n * customs_fixed_fee
    shortage_cost_actual = n * c_s * E_short
    constant = purchase_price * D + inout_fee * D
    total_cost_period = order_cost_fixed_actual + holding_cost_actual + transport_cost_actual + var_cost_actual + customs_fixed_actual + shortage_cost_actual + constant
    total_cost_daily = total_cost_period / T_days
    return (R, SS, E_short, Q_opt, n, total_cost_daily)


def optimize_tS(D, sigma_d, mu, sigma_L, rho,
                C_fixed, customs_fixed_fee, C2, container_capacity,
                trans_cost_per_container, customs_var_per_unit,
                inout_fee, purchase_price, z_service, T_days):
    """
    求解 (t,S) 周期性检查策略
    返回：(T_opt, Q_opt, SS_opt, S_target, E_short, n_opt, total_cost_daily)
    注意：total_cost_daily 已经是日均总成本，前端直接使用，不要再除以 T_days。
    """
    d_bar = D / T_days
    constant = purchase_price * D + inout_fee * D
    # 候选检查周期：14天、30天、45天（符合跨境电商常见运营节奏）
    candidate_T = [14, 30, 45]
    best_cost_daily = float('inf')
    best_params = None
    for T in candidate_T:
        if T > T_days:   # 检查周期不能超过计划期总天数（如90天）
            continue
        Q = d_bar * T
        if Q < 1:
            continue
        sigma_TL = np.sqrt(sigma_d**2 * (T + mu) + (d_bar * sigma_L)**2)
        z = z_service
        SS = z * sigma_TL
        E_short = sigma_TL * norm.pdf(z) - SS * (1 - norm.cdf(z))
        n = D / Q
        containers_per_order = int(np.ceil(Q / container_capacity))
        transport_cost_per_order = containers_per_order * trans_cost_per_container
        order_cost = n * C_fixed
        holding_cost = C2 * (Q / 2 + SS)
        transport_cost = n * transport_cost_per_order
        var_cost = n * customs_var_per_unit * Q
        customs_fixed_cost = n * customs_fixed_fee
        shortage_cost = n * rho * E_short
        total_cost_period = order_cost + holding_cost + transport_cost + var_cost + customs_fixed_cost + shortage_cost + constant
        cost_daily = total_cost_period / T_days
        if cost_daily < best_cost_daily:
            best_cost_daily = cost_daily
            S_target = d_bar * (T + mu) + SS
            # 注意：返回的 cost_daily 就是日均总成本
            best_params = (T, int(Q), SS, S_target, E_short, int(n), cost_daily)
    # 保底返回 T=1（理论上候选列表中至少有一个有效，但以防万一）
    if best_params is None:
        T = 1
        Q = d_bar
        sigma_TL = np.sqrt(sigma_d**2 * (1 + mu) + (d_bar * sigma_L)**2)
        z = z_service
        SS = z * sigma_TL
        E_short = sigma_TL * norm.pdf(z) - SS * (1 - norm.cdf(z))
        n = D / Q
        containers_per_order = int(np.ceil(Q / container_capacity))
        transport_cost_per_order = containers_per_order * trans_cost_per_container
        order_cost = n * C_fixed
        holding_cost = C2 * (Q / 2 + SS)
        transport_cost = n * transport_cost_per_order
        var_cost = n * customs_var_per_unit * Q
        customs_fixed_cost = n * customs_fixed_fee
        shortage_cost = n * rho * E_short
        total_cost_period = order_cost + holding_cost + transport_cost + var_cost + customs_fixed_cost + shortage_cost + constant
        cost_daily = total_cost_period / T_days
        S_target = d_bar * (1 + mu) + SS
        best_params = (T, int(Q), SS, S_target, E_short, int(n), cost_daily)
    return best_params, None