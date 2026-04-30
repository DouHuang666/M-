import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def weighted_rmse(errors, weights=None):
    if weights is None:
        n = len(errors)
        weights = np.arange(1, n+1)
    weights = np.array(weights) / np.sum(weights)
    return np.sqrt(np.sum(weights * np.array(errors)**2))

def calculate_mape(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    mask = actual != 0
    if np.sum(mask) == 0:
        return 100.0
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100

def holt_exp_smoothing(train, horizon, sku_class=None):
    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_alpha = None
    best_train_rmse = float('inf')
    init_val = np.mean(train[:3]) if len(train) >= 3 else train[0]
    for alpha in alphas:
        s1 = [init_val]
        s2 = [init_val]
        errors = []
        for t in range(1, len(train)):
            s1_t = alpha * train[t] + (1 - alpha) * s1[-1]
            s2_t = alpha * s1_t + (1 - alpha) * s2[-1]
            a_t = 2 * s1_t - s2_t
            b_t = (alpha / (1 - alpha)) * (s1_t - s2_t) if alpha < 1 else 0
            pred = a_t + b_t
            errors.append((pred - train[t]) ** 2)
            s1.append(s1_t)
            s2.append(s2_t)
        train_rmse = weighted_rmse(errors)
        if train_rmse < best_train_rmse:
            best_train_rmse = train_rmse
            best_alpha = alpha
    s1 = [init_val]
    s2 = [init_val]
    for t in range(1, len(train)):
        s1_t = best_alpha * train[t] + (1 - best_alpha) * s1[-1]
        s2_t = best_alpha * s1_t + (1 - best_alpha) * s2[-1]
        s1.append(s1_t)
        s2.append(s2_t)
    a_last = 2 * s1[-1] - s2[-1]
    b_last = (best_alpha / (1 - best_alpha)) * (s1[-1] - s2[-1]) if best_alpha < 1 else 0
    preds = [max(0, a_last + b_last * (k+1)) for k in range(horizon)]
    return preds, best_alpha

def simple_moving_average(train, horizon, window=3, recursive=False):
    preds = []
    if recursive:
        hist = list(train)
        for _ in range(horizon):
            if len(hist) >= window:
                pred = np.mean(hist[-window:])
            else:
                pred = np.mean(hist)
            preds.append(pred)
            hist.append(pred)
    else:
        if len(train) >= window:
            base_pred = np.mean(train[-window:])
        else:
            base_pred = np.mean(train)
        preds = [base_pred] * horizon
    return preds

def weighted_moving_average(train, horizon, window=3, recursive=False):
    N = window
    denominator = N * (N + 1) / 2
    weights_recent_first = np.array([(N - i) / denominator for i in range(N)])
    weights_old_to_new = weights_recent_first[::-1]
    preds = []
    if recursive:
        hist = list(train)
        for _ in range(horizon):
            if len(hist) >= N:
                recent = hist[-N:]
                pred = np.dot(recent, weights_old_to_new)
            else:
                pred = np.mean(hist)
            preds.append(pred)
            hist.append(pred)
    else:
        if len(train) >= N:
            recent = train[-N:]
            base_pred = np.dot(recent, weights_old_to_new)
        else:
            base_pred = np.mean(train)
        preds = [base_pred] * horizon
    preds = [max(0, p) for p in preds]
    return preds

def arima_forecast(train, horizon):
    adf_res = adfuller(train)
    d = 1 if adf_res[1] > 0.05 else 0
    best_aic = np.inf
    best_order = None
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(train, order=(p, d, q))
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except:
                continue
    if best_order is None:
        best_order = (1, d, 1)
    model = ARIMA(train, order=best_order)
    fit = model.fit()
    preds = fit.forecast(steps=horizon).tolist()
    preds = [max(0, p) for p in preds]
    return preds, best_order

def holt_winters_forecast(train, horizon, seasonal=12, seasonal_type='mul'):
    if len(train) < seasonal * 2:
        return None
    try:
        trend = 'add'
        model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal_type,
                                     seasonal_periods=seasonal,
                                     initialization_method='estimated')
        fit = model.fit(optimized=True, use_brute=True)
        preds = fit.forecast(horizon).tolist()
        preds = [max(0, p) for p in preds]
        return preds
    except Exception as e:
        print(f"Holt-Winters {seasonal_type} 模型失败: {e}")
        return None

def rolling_multi_step_backtest(history, horizon, window=24, sku_class=None):
    n = len(history)
    max_start = n - window - horizon + 1
    if max_start <= 0:
        return None

    def _predict_sma(train, horizon):
        min_window = max(3, len(train)//4) if sku_class not in ['AX','AY','BX'] else max(6, len(train)//4)
        max_window = min(12, len(train))
        best_win = 3
        best_rmse = np.inf
        for win in range(min_window, max_window+1):
            hist = list(train[:win])
            errs = []
            for t in range(win, len(train)):
                pred = np.mean(hist[-win:])
                errs.append((pred - train[t])**2)
                hist.append(pred)
            rmse = np.sqrt(np.mean(errs))
            if rmse < best_rmse:
                best_rmse = rmse
                best_win = win
        hist = list(train[-best_win:])
        preds = []
        for _ in range(horizon):
            pred = np.mean(hist[-best_win:])
            preds.append(pred)
            hist.append(pred)
        return preds

    def _predict_weighted(train, horizon):
        min_window = max(3, len(train)//4) if sku_class not in ['AX','AY','BX'] else max(6, len(train)//4)
        max_window = min(12, len(train))
        best_win = 3
        best_rmse = np.inf
        for win in range(min_window, max_window+1):
            denominator = win * (win + 1) / 2
            weights_old_to_new = np.array([(i + 1) / denominator for i in range(win)])
            hist = list(train[:win])
            errs = []
            for t in range(win, len(train)):
                recent = hist[-win:]
                pred = np.dot(recent, weights_old_to_new)
                errs.append((pred - train[t])**2)
                hist.append(pred)
            rmse = np.sqrt(np.mean(errs))
            if rmse < best_rmse:
                best_rmse = rmse
                best_win = win
        denominator = best_win * (best_win + 1) / 2
        weights_old_to_new = np.array([(i + 1) / denominator for i in range(best_win)])
        hist = list(train[-best_win:])
        preds = []
        for _ in range(horizon):
            pred = np.dot(hist[-best_win:], weights_old_to_new)
            preds.append(pred)
            hist.append(pred)
        return preds

    def _predict_holt(train, horizon):
        preds, _ = holt_exp_smoothing(train, horizon, sku_class)
        return preds

    def _predict_arima(train, horizon):
        preds, _ = arima_forecast(train, horizon)
        return preds

    def _predict_hw_mul(train, horizon):
        return holt_winters_forecast(train, horizon, seasonal=12, seasonal_type='mul')

    def _predict_hw_add(train, horizon):
        return holt_winters_forecast(train, horizon, seasonal=12, seasonal_type='add')

    model_dict = {}
    model_dict['简单移动平均法'] = _predict_sma
    model_dict['加权移动平均法'] = _predict_weighted
    model_dict['二次指数平滑法'] = _predict_holt
    if sku_class not in ['AX', 'AY', 'BX']:
        model_dict['ARIMA(自预测)'] = _predict_arima
        model_dict['Holt-Winters乘法'] = _predict_hw_mul
        model_dict['Holt-Winters加法'] = _predict_hw_add

    model_rmse = {name: [] for name in model_dict}
    model_mape = {name: [] for name in model_dict}

    for start in range(max_start):
        train = history[start:start+window]
        actual = history[start+window:start+window+horizon]
        for name, predict_func in model_dict.items():
            try:
                preds = predict_func(train, horizon)
                if preds is None:
                    model_rmse[name].append(1e10)
                    model_mape[name].append(1e10)
                else:
                    rmse = np.sqrt(np.mean((np.array(preds) - np.array(actual))**2))
                    mape = calculate_mape(actual, preds)
                    model_rmse[name].append(rmse)
                    model_mape[name].append(mape)
            except Exception as e:
                print(f"模型 {name} 滚动回测失败: {e}")
                model_rmse[name].append(1e10)
                model_mape[name].append(1e10)

    results = {}
    for name in model_dict:
        if not model_rmse[name]:
            results[name] = {'rmse': np.inf, 'mape': np.inf}
        else:
            weights = np.arange(1, len(model_rmse[name])+1)
            rmse_avg = np.average(model_rmse[name], weights=weights)
            mape_avg = np.average(model_mape[name], weights=weights)
            results[name] = {'rmse': rmse_avg, 'mape': mape_avg}
    return results

def select_best_model(sku_combined_class, history, horizon):
    window = 24
    if len(history) < window + horizon:
        train = history[:-(horizon)] if len(history) > horizon else history
        test = history[-(horizon):] if len(history) > horizon else []
        if len(test) == 0:
            preds = simple_moving_average(history, horizon, window=3, recursive=False)
            return '简单移动平均法', preds, np.inf, np.inf, {'window': 3}, {'简单移动平均法': {'preds': preds, 'rmse': np.inf, 'mape': np.inf, 'params': 'N=3'}}
        
        candidates = {}
        all_models = {}
        
        best_sma_rmse = np.inf
        best_sma_mape = np.inf
        best_sma_preds = None
        best_window = 3
        min_window = max(3, len(train)//4) if sku_combined_class not in ['AX','AY','BX'] else max(6, len(train)//4)
        max_window = min(12, len(train))
        for w in range(min_window, max_window+1):
            preds = simple_moving_average(train, len(test), window=w, recursive=False)
            rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
            mape = calculate_mape(test, preds)
            if rmse < best_sma_rmse:
                best_sma_rmse = rmse
                best_sma_mape = mape
                best_sma_preds = preds
                best_window = w
        candidates['简单移动平均法'] = {'preds': best_sma_preds, 'rmse': best_sma_rmse, 'mape': best_sma_mape, 'model_info': {'window': best_window}}
        all_models['简单移动平均法'] = {'preds': best_sma_preds, 'rmse': best_sma_rmse, 'mape': best_sma_mape, 'params': f'N={best_window}'}
        
        best_wma_rmse = np.inf
        best_wma_mape = np.inf
        best_wma_preds = None
        best_wma_window = 3
        for w in range(min_window, max_window+1):
            preds = weighted_moving_average(train, len(test), window=w, recursive=False)
            rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
            mape = calculate_mape(test, preds)
            if rmse < best_wma_rmse:
                best_wma_rmse = rmse
                best_wma_mape = mape
                best_wma_preds = preds
                best_wma_window = w
        candidates['加权移动平均法'] = {'preds': best_wma_preds, 'rmse': best_wma_rmse, 'mape': best_wma_mape, 'model_info': {'window': best_wma_window}}
        all_models['加权移动平均法'] = {'preds': best_wma_preds, 'rmse': best_wma_rmse, 'mape': best_wma_mape, 'params': f'N={best_wma_window}'}
        
        preds, alpha = holt_exp_smoothing(train, len(test), sku_combined_class)
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
        mape = calculate_mape(test, preds)
        candidates['二次指数平滑法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'model_info': {'alpha': alpha}}
        all_models['二次指数平滑法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'params': f'α={alpha:.2f}'}
        
        if sku_combined_class not in ['AX', 'AY', 'BX']:
            try:
                preds, order = arima_forecast(train, len(test))
                rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
                mape = calculate_mape(test, preds)
                candidates['ARIMA(自预测)'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'model_info': {'order': order}}
                all_models['ARIMA(自预测)'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'params': f'({order[0]},{order[1]},{order[2]})'}
            except Exception as e:
                print(f"ARIMA 模型失败: {e}")
            
            if len(train) >= 24:
                try:
                    preds = holt_winters_forecast(train, len(test), seasonal=12, seasonal_type='mul')
                    if preds is not None:
                        rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
                        mape = calculate_mape(test, preds)
                        candidates['Holt-Winters乘法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'model_info': {}}
                        all_models['Holt-Winters乘法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'params': '季节周期12'}
                except Exception as e:
                    print(f"Holt-Winters乘法 模型失败: {e}")
                
                try:
                    preds = holt_winters_forecast(train, len(test), seasonal=12, seasonal_type='add')
                    if preds is not None:
                        rmse = np.sqrt(np.mean((np.array(preds) - np.array(test))**2))
                        mape = calculate_mape(test, preds)
                        candidates['Holt-Winters加法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'model_info': {}}
                        all_models['Holt-Winters加法'] = {'preds': preds, 'rmse': rmse, 'mape': mape, 'params': '季节周期12'}
                except Exception as e:
                    print(f"Holt-Winters加法 模型失败: {e}")
        
        if not candidates:
            preds = simple_moving_average(train, len(test), window=3, recursive=False)
            return '简单移动平均法', preds, np.inf, np.inf, {'window': 3}, {'简单移动平均法': {'preds': preds, 'rmse': np.inf, 'mape': np.inf, 'params': 'N=3'}}
        
        best_name = min(candidates, key=lambda x: candidates[x]['rmse'])
        best = candidates[best_name]
        return best_name, best['preds'], best['rmse'], best['mape'], best['model_info'], all_models
    else:
        rolling_results = rolling_multi_step_backtest(history, horizon, window=24, sku_class=sku_combined_class)
        if rolling_results is None or not rolling_results:
            return select_best_model(sku_combined_class, history, horizon)
        
        all_models = {}
        for model_name, metrics in rolling_results.items():
            if metrics['rmse'] >= 1e9:
                continue
            if model_name == '简单移动平均法':
                min_window = max(3, len(history)//4) if sku_combined_class not in ['AX','AY','BX'] else max(6, len(history)//4)
                max_window = min(12, len(history))
                best_win = 3
                best_rmse_fit = np.inf
                for w in range(min_window, max_window+1):
                    hist = list(history[:w])
                    errs = []
                    for t in range(w, len(history)):
                        pred = np.mean(hist[-w:])
                        errs.append((pred - history[t])**2)
                        hist.append(history[t])
                    rmse_fit = np.sqrt(np.mean(errs))
                    if rmse_fit < best_rmse_fit:
                        best_rmse_fit = rmse_fit
                        best_win = w
                all_models[model_name] = {
                    'preds': None,
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'params': f'N={best_win}'
                }
            elif model_name == '加权移动平均法':
                min_window = max(3, len(history)//4) if sku_combined_class not in ['AX','AY','BX'] else max(6, len(history)//4)
                max_window = min(12, len(history))
                best_win = 3
                best_rmse_fit = np.inf
                for w in range(min_window, max_window+1):
                    denominator = w * (w + 1) / 2
                    weights_old_to_new = np.array([(i + 1) / denominator for i in range(w)])
                    hist = list(history[:w])
                    errs = []
                    for t in range(w, len(history)):
                        recent = hist[-w:]
                        pred = np.dot(recent, weights_old_to_new)
                        errs.append((pred - history[t])**2)
                        hist.append(pred)
                    rmse_fit = np.sqrt(np.mean(errs))
                    if rmse_fit < best_rmse_fit:
                        best_rmse_fit = rmse_fit
                        best_win = w
                all_models[model_name] = {
                    'preds': None,
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'params': f'N={best_win}'
                }
            elif model_name == '二次指数平滑法':
                _, alpha = holt_exp_smoothing(history, horizon, sku_combined_class)
                all_models[model_name] = {
                    'preds': None,
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'params': f'α={alpha:.2f}'
                }
            elif model_name == 'ARIMA(自预测)':
                try:
                    _, order = arima_forecast(history, horizon)
                    all_models[model_name] = {
                        'preds': None,
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape'],
                        'params': f'({order[0]},{order[1]},{order[2]})'
                    }
                except:
                    all_models[model_name] = {
                        'preds': None,
                        'rmse': metrics['rmse'],
                        'mape': metrics['mape'],
                        'params': '自动'
                    }
            elif model_name in ['Holt-Winters乘法', 'Holt-Winters加法']:
                all_models[model_name] = {
                    'preds': None,
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape'],
                    'params': '季节周期12'
                }
        
        if not all_models:
            return select_best_model(sku_combined_class, history, horizon)
        
        best_name = min(rolling_results, key=lambda x: rolling_results[x]['rmse'] if rolling_results[x]['rmse'] < 1e9 else float('inf'))
        best_rmse = rolling_results[best_name]['rmse']
        best_mape = rolling_results[best_name]['mape']
        
        train = history
        if best_name == '简单移动平均法':
            min_window = max(3, len(train)//4) if sku_combined_class not in ['AX','AY','BX'] else max(6, len(train)//4)
            max_window = min(12, len(train))
            best_win = 3
            best_rmse_fit = np.inf
            for w in range(min_window, max_window+1):
                hist = list(train[:w])
                errs = []
                for t in range(w, len(train)):
                    pred = np.mean(hist[-w:])
                    errs.append((pred - train[t])**2)
                    hist.append(pred)
                rmse_fit = np.sqrt(np.mean(errs))
                if rmse_fit < best_rmse_fit:
                    best_rmse_fit = rmse_fit
                    best_win = w
            preds = simple_moving_average(train, horizon, window=best_win, recursive=True)
            model_info = {'window': best_win}
        elif best_name == '加权移动平均法':
            min_window = max(3, len(train)//4) if sku_combined_class not in ['AX','AY','BX'] else max(6, len(train)//4)
            max_window = min(12, len(train))
            best_win = 3
            best_rmse_fit = np.inf
            for w in range(min_window, max_window+1):
                denominator = w * (w + 1) / 2
                weights_old_to_new = np.array([(i + 1) / denominator for i in range(w)])
                hist = list(train[:w])
                errs = []
                for t in range(w, len(train)):
                    recent = hist[-w:]
                    pred = np.dot(recent, weights_old_to_new)
                    errs.append((pred - train[t])**2)
                    hist.append(pred)
                rmse_fit = np.sqrt(np.mean(errs))
                if rmse_fit < best_rmse_fit:
                    best_rmse_fit = rmse_fit
                    best_win = w
            preds = weighted_moving_average(train, horizon, window=best_win, recursive=True)
            model_info = {'window': best_win}
        elif best_name == '二次指数平滑法':
            preds, alpha = holt_exp_smoothing(train, horizon, sku_combined_class)
            model_info = {'alpha': alpha}
        elif best_name == 'ARIMA(自预测)':
            preds, order = arima_forecast(train, horizon)
            model_info = {'order': order}
        elif best_name == 'Holt-Winters乘法':
            preds = holt_winters_forecast(train, horizon, seasonal=12, seasonal_type='mul')
            if preds is None:
                preds, alpha = holt_exp_smoothing(train, horizon, sku_combined_class)
                best_name = '二次指数平滑法'
                model_info = {'alpha': alpha}
            else:
                model_info = {}
        elif best_name == 'Holt-Winters加法':
            preds = holt_winters_forecast(train, horizon, seasonal=12, seasonal_type='add')
            if preds is None:
                preds, alpha = holt_exp_smoothing(train, horizon, sku_combined_class)
                best_name = '二次指数平滑法'
                model_info = {'alpha': alpha}
            else:
                model_info = {}
        else:
            preds = simple_moving_average(train, horizon, window=3, recursive=True)
            best_name = '简单移动平均法'
            best_rmse = np.inf
            best_mape = np.inf
            model_info = {'window': 3}
        
        return best_name, preds, best_rmse, best_mape, model_info, all_models

def forecast_future(sku_combined_class, history, horizon, best_model_name, model_info):
    if best_model_name == '简单移动平均法':
        window = model_info.get('window', 3)
        preds = simple_moving_average(history, horizon, window=window, recursive=True)
        return preds
    elif best_model_name == '加权移动平均法':
        window = model_info.get('window', 3)
        preds = weighted_moving_average(history, horizon, window=window, recursive=True)
        return preds
    elif best_model_name == '二次指数平滑法':
        preds, _ = holt_exp_smoothing(history, horizon, sku_combined_class)
        return preds
    elif best_model_name == 'ARIMA(自预测)':
        preds, _ = arima_forecast(history, horizon)
        return preds
    elif best_model_name == 'Holt-Winters乘法':
        preds = holt_winters_forecast(history, horizon, seasonal=12, seasonal_type='mul')
        if preds is None:
            preds, _ = holt_exp_smoothing(history, horizon, sku_combined_class)
        return preds
    elif best_model_name == 'Holt-Winters加法':
        preds = holt_winters_forecast(history, horizon, seasonal=12, seasonal_type='add')
        if preds is None:
            preds, _ = holt_exp_smoothing(history, horizon, sku_combined_class)
        return preds
    else:
        preds = simple_moving_average(history, horizon, window=3, recursive=True)
        return preds

# ==================== 新增：二次移动平均法 ====================
def double_moving_average(train, horizon, N=3):
    """
    二次移动平均法（双移动平均法）
    适用于线性趋势的时间序列预测。
    train: 历史数据列表（按时间顺序）
    horizon: 预测步长
    N: 移动平均期数
    返回预测值列表
    """
    if len(train) < N:
        return [train[-1]] * horizon
    ma1 = [sum(train[i:i+N])/N for i in range(len(train)-N+1)]
    if len(ma1) < N:
        return [train[-1]] * horizon
    ma2 = [sum(ma1[i:i+N])/N for i in range(len(ma1)-N+1)]
    if not ma2:
        return [train[-1]] * horizon
    a_t = 2 * ma1[-1] - ma2[-1]
    b_t = 2/(N-1) * (ma1[-1] - ma2[-1]) if N > 1 else 0
    preds = [max(0, a_t + b_t * (k+1)) for k in range(horizon)]
    return preds
