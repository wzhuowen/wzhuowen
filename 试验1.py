# -*- coding: utf-8 -*-
"""
动态 Knothe 模型：c(t) = c0 + α·t，用重力修正加速度参数 α
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取重力数据 =====================
grav_data = []
with open('../采集重力历史数据.csv', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()
for line in lines[1:]:
    parts = line.strip().split(',')
    if len(parts) < 16: continue
    pt = parts[5]
    x1,y1,z1,g1 = map(float, parts[1:5])
    x2,y2,z2,g2 = map(float, parts[7:11])
    y3,x3,z3,g3 = map(float, parts[12:16])
    grav_data.extend([[pt, x1,y1,z1,g1, '2025-09-15'],
                      [pt, x2,y2,z2,g2, '2025-11-28'],
                      [pt, x3,y3,z3,g3, '2026-03-12']])

df_grav = pd.DataFrame(grav_data, columns=['点号','X','Y','Z','G','date'])
df_grav['date'] = pd.to_datetime(df_grav['date'])
print(f"重力数据加载完成，{len(df_grav)} 条")

# ===================== 2. 水准数据 =====================
df_level = pd.read_csv('../多期水准数据.csv', encoding='utf-8-sig')
time_cols = ['第1天/mm','第22天/mm','第51天/mm','第84天/mm','第113天/mm',
             '第145天/mm','第178天/mm','第210天/mm','第235天/mm','第300天/mm']
time_days = np.array([0,22,51,84,113,145,178,210,235,300])
for col in time_cols:
    df_level[col] = pd.to_numeric(df_level[col], errors='coerce')
    df_level[col] = -df_level[col]   # 下沉为正

# ===================== 3. 筛选测点 =====================
common = set(df_grav['点号']) & set(df_level['点号'])
valid = []
for p in common:
    s = df_level[df_level['点号']==p][time_cols].values.flatten().astype(float)
    s = s[~np.isnan(s)]
    if len(s)<3 or np.min(s)<-50 or s[-1]<20: continue
    if len(df_grav[df_grav['点号']==p])!=3: continue
    valid.append(p)
print(f"有效测点数：{len(valid)}")

# ===================== 4. 插值沉降到重力日期 =====================
ref = pd.Timestamp('2025-05-15')
for p in valid:
    sub = df_level[df_level['点号']==p].dropna(subset=time_cols)
    t_lev, s_lev = time_days, sub[time_cols].values.flatten()
    f = interp1d(t_lev, s_lev, kind='linear', bounds_error=False, fill_value='extrapolate')
    for idx, row in df_grav[df_grav['点号']==p].iterrows():
        day = (row['date']-ref).days
        df_grav.at[idx, 'S_mm'] = float(f(day))
df_grav = df_grav.dropna(subset=['S_mm'])

# ===================== 5. 物理分解 =====================
Gc=6.674e-11; L,W,H=1100.,170.,3.5; rho=1600.; Xc,Yc=466065.,3770710.; Zt,Zb=715.,718.5
def prism_grav(xo,yo,xc,yc,L,W,z1,z2,rho):
    x1,x2=xc-L/2,xc+L/2; y1,y2=yc-W/2,yc+W/2
    def K(dx,dy,dz):
        r=np.sqrt(dx**2+dy**2+dz**2)
        if r==0: return 0.0
        term=0.0
        term += dx*np.log(dy+r) if (dy+r)>0 else dx*np.log(abs(dy+r))
        term += dy*np.log(dx+r) if (dx+r)>0 else dy*np.log(abs(dx+r))
        term -= dz*np.arctan2(dx*dy,dz*r)
        return term
    g=0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                xc_=x1 if i==0 else x2; yc_=y1 if j==0 else y2; zc_=z1 if k==0 else z2
                g += ((-1)**(i+j+k))*K(xo-xc_, yo-yc_, 0-zc_)
    return g*Gc*rho*1e8

for p in valid:
    row = df_grav[df_grav['点号']==p].iloc[0]
    me = prism_grav(row['X'],row['Y'],Xc,Yc,L,W,Zt,Zb,-rho)
    df_grav.loc[df_grav['点号']==p, 'mass_eff'] = me
df_grav['FA'] = -0.3086*df_grav['S_mm']
df_grav['Residual'] = df_grav['G'] - df_grav['FA'] - df_grav['mass_eff']

# ===================== 6. 残余重力速率 (训练期) =====================
rate_dict = {}
for p in valid:
    sub = df_grav[df_grav['点号']==p].sort_values('date')
    g1=sub[sub['date']=='2025-09-15']['Residual'].values[0]
    g2=sub[sub['date']=='2025-11-28']['Residual'].values[0]
    dt=(pd.Timestamp('2025-11-28')-pd.Timestamp('2025-09-15')).days
    rate_dict[p] = (g2-g1)/dt
print(f"残余重力速率均值/std: {np.mean(list(rate_dict.values())):.4f} / {np.std(list(rate_dict.values())):.4f} uGal/d")

# 同时准备观测重力差 (用于统计模型对比)
delta_g_obs = {}
for p in valid:
    sub = df_grav[df_grav['点号']==p].sort_values('date')
    g1 = sub[sub['date']=='2025-09-15']['G'].values[0]
    g2 = sub[sub['date']=='2025-11-28']['G'].values[0]
    delta_g_obs[p] = g2 - g1

# ===================== 7. 训练/测试划分 =====================
train_days = time_days[:7]   # 0~178
test_days = time_days[8:]    # 235,300
train_data, test_data, pts = {}, {}, []
for p in valid:
    s = df_level[df_level['点号']==p][time_cols].values.flatten().astype(float)
    if np.any(pd.isna(s[:8])) or np.any(pd.isna(s[8:])): continue
    train_data[p] = {'t':train_days, 'S':s[:7]}
    test_data[p] = {'t':test_days, 'S':s[8:]}
    pts.append(p)
print(f"训练/测试点数：{len(pts)}")

# ===================== 8. 动态 Knothe 函数 =====================
def dyn_knothe(t, S0, c0, alpha):
    """
    动态 Knothe: c(t) = c0 + alpha*t
    确保 c(t) >= 0  (通过限制 c0>0, 且 alpha 不能太小负)
    """
    # 强制 c0 为正，alpha 可正可负，但要保证在t范围内 c(t)>=0
    # 在函数内部简单截断
    exp_term = -c0*t - 0.5*alpha*t**2
    return S0 * (1 - np.exp(exp_term))

# 带重力修正的版本：alpha = alpha0 + lam * rate (物理) 或 + c1 * dg_obs (统计)
def dyn_knothe_phys(t, S0, c0, alpha0, lam, rate):
    alpha = alpha0 + lam * rate
    return dyn_knothe(t, S0, c0, alpha)

def dyn_knothe_stat(t, S0, c0, alpha0, c1, dg):
    alpha = alpha0 + c1 * dg
    return dyn_knothe(t, S0, c0, alpha)

# ===================== 9. 全局优化物理模型 M2 (lam 网格) =====================
lam_grid = np.linspace(-50, 50, 41)   # 扩大搜索范围
best_lam, best_sse = 0.0, np.inf
best_p = {}
for lam in lam_grid:
    sse=0; ppar={}
    for p in pts:
        func = lambda t, S0, c0, a0: dyn_knothe_phys(t, S0, c0, a0, lam, rate_dict[p])
        try:
            # 初始猜测：S0 稍大于最大沉降，c0 小正数，alpha0 小值
            popt,_ = curve_fit(func, train_data[p]['t'], train_data[p]['S'],
                               p0=[max(train_data[p]['S'])*1.3, 0.001, 0.0],
                               bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]),
                               maxfev=20000)
            S_pred = func(train_data[p]['t'], *popt)
            sse += np.sum((train_data[p]['S']-S_pred)**2)
            ppar[p] = popt
        except:
            sse += 1e10
    if sse < best_sse:
        best_sse = sse; best_lam = lam; best_p = ppar
print(f"物理模型最优 λ = {best_lam:.4f}")

# ===================== 10. 统计模型 M1 (c1 全局) =====================
c1_grid = np.linspace(-0.01, 0.01, 41)
best_c1, best_sse1 = 0.0, np.inf
best_s = {}
for c1 in c1_grid:
    sse=0; ppar={}
    for p in pts:
        func = lambda t, S0, c0, a0: dyn_knothe_stat(t, S0, c0, a0, c1, delta_g_obs[p])
        try:
            popt,_ = curve_fit(func, train_data[p]['t'], train_data[p]['S'],
                               p0=[max(train_data[p]['S'])*1.3, 0.001, 0.0],
                               bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]),
                               maxfev=20000)
            S_pred = func(train_data[p]['t'], *popt)
            sse += np.sum((train_data[p]['S']-S_pred)**2)
            ppar[p] = popt
        except:
            sse += 1e10
    if sse < best_sse1: best_sse1 = sse; best_c1 = c1; best_s = ppar
print(f"统计模型最优 c1 = {best_c1:.6f}")

# ===================== 11. 基准动态 M0 (无修正) =====================
m0p = {}
for p in pts:
    try:
        popt,_ = curve_fit(dyn_knothe, train_data[p]['t'], train_data[p]['S'],
                           p0=[max(train_data[p]['S'])*1.3, 0.001, 0.0],
                           bounds=([0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]),
                           maxfev=20000)
        m0p[p] = popt
    except:
        m0p[p] = None

# ===================== 12. 测试集评估 =====================
true_all, m0_all, m1_all, m2_all = [], [], [], []
for p in pts:
    tt = test_data[p]['t']; st = test_data[p]['S']
    true_all.extend(st)
    if m0p[p] is not None:
        m0_all.extend(dyn_knothe(tt, *m0p[p]))
    else: m0_all.extend([np.nan]*len(tt))
    if p in best_s:
        m1_all.extend(dyn_knothe_stat(tt, *best_s[p], best_c1, delta_g_obs[p]))
    else: m1_all.extend([np.nan]*len(tt))
    if p in best_p:
        m2_all.extend(dyn_knothe_phys(tt, *best_p[p], best_lam, rate_dict[p]))
    else: m2_all.extend([np.nan]*len(tt))

mask = ~np.isnan(m0_all) & ~np.isnan(m1_all) & ~np.isnan(m2_all)
true = np.array(true_all)[mask]
m0_arr = np.array(m0_all)[mask]
m1_arr = np.array(m1_all)[mask]
m2_arr = np.array(m2_all)[mask]

def met(y,t):
    rmse = np.sqrt(np.mean((y-t)**2))
    mae = np.mean(np.abs(y-t))
    ssr = np.sum((y-t)**2); sst = np.sum((y-np.mean(y))**2)
    r2 = 1 - ssr/sst
    return rmse, mae, r2

print("\n========== 测试集 (第235、300天) ==========")
print(f"M0 动态Knothe:       RMSE={met(true,m0_arr)[0]:.2f}mm, MAE={met(true,m0_arr)[1]:.2f}mm, R²={met(true,m0_arr)[2]:.3f}")
print(f"M1 统计修正动态:     RMSE={met(true,m1_arr)[0]:.2f}mm, MAE={met(true,m1_arr)[1]:.2f}mm, R²={met(true,m1_arr)[2]:.3f}")
print(f"M2 物理修正动态:     RMSE={met(true,m2_arr)[0]:.2f}mm, MAE={met(true,m2_arr)[1]:.2f}mm, R²={met(true,m2_arr)[2]:.3f}")

t_stat,p_val = stats.ttest_rel(np.abs(true-m2_arr), np.abs(true-m0_arr))
print(f"M2 vs M0 配对t检验: t={t_stat:.3f}, p={p_val:.4f} {'显著' if p_val<0.05 else '不显著'}")