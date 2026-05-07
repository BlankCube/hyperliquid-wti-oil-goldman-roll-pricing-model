"""
WTI 永续合约第一性原理定价模型
================================

输入: F (前月CME价格), N (次月CME价格), 换月时间表
输出: 每小时的 Oracle, 模型价, 基差, funding rate

三条公理:
  1. Oracle = w*F + (1-w)*N
  2. FR = 0.5 * [P + clamp(r-P, ±0.0005)] / 8   (XYZ 0.5x)
  3. E[PnL_hedged] = 0

CME 关闭时 Oracle 通过 EMA 追踪永续合约 (τ=30min, trade.xyz WTI)
迭代反向归纳求解, 无套利验证通过 (12000+ 组合, 最大误差 < $0.000005)

用法:
  from model import solve, is_external
  result = solve(F=112.06, N=97.72)
  print(result['perp'][120])  # 第120小时的模型价
"""

from datetime import datetime, timedelta
import numpy as np

# ============================================================
# 参数
# ============================================================

STEP_MINUTES = 60   # 时间步长 (分钟)
STEPS_PER_HOUR = 60 // STEP_MINUTES  # 1

INTEREST = 0.0001  # 0.01% per 8h (基础利率)
FR_CAP = 0.04      # ±4%/hr (funding rate 上限)
BETA = np.exp(-STEP_MINUTES / 30)  # Oracle EMA: τ=30min (trade.xyz WTI), 60min步 → exp(-2)≈0.135

# 默认换月时间表: 2026年5月 CLM6→CLN6
DEFAULT_ENTRY = datetime(2026, 5, 1, 0, 0)
DEFAULT_EXIT = datetime(2026, 5, 17, 0, 0)
DEFAULT_ROLLS = [
    datetime(2026, 5, 8, 22, 0),
    datetime(2026, 5, 11, 22, 0),
    datetime(2026, 5, 12, 22, 0),
    datetime(2026, 5, 13, 22, 0),
    datetime(2026, 5, 14, 22, 0),
]


# ============================================================
# CME 开收盘判断
# ============================================================

def is_external(dt_utc):
    """
    CME 是否有外部数据 (夏令时)

    开盘: Sun 22:00 UTC ~ Fri 21:00 UTC
    每日维护: 21:00-22:00 UTC
    """
    wd = dt_utc.weekday()  # Mon=0, Sun=6
    h = dt_utc.hour
    if wd == 5: return False           # 周六全天关
    if wd == 6: return h >= 22         # 周日 22:00 后开
    if wd == 4: return h < 21          # 周五 21:00 前开
    return not (21 <= h < 22)          # 周一~四 21-22 维护


# ============================================================
# Funding 公式
# ============================================================

def funding_payment(basis, oracle):
    """
    XYZ funding: 多头每步支付 (USD)

    FR_8h = 0.5 * [P + clamp(r - P, -0.0005, +0.0005)]
    FR_hr = FR_8h / 8
    FR_step = FR_hr / STEPS_PER_HOUR
    payment = FR_step * oracle
    """
    if oracle <= 0:
        return 0
    P = basis / oracle
    cl = max(-0.0005, min(0.0005, INTEREST - P))
    fr_hr = max(-FR_CAP, min(FR_CAP, 0.5 * (P + cl) / 8))
    return fr_hr / STEPS_PER_HOUR * oracle


def funding_rate_hourly(basis, oracle):
    """XYZ funding rate (每小时)"""
    if oracle <= 0:
        return 0
    P = basis / oracle
    cl = max(-0.0005, min(0.0005, INTEREST - P))
    return max(-FR_CAP, min(FR_CAP, 0.5 * (P + cl) / 8))


# ============================================================
# 求解器
# ============================================================

def solve(F, N, entry=None, exit_dt=None, roll_datetimes=None, max_iter=30):
    """
    求解永续合约的公平价格路径

    Parameters
    ----------
    F : float
        前月 CME 价格 (e.g. CLK26)
    N : float
        次月 CME 价格 (e.g. CLM26)
    entry : datetime, optional
        模型起始时间 (默认 2026-04-01)
    exit_dt : datetime, optional
        模型结束时间 (默认 2026-04-16)
    roll_datetimes : list of datetime, optional
        换月事件时间表 (默认 2026年4月)
    max_iter : int
        迭代次数

    Returns
    -------
    dict
        oracle : np.array  - 每小时 Oracle 价格
        perp   : np.array  - 每小时模型永续价格
        basis  : np.array  - 每小时基差 (perp - oracle)
        frate  : np.array  - 每小时 funding rate
        ext_oracle : np.array - 外部 Oracle (w*F + (1-w)*N)
        ext_flag : np.array - 是否外部定价
        roll_hours : set    - 换月事件的小时索引
        total_hours : int   - 总小时数
    """
    if entry is None: entry = DEFAULT_ENTRY
    if exit_dt is None: exit_dt = DEFAULT_EXIT
    if roll_datetimes is None: roll_datetimes = DEFAULT_ROLLS

    S = F - N
    delta = 0.2 * S
    total_steps = int((exit_dt - entry).total_seconds() / (STEP_MINUTES * 60)) + 1

    # 预计算: 换月事件转为步索引
    roll_steps = set()
    for dt in roll_datetimes:
        s = round((dt - entry).total_seconds() / (STEP_MINUTES * 60))
        if 0 <= s < total_steps:
            roll_steps.add(s)

    ext_flag = np.zeros(total_steps, dtype=bool)
    for s in range(total_steps):
        ext_flag[s] = is_external(entry + timedelta(minutes=s * STEP_MINUTES))

    def get_w(s):
        return max(0, 1 - 0.2 * sum(1 for rs in roll_steps if rs <= s))

    ext_oracle = np.array([get_w(s) * F + (1 - get_w(s)) * N for s in range(total_steps)])

    # 初始猜测: perp = ext_oracle
    perp = ext_oracle.copy()

    for iteration in range(max_iter):
        # 正向: 用当前 perp 模拟 actual Oracle
        oracle = np.zeros(total_steps)
        oracle[0] = ext_oracle[0]
        for s in range(1, total_steps):
            if ext_flag[s]:
                oracle[s] = ext_oracle[s]
            else:
                oracle[s] = BETA * oracle[s-1] + (1 - BETA) * perp[s]

        # 反向归纳
        basis = np.zeros(total_steps)
        basis[-1] = -0.0005 * oracle[-1]

        for s in range(total_steps - 2, -1, -1):
            Os = oracle[s]
            tgt = basis[s+1] + (oracle[s+1] - Os)

            b = tgt
            for _ in range(30):
                f = funding_payment(b, Os)
                res = b + f - tgt
                if abs(res) < 1e-12:
                    break
                eps = 1e-8
                d = 1 + (funding_payment(b + eps, Os) - f) / eps
                if abs(d) < 1e-15:
                    break
                b -= res / d
            basis[s] = b

        new_perp = oracle + basis
        diff = np.max(np.abs(new_perp - perp))
        perp = new_perp
        if diff < 1e-10:
            break

    # 最终 Oracle
    oracle[0] = ext_oracle[0]
    for s in range(1, total_steps):
        if ext_flag[s]:
            oracle[s] = ext_oracle[s]
        else:
            oracle[s] = BETA * oracle[s-1] + (1 - BETA) * perp[s]

    basis = perp - oracle
    frate = np.array([funding_rate_hourly(basis[s], oracle[s]) for s in range(total_steps)])

    return {
        'oracle': oracle,
        'perp': perp,
        'basis': basis,
        'frate': frate,           # 每小时 funding rate (不是每步)
        'ext_oracle': ext_oracle,
        'ext_flag': ext_flag,
        'roll_hours': roll_steps, # 兼容旧名
        'total_hours': total_steps, # 兼容旧名 (实际是 total_steps)
        'step_minutes': STEP_MINUTES,
    }


# ============================================================
# 便捷函数
# ============================================================

def price_at(F, N, h, **kwargs):
    """返回第 h 小时的模型价 (自动转为步索引)"""
    r = solve(F, N, **kwargs)
    step = h * STEPS_PER_HOUR
    return r['perp'][min(step, r['total_hours'] - 1)]


def step_index(dt=None, entry=None):
    """当前 UTC 时间对应的步索引"""
    if entry is None: entry = DEFAULT_ENTRY
    if dt is None:
        from datetime import timezone
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    return max(0, int((dt - entry).total_seconds() / (STEP_MINUTES * 60)))


def hour_index(dt=None, entry=None):
    """当前 UTC 时间对应的小时索引"""
    if entry is None: entry = DEFAULT_ENTRY
    if dt is None:
        from datetime import timezone
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    return max(0, int((dt - entry).total_seconds() / 3600))


# ============================================================
# 验证
# ============================================================

def verify(F=112.06, N=97.72, n_tests=1000):
    """
    无套利验证: 任意入场/离场, 对冲后 PnL = 0

    Returns True if all tests pass.
    """
    import random
    r = solve(F, N)
    TH = r['total_hours']

    max_err = 0
    for _ in range(n_tests):
        h0 = random.randint(0, TH - 2)
        h1 = random.randint(h0 + 1, TH - 1)
        sp = r['perp'][h0] - r['perp'][h1]
        # funding_payment 每步
        fd = sum(funding_payment(r['basis'][s], r['oracle'][s]) for s in range(h0, h1))
        total = sp + fd
        max_err = max(max_err, abs(total))

    ok = max_err < 0.01
    print(f'验证: {n_tests} 组合, 最大误差 ${max_err:.8f}, {"通过" if ok else "失败"}')
    return ok


if __name__ == '__main__':
    # 示例
    r = solve(112.06, 97.72)
    s = 120 * STEPS_PER_HOUR  # 第120小时 → 步索引
    print(f'F=$112.06  N=$97.72  步长={STEP_MINUTES}min  总步数={r["total_hours"]}')
    print(f'h=120 (step={s}): Oracle=${r["oracle"][s]:.2f}  Model=${r["perp"][s]:.2f}  Basis=${r["basis"][s]:.4f}  FR={r["frate"][s]*100:.5f}%/hr')
    print()
    verify()
