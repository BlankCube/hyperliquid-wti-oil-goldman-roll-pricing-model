"""
Hyperliquid WTI 永续合约交易所模拟器 v3
========================================

只有一个 Oracle, 一个基差, 一个 funding.

External (CME开盘): Oracle = w*F + (1-w)*N
Internal (CME关):   Oracle = EMA 追踪 perp (τ=30min)

basis = perp - oracle (唯一定义)
funding = f(basis, oracle) (永远一致)

反向归纳: 正向模拟 Oracle 路径 + 反向求 perp 价格
因为 internal 模式下 Oracle 依赖 perp (循环), 用迭代法收敛.
"""

from datetime import datetime, timedelta
import numpy as np


# ============================================================
# 参数
# ============================================================

STEP_MINUTES = 60
STEPS_PER_HOUR = 60 // STEP_MINUTES
INTEREST = 0.0001  # 0.01% per 8h
FR_CAP = 0.04
BETA = np.exp(-STEP_MINUTES / 30)  # EMA: τ=30min (trade.xyz WTI), 60min步 → exp(-2)≈0.135


def is_external(dt_utc):
    wd = dt_utc.weekday()
    h = dt_utc.hour
    if wd == 5: return False
    if wd == 6: return h >= 22
    if wd == 4: return h < 21
    return not (21 <= h < 22)


def funding_payment(basis, oracle):
    """多头每步支付 (USD), 步长=STEP_MINUTES"""
    if oracle <= 0: return 0
    P = basis / oracle
    cl = max(-0.0005, min(0.0005, INTEREST - P))
    fr_hr = max(-FR_CAP, min(FR_CAP, 0.5 * (P + cl) / 8))
    return fr_hr / STEPS_PER_HOUR * oracle


def funding_rate(basis, oracle):
    """每小时 funding rate"""
    if oracle <= 0: return 0
    P = basis / oracle
    cl = max(-0.0005, min(0.0005, INTEREST - P))
    return max(-FR_CAP, min(FR_CAP, 0.5 * (P + cl) / 8))


# ============================================================
# 求解
# ============================================================

def solve(F, N, entry, exit_dt, roll_datetimes, max_iter=30):
    """
    求解永续合约的公平价格路径

    返回每小时的: oracle, perp, basis, funding_rate
    所有量互相一致 (basis = perp - oracle, funding = f(basis))

    方法: 迭代
    1. 猜一个 perp 路径 (初始 = ext_oracle)
    2. 正向模拟 Oracle (external 用 CME, internal 用 EMA 追踪 perp)
    3. 有了 Oracle → 算 basis → 算 funding
    4. 反向归纳: 用 funding 求新的 perp 路径
    5. 重复直到收敛
    """
    S = F - N
    delta = 0.2 * S
    total_hours = int((exit_dt - entry).total_seconds() / (STEP_MINUTES * 60)) + 1

    # 预计算
    roll_hours = set()
    for dt in roll_datetimes:
        h = round((dt - entry).total_seconds() / (STEP_MINUTES * 60))
        if 0 <= h < total_hours:
            roll_hours.add(h)

    ext_flag = np.zeros(total_hours, dtype=bool)
    for h in range(total_hours):
        ext_flag[h] = is_external(entry + timedelta(minutes=h * STEP_MINUTES))

    def get_w(h):
        return max(0, 1 - 0.2 * sum(1 for rh in roll_hours if rh <= h))

    ext_oracle = np.array([get_w(h) * F + (1 - get_w(h)) * N for h in range(total_hours)])

    # 初始猜测: perp = ext_oracle
    perp = ext_oracle.copy()

    for iteration in range(max_iter):
        # --- 正向: 用当前 perp 路径模拟 Oracle ---
        oracle = np.zeros(total_hours)
        oracle[0] = ext_oracle[0] if ext_flag[0] else ext_oracle[0]

        for h in range(total_hours):
            if ext_flag[h]:
                oracle[h] = ext_oracle[h]
            else:
                prev = oracle[h - 1] if h > 0 else ext_oracle[0]
                oracle[h] = BETA * prev + (1 - BETA) * perp[h]

        # --- 算 basis 和 funding ---
        basis = perp - oracle
        fp = np.array([funding_payment(basis[h], oracle[h]) for h in range(total_hours)])

        # --- 反向归纳: 求新 perp ---
        # perp(h) = oracle(h) + b(h)
        # 无套利: (perp(h+1) - perp(h)) = funding_paid_by_long(h)
        # → perp(h) = perp(h+1) - fp(h)
        #
        # 但 oracle(h) 依赖 perp(h) (internal 模式), 所以这里
        # 我们固定 oracle (用本轮模拟结果), 只更新 basis:
        #
        # b(h) + funding(b(h), oracle(h)) = b_target
        # 其中 b_target = b(h+1), 换月时 b_target = b(h+1) - delta
        #
        # 牛顿法解 b(h), 然后 perp(h) = oracle(h) + b(h)

        new_b = np.zeros(total_hours)
        new_b[-1] = -0.0005 * oracle[-1]  # 稳态

        for h in range(total_hours - 2, -1, -1):
            O_h = oracle[h]
            # 通用公式 (无论 external/internal, 有无 roll):
            # b(h) + fp(b(h), O(h)) = b(h+1) + (O(h+1) - O(h))
            # 用 actual Oracle 变化量, 不硬编码 delta
            oracle_change = oracle[h + 1] - oracle[h]
            b_target = new_b[h + 1] + oracle_change

            # 牛顿法解 b
            b = b_target
            for _ in range(50):
                f = funding_payment(b, O_h)
                res = b + f - b_target
                if abs(res) < 1e-12: break
                eps = 1e-8
                d = 1 + (funding_payment(b + eps, O_h) - f) / eps
                if abs(d) < 1e-15: break
                b -= res / d
            new_b[h] = b

        new_perp = oracle + new_b

        # 收敛检查
        diff = np.max(np.abs(new_perp - perp))
        perp = new_perp

        if diff < 1e-10:
            break

    # 最终一致性: 重新正向算 Oracle
    for h in range(total_hours):
        if ext_flag[h]:
            oracle[h] = ext_oracle[h]
        else:
            prev = oracle[h - 1] if h > 0 else ext_oracle[0]
            oracle[h] = BETA * prev + (1 - BETA) * perp[h]

    basis = perp - oracle
    fr_arr = np.array([funding_rate(basis[h], oracle[h]) for h in range(total_hours)])
    fp_arr = np.array([funding_payment(basis[h], oracle[h]) for h in range(total_hours)])

    return {
        'total_hours': total_hours,
        'oracle': oracle,
        'perp': perp,
        'basis': basis,
        'ext_oracle': ext_oracle,
        'funding_rate': fr_arr,
        'funding_payment': fp_arr,
        'ext_flag': ext_flag,
        'roll_hours': roll_hours,
    }


# ============================================================
# 验证
# ============================================================

def test_pnl(result, h0, h1):
    """空永续+多N, PnL"""
    sp = result['perp'][h0] - result['perp'][h1]
    fd = sum(result['funding_payment'][h] for h in range(h0, h1))
    return sp + fd, sp, fd


def main():
    F, N = 112.06, 97.72
    entry = datetime(2026, 4, 25, 0, 0)
    exit_ = datetime(2026, 5, 19, 0, 0)
    rolls = [
        # docs.trade.xyz: BD 5–9 of the month (was BD 6–10, off by 1 day)
        datetime(2026, 5, 7, 22, 0),
        datetime(2026, 5, 8, 22, 0),
        datetime(2026, 5, 11, 22, 0),
        datetime(2026, 5, 12, 22, 0),
        datetime(2026, 5, 13, 22, 0),
    ]

    r = solve(F, N, entry, exit_, rolls)
    TH = r['total_hours']

    # 一致性检查: basis 符号 和 funding 符号 必须一致
    print('=' * 75)
    print('  一致性检查: basis > 0 时 funding > 0, basis < 0 时 funding < 0')
    print('=' * 75)
    inconsistent = 0
    for h in range(TH):
        b = r['basis'][h]
        fr = r['funding_rate'][h]
        # 当 b 足够大 (超过利率项), 符号应一致
        # 利率项使得 b=0 时 fr > 0, 所以 b 略负时 fr 仍可能 > 0
        # 真正不一致: b > $0.10 但 fr < 0, 或 b < -$0.10 但 fr > 0
        if (b > 0.10 and fr < 0) or (b < -0.10 and fr > 0):
            dt = entry + timedelta(hours=h)
            print(f'  ✗ h={h} {dt.strftime("%m/%d %H:00")} b=${b:.4f} fr={fr*100:.5f}%')
            inconsistent += 1
    if inconsistent == 0:
        print('  ✓ 全部一致')
    print()

    # 关键时刻
    roll_list = sorted(r['roll_hours'])
    print(f'{"h":>4} {"时间":<18} {"模式":<4} {"w":>4} {"Oracle":>8} {"Perp":>8} {"Basis":>8} {"b%":>7} {"FR/hr":>10}')
    print('-' * 75)

    key = sorted(set(
        [0] +
        [rh - 1 for rh in roll_list] + roll_list + [rh + 1 for rh in roll_list] +
        list(range(
            round((datetime(2026, 4, 11, 0, 0) - entry).total_seconds() / 3600),
            round((datetime(2026, 4, 13, 0, 0) - entry).total_seconds() / 3600), 8
        )) + [TH - 1]
    ))

    for h in key:
        if h < 0 or h >= TH: continue
        dt = entry + timedelta(hours=h)
        wd = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dt.weekday()]
        mode = 'EXT' if r['ext_flag'][h] else 'INT'
        rd = ' ←RD' if h in r['roll_hours'] else ''
        bpct = r['basis'][h] / r['oracle'][h] * 100 if r['oracle'][h] > 0 else 0
        print(f'{h:>4} {dt.strftime("%m/%d %H:00")} {wd:<4} {mode:<4} '
              f'{r["ext_oracle"][h]:>4.2f}'[-4:] + f' '
              f'${r["oracle"][h]:>7.2f} ${r["perp"][h]:>7.2f} '
              f'${r["basis"][h]:>+7.4f} {bpct:>+6.3f}% '
              f'{r["funding_rate"][h]*100:>+9.5f}%{rd}')

    # PnL 测试
    print()
    print('=' * 75)
    print('  PnL 验证')
    print('=' * 75)

    tests = [
        (0, TH - 1, '全程'),
        (0, roll_list[0], '入场→RD1'),
    ]
    for i in range(len(roll_list) - 1):
        tests.append((roll_list[i], roll_list[i + 1], f'RD{i+1}→RD{i+2}'))
    tests.append((roll_list[-1], TH - 1, 'RD5→离场'))

    # 周末
    sat = round((datetime(2026, 4, 11, 0, 0) - entry).total_seconds() / 3600)
    mon = round((datetime(2026, 4, 13, 0, 0) - entry).total_seconds() / 3600)
    tests.append((sat, mon, '周末'))
    tests.append((roll_list[2] - 2, roll_list[3] + 2, 'RD3前→RD4后'))

    # 随机
    import random
    random.seed(42)
    for _ in range(10):
        a = random.randint(0, TH - 2)
        b = random.randint(a + 1, TH - 1)
        tests.append((a, b, f'随机 {a}→{b}'))

    print(f'{"描述":<22} {"h0":>4}-{"h1":>4} {"时长":>5} {"PnL":>10}')
    print('-' * 55)
    all_ok = True
    for h0, h1, desc in tests:
        total, sp, fd = test_pnl(r, h0, h1)
        ok = '✓' if abs(total) < 0.01 else '✗'
        if abs(total) >= 0.01: all_ok = False
        print(f'{desc:<22} {h0:>4}-{h1:>4} {h1-h0:>4}h ${total:>+9.6f} {ok}')

    print()
    print('★ 全部通过!' if all_ok else '有未通过项')


if __name__ == '__main__':
    main()
