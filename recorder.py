"""
WTI 永续合约: 数据记录 + 定价模型 + API
==========================================

单进程运行:
  1. IBKR tick-by-tick streaming (CME 前月+次月期货)
  2. Hyperliquid orderbook websocket + funding REST polling
  3. 3秒对齐数据记录 → CSV
  4. Flask API → 前端可视化

python recorder.py
"""

import asyncio
import json
import csv
import os
import time
import math
import threading
from datetime import datetime, timezone, timedelta
import numpy as np
import requests
import websockets
import aiohttp
from dotenv import load_dotenv
from flask import Flask, jsonify, request as flask_req
from flask_cors import CORS
from ib_insync import IB, Future, util
from exchange_v3 import solve, is_external

load_dotenv()

# ============================================================
# 配置 — 修改这里以适应不同的换月周期
# ============================================================

IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7496'))
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '3'))

# Hyperliquid xyz:CL asset id
HL_ASSET_ID = 110029
HL_COIN = 'xyz:CL'

# 模型时间范围 (覆盖整个换月周期)
ENTRY = datetime(2026, 5, 1, 0, 0)
EXIT  = datetime(2026, 5, 19, 0, 0)

# Goldman Roll 5 个交易日 (CME 5:30PM ET → snap 到下一整点 22:00 UTC)
# Per docs.trade.xyz: BD 5–9 of the month (NOT BD 6–10).
# Previous schedule (RD1=5/8) was off by 1 day, under-pricing the basis
# correction by ~30 bp during the pre-roll window.
ROLLS = [
    datetime(2026, 5, 7,  22, 0),  # RD1 Thu (BD 5)
    datetime(2026, 5, 8,  22, 0),  # RD2 Fri (BD 6)
    datetime(2026, 5, 11, 22, 0),  # RD3 Mon (BD 7)
    datetime(2026, 5, 12, 22, 0),  # RD4 Tue (BD 8)
    datetime(2026, 5, 13, 22, 0),  # RD5 Wed (BD 9)
]

# Boros 到期日 (用于 implied APR 计算)
BOROS_EXPIRY = datetime(2026, 5, 20, 0, 0)

# CME 期货合约月份 (前月/次月)
CL_FRONT_MONTH = '202606'   # CLM26 = June 2026
CL_NEXT_MONTH  = '202607'   # CLN26 = July 2026

# CME 休市期间的初始价格 (启动后会被实时数据覆盖)
INIT_F_BID, INIT_F_ASK = 95.75, 95.81
INIT_N_BID, INIT_N_ASK = 89.10, 89.16

from exchange_v3 import STEP_MINUTES, STEPS_PER_HOUR
TH = int((EXIT - ENTRY).total_seconds() / (STEP_MINUTES * 60)) + 1
TICK_SEC = 3

CSV_FILE = 'live_log.csv'
CSV_HEADERS = [
    'timestamp_utc', 'ib_data_ts', 'hl_data_ts',
    'h', 'mode', 'w',
    'F_bid', 'F_ask', 'F_mid', 'N_bid', 'N_ask', 'N_mid',
    'S', 'S_pct',
    'oracle', 'model_price', 'model_price_1h',
    'model_basis', 'model_basis_pct', 'model_FR_hr', 'model_FR_apr',
    'hl_bid', 'hl_ask', 'hl_mid', 'hl_FR_hr', 'hl_FR_apr',
    'deviation', 'deviation_pct',
    'boros_avg_fr_apr',
]


# ============================================================
# 工具函数
# ============================================================

def now_utc():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def step_index():
    dt = now_utc()
    return max(0, min(int((dt - ENTRY).total_seconds() / (STEP_MINUTES * 60)), TH - 1))

def hour_index():
    dt = now_utc()
    return max(0, min(int((dt - ENTRY).total_seconds() / 3600), TH // STEPS_PER_HOUR))


# ============================================================
# 全局状态
# ============================================================

class G:
    F_bid = INIT_F_BID; F_ask = INIT_F_ASK
    N_bid = INIT_N_BID; N_ask = INIT_N_ASK
    hl_bid = hl_ask = None
    hl_fr = None
    ib_ts = 0.0
    hl_ts = 0.0

    model = None
    model_F = model_N = None
    _model_s = -1

    next_tick = 0
    latest = {}

    ib = None

g = G()


def log(msg):
    t = now_utc().strftime('%H:%M:%S')
    print(f'{t} {msg}')


# ============================================================
# 模型计算 (缓存: 价格变化 >0.005 或步索引变化时重算)
# ============================================================

def compute_model():
    if g.F_bid is None or g.N_bid is None:
        return None, None
    F = (g.F_bid + g.F_ask) / 2
    N = (g.N_bid + g.N_ask) / 2
    s = step_index()
    if (g.model is None
        or abs(F - (g.model_F or 0)) > 0.005
        or abs(N - (g.model_N or 0)) > 0.005
        or g._model_s != s):
        g.model = solve(F, N, ENTRY, EXIT, ROLLS)
        g.model_F, g.model_N = F, N
        g._model_s = s
    return g.model['perp'][s], g.model


# ============================================================
# CSV 记录
# ============================================================

csv_file = None
csv_writer = None

def init_csv():
    global csv_file, csv_writer
    exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
    if exists:
        with open(CSV_FILE, 'r') as f:
            first_line = f.readline().strip()
            if first_line != ','.join(CSV_HEADERS):
                try:
                    os.rename(CSV_FILE, CSV_FILE + f'.{int(time.time())}.bak')
                    exists = False
                except:
                    pass
    csv_file = open(CSV_FILE, 'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if not exists:
        csv_writer.writerow(CSV_HEADERS)
        csv_file.flush()


def try_record():
    now = time.time()
    if now < g.next_tick:
        return
    if g.F_bid is None or g.N_bid is None or g.hl_bid is None:
        return
    do_record()
    g.next_tick = math.ceil(now / TICK_SEC) * TICK_SEC + TICK_SEC


def do_record():
    t = now_utc()
    s = step_index()
    s1 = min(s + STEPS_PER_HOUR, TH - 1)
    h = hour_index()
    mode = 'EXT' if is_external(t) else 'INT'
    F = (g.F_bid + g.F_ask) / 2
    N = (g.N_bid + g.N_ask) / 2
    S = F - N
    hl_mid = (g.hl_bid + g.hl_ask) / 2

    model_price, m = compute_model()
    if m is None:
        return
    r = m
    wt = max(0, 1 - 0.2 * sum(1 for rh in r['roll_hours'] if rh <= s))
    mO, mP, mP1 = r['oracle'][s], r['perp'][s], r['perp'][s1]
    mb = r['basis'][s]
    mbp = mb / mO * 100 if mO > 0 else 0
    mfr = r['funding_rate'][s]
    mapr = mfr * 24 * 365 * 100
    hl_apr = g.hl_fr * 24 * 365 * 100 if g.hl_fr else None
    dev = hl_mid - mP
    devp = dev / mP * 100 if mP > 0 else None

    # Boros: 累计剩余 FR → 平均 APR
    hours_to_expiry = max(0, (BOROS_EXPIRY - t).total_seconds() / 3600)
    model_end_step = r['total_hours'] - 1
    cum_fr_model = sum(r['funding_rate'][ss] / STEPS_PER_HOUR for ss in range(s, model_end_step))
    STEADY_FR_HR = 0.000013
    model_end_time = ENTRY + timedelta(minutes=model_end_step * STEP_MINUTES)
    remaining_after = max(0, (BOROS_EXPIRY - model_end_time).total_seconds() / 3600)
    cum_fr_total = cum_fr_model + STEADY_FR_HR * remaining_after
    boros_avg_fr_apr = (cum_fr_total / hours_to_expiry * 24 * 365 * 100) if hours_to_expiry > 0 else 0

    row = [
        t.strftime('%Y-%m-%d %H:%M:%S'), f'{g.ib_ts:.3f}', f'{g.hl_ts:.3f}',
        h, mode, f'{wt:.2f}',
        f'{g.F_bid:.2f}', f'{g.F_ask:.2f}', f'{F:.2f}',
        f'{g.N_bid:.2f}', f'{g.N_ask:.2f}', f'{N:.2f}',
        f'{S:.2f}', f'{S/mO*100:.3f}' if mO > 0 else '',
        f'{mO:.2f}', f'{mP:.2f}', f'{mP1:.2f}',
        f'{mb:.4f}', f'{mbp:.3f}', f'{mfr*100:.6f}', f'{mapr:.1f}',
        f'{g.hl_bid:.2f}', f'{g.hl_ask:.2f}', f'{hl_mid:.2f}',
        f'{g.hl_fr*100:.6f}' if g.hl_fr else '', f'{hl_apr:.1f}' if hl_apr else '',
        f'{dev:.2f}', f'{devp:.3f}' if devp is not None else '',
        f'{boros_avg_fr_apr:.1f}',
    ]
    csv_writer.writerow(row)
    csv_file.flush()

    # Delta 对冲比例 (每5分钟更新一次)
    if not hasattr(g, '_delta_F') or time.time() - getattr(g, '_delta_ts', 0) > 300:
        bump = 0.01
        rFu = solve(F + bump, N, ENTRY, EXIT, ROLLS)
        rFd = solve(F - bump, N, ENTRY, EXIT, ROLLS)
        rNu = solve(F, N + bump, ENTRY, EXIT, ROLLS)
        rNd = solve(F, N - bump, ENTRY, EXIT, ROLLS)
        g._delta_F = (rFu['perp'][s] - rFd['perp'][s]) / (2 * bump)
        g._delta_N = (rNu['perp'][s] - rNd['perp'][s]) / (2 * bump)
        g._delta_ts = time.time()

    g.latest = {
        'F': F, 'N': N, 'S': S, 'S_pct': S/mO*100 if mO > 0 else 0,
        'F_bid': g.F_bid, 'F_ask': g.F_ask, 'N_bid': g.N_bid, 'N_ask': g.N_ask,
        'oracle': mO, 'model_price': mP, 'model_price_1h': mP1,
        'model_basis': mb, 'model_basis_pct': mbp,
        'model_FR_hr': mfr, 'model_FR_apr': mapr,
        'hl_bid': g.hl_bid, 'hl_ask': g.hl_ask, 'hl_mid': hl_mid,
        'hl_FR_hr': g.hl_fr, 'hl_FR_apr': hl_apr,
        'deviation': dev, 'deviation_pct': devp,
        'h': h, 'w': wt, 'mode': mode,
        'ib_data_ts': g.ib_ts, 'hl_data_ts': g.hl_ts,
        'server_ts': time.time(),
        'boros_hours_left': hours_to_expiry,
        'boros_cum_fr': cum_fr_total * 100,
        'boros_avg_fr_apr': boros_avg_fr_apr,
        'delta_F': g._delta_F,
        'delta_N': g._delta_N,
    }


# ============================================================
# 数据源回调
# ============================================================

def on_ib(which, bid, ask, data_time):
    if which == 'F':
        if bid and not np.isnan(bid) and bid > 0: g.F_bid = bid
        if ask and not np.isnan(ask) and ask > 0: g.F_ask = ask
    else:
        if bid and not np.isnan(bid) and bid > 0: g.N_bid = bid
        if ask and not np.isnan(ask) and ask > 0: g.N_ask = ask
    if data_time:
        g.ib_ts = data_time.timestamp() if hasattr(data_time, 'timestamp') else float(data_time)
    try_record()


def on_hl_book(bid, ask, ts_ms):
    g.hl_bid = bid
    g.hl_ask = ask
    g.hl_ts = ts_ms / 1000.0
    try_record()


# ============================================================
# Flask API
# ============================================================

app = Flask(__name__)
CORS(app)

@app.route('/api/prices')
def api_prices():
    return jsonify(g.latest if g.latest else {'F': None})

@app.route('/api/history')
def api_history():
    max_points = int(flask_req.args.get('n', 5000))
    if not os.path.exists(CSV_FILE):
        return jsonify([])
    keep = {'timestamp_utc','model_price','hl_mid','deviation_pct',
            'model_FR_apr','hl_FR_apr','boros_avg_fr_apr'}
    rows = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: r.get(k) for k in keep})
    # 服务端降采样
    if len(rows) > max_points:
        step = len(rows) // max_points
        rows = rows[::step]
    return jsonify(rows)


# ============================================================
# Async 任务
# ============================================================

async def hl_book_ws():
    uri = 'wss://api.hyperliquid.xyz/ws'
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20) as ws:
                await ws.send(json.dumps({
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": HL_COIN}
                }))
                log(f'[HL] orderbook ws 已连接 ({HL_COIN})')
                async for msg in ws:
                    d = json.loads(msg)
                    if d.get('channel') == 'l2Book':
                        levels = d['data']['levels']
                        if levels[0] and levels[1]:
                            on_hl_book(float(levels[0][0]['px']),
                                       float(levels[1][0]['px']),
                                       d['data']['time'])
        except Exception as e:
            log(f'[HL] book ws 断开: {e}')
            await asyncio.sleep(3)


async def hl_funding_poll():
    while True:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post('https://api.hyperliquid.xyz/info',
                    json={'type': 'fundingHistory', 'coin': HL_COIN,
                          'startTime': int(time.time() * 1000) - 86400000}) as r:
                    d = await r.json()
                    if d:
                        g.hl_fr = float(d[-1]['fundingRate'])
        except: pass
        await asyncio.sleep(900)


async def ibkr_stream():
    while True:
        try:
            g.ib = IB()
            await g.ib.connectAsync(IBKR_HOST, IBKR_PORT,
                                    clientId=IBKR_CLIENT_ID, readonly=True)
            log(f'[IBKR] 已连接 {IBKR_HOST}:{IBKR_PORT}')

            front = Future('CL', CL_FRONT_MONTH, 'NYMEX')
            back  = Future('CL', CL_NEXT_MONTH,  'NYMEX')
            g.ib.qualifyContracts(front, back)
            g.ib.reqMktData(front)
            g.ib.reqMktData(back)

            tf = g.ib.ticker(front)
            tb = g.ib.ticker(back)
            tf.updateEvent += lambda t: on_ib('F', t.bid, t.ask, t.time)
            tb.updateEvent += lambda t: on_ib('N', t.bid, t.ask, t.time)

            await asyncio.sleep(3)
            on_ib('F', tf.bid, tf.ask, tf.time)
            on_ib('N', tb.bid, tb.ask, tb.time)
            log(f'[IBKR] F={g.F_bid}/{g.F_ask} N={g.N_bid}/{g.N_ask}')

            while g.ib.isConnected():
                await asyncio.sleep(1)
            log('[IBKR] 连接断开, 30s后重连...')
        except Exception as e:
            log(f'[IBKR] 连接失败: {e}, 30s后重连...')
        await asyncio.sleep(30)


# ============================================================
# 入口
# ============================================================

async def main():
    print('=' * 60)
    print('  WTI 永续合约定价模型 — 数据记录器')
    print(f'  ENTRY={ENTRY.date()}  EXIT={EXIT.date()}')
    print(f'  CME: CL{CL_FRONT_MONTH} → CL{CL_NEXT_MONTH}')
    print(f'  HL:  {HL_COIN}')
    print(f'  API: http://localhost:5111')
    print('=' * 60)

    init_csv()
    g.next_tick = math.ceil(time.time() / TICK_SEC) * TICK_SEC

    # Flask
    t = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5111, debug=False, use_reloader=False),
        daemon=True)
    t.start()
    log('[API] Flask :5111')

    await asyncio.gather(
        ibkr_stream(),
        hl_book_ws(),
        hl_funding_poll(),
    )


if __name__ == '__main__':
    util.patchAsyncio()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\n停止')
