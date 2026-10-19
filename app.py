"""Streamlit: พยากรณ์ราคา/ทิศทาง QUSDT, BRUSDT, GTCUSDT, PHAUSDT จากไฟล์ CSV รายวัน ด้วย 2 โมเดลที่ดีที่สุด"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crypto_ml import COINS, MODEL_LABELS, DataError, forecast, prepare

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FALLBACK_MODELS = ["extra_trees", "random_forest"]

st.set_page_config(page_title="พยากรณ์ Crypto จาก CSV", page_icon="📈", layout="wide")


def load_config() -> dict:
    p = ROOT / "model_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"models": FALLBACK_MODELS}


CONFIG = load_config()
MODELS = CONFIG["models"][:2]


@st.cache_data(show_spinner=False)
def cached_prepare_file(path: str, mtime: float, repair: bool):
    return prepare(path, repair)


@st.cache_data(show_spinner=False)
def cached_prepare_bytes(data: bytes, repair: bool):
    return prepare(io.BytesIO(data), repair)


@st.cache_data(show_spinner=False)
def cached_forecast(df, models, horizon):
    return forecast(df, list(models), horizon)


def fmt_price(x: float) -> str:
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:,.5f}".rstrip("0").rstrip(".")


def signal(p: float) -> tuple[str, str]:
    if p >= 0.55:
        return "แนวโน้มขึ้น 🟢", "#2e9e5b"
    if p <= 0.45:
        return "แนวโน้มลง 🔴", "#d64545"
    return "ไม่ชัดเจน ⚪", "#888888"


def card(title, prob, ret, price) -> str:
    label, color = signal(prob)
    return (f"<div style='border:1px solid {color};border-left:8px solid {color};border-radius:8px;padding:12px 16px;'>"
            f"<div style='font-size:.85rem;opacity:.75'>{title}</div>"
            f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{label}</div>"
            f"<div style='font-size:.9rem'>โอกาสขึ้น {prob*100:.1f}%<br>"
            f"คาดการณ์เปลี่ยนแปลง {ret*100:+.2f}% → ${fmt_price(price)}</div></div>")


# ------------------------------------------------------------ Sidebar
st.title("📈 พยากรณ์ Crypto ด้วย Machine Learning (จากไฟล์ CSV)")
with st.sidebar:
    st.header("ตั้งค่า")
    symbol = st.selectbox("เหรียญ", COINS)
    horizon = st.slider("พยากรณ์ล่วงหน้า (วัน)", 1, 7, 1)
    repair = st.checkbox("ซ่อมจุดราคาผิดปกติ (กระโดดแล้วกลับ)", value=True,
                         help="ตรวจจับ bad tick ที่ราคากระโดดแล้วกลับระดับเดิมภายใน 1–4 วัน ดูรายการจุดที่แก้ได้ในแท็บ 'คุณภาพข้อมูล'")
    upload = st.file_uploader(f"อัปเดตข้อมูล {symbol} (CSV ใหม่)", type="csv",
                              help="คอลัมน์: event_date, close_price_usd, market_cap_usd, volume_usd (รูปแบบเดียวกับไฟล์ตั้งต้น)")
    st.divider()
    st.caption("โมเดลที่ใช้: " + " + ".join(MODEL_LABELS[m] for m in MODELS))
    if "created" not in CONFIG:
        st.warning("ยังไม่พบ `model_config.json` จึงใช้โมเดลเริ่มต้น — รัน `python benchmark.py`")

# ------------------------------------------------------------ Data
path = DATA_DIR / f"{symbol}.csv"
try:
    if upload is not None:
        df, info = cached_prepare_bytes(upload.getvalue(), repair)
        source = f"ไฟล์ที่อัปโหลด ({upload.name})"
    elif path.exists():
        df, info = cached_prepare_file(str(path), path.stat().st_mtime, repair)
        source = f"data/{path.name}"
    else:
        st.error(f"ไม่พบไฟล์ data/{symbol}.csv — อัปโหลด CSV ที่แถบด้านซ้าย")
        st.stop()
    with st.spinner("กำลังฝึกโมเดลและประเมินแบบ walk-forward..."):
        result = cached_forecast(df, tuple(MODELS), horizon)
except DataError as e:
    st.error(f"ใช้ข้อมูล {symbol} ไม่ได้: {e}")
    st.stop()

last = df.iloc[-1]
target_date = last["time"] + pd.Timedelta(days=horizon)
sigma = float(np.log(df["close"]).diff().tail(30).std())
age_days = (pd.Timestamp.now("UTC").tz_localize(None).normalize() - last["time"].normalize()).days

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"ราคาปิดล่าสุด {symbol} (USD)", fmt_price(last["close_raw"]))
c2.metric("วันที่ล่าสุดในข้อมูล", f"{last['time']:%d/%m/%Y}")
c3.metric("จำนวนวัน", f"{len(df):,}")
c4.metric("แหล่งข้อมูล", source)
if age_days > 2:
    st.warning(f"ข้อมูลล่าสุดคือ {last['time']:%d/%m/%Y} (เก่ากว่า {age_days} วัน) — ผลพยากรณ์นับจากวันนั้น "
               "อัปโหลด CSV ใหม่ที่แถบด้านซ้ายเพื่ออัปเดต")

tab_fc, tab_perf, tab_dq, tab_about = st.tabs(["ผลพยากรณ์", "ประสิทธิภาพโมเดล", "คุณภาพข้อมูล", "ข้อจำกัด"])

with tab_fc:
    st.subheader(f"พยากรณ์ราคาปิดวันที่ {target_date:%d/%m/%Y} (+{horizon} วัน)")
    avg_p = np.mean([r["prob_up"] for r in result.values()])
    avg_r = np.mean([r["pred_return"] for r in result.values()])
    avg_price = float(last["close"]) * float(np.exp(avg_r))
    cols = st.columns(3)
    for col, (name, r) in zip(cols, result.items()):
        col.markdown(card(MODEL_LABELS[name], r["prob_up"], r["pred_return"], r["pred_price"]), unsafe_allow_html=True)
    cols[2].markdown(card("ค่าเฉลี่ย 2 โมเดล", avg_p, avg_r, avg_price), unsafe_allow_html=True)

    band = sigma * np.sqrt(horizon)
    lo, hi = avg_price * np.exp(-band), avg_price * np.exp(band)
    st.info(f"ความผันผวนปกติของเหรียญนี้ (30 วันล่าสุด) ทำให้ช่วง ±1σ ของวันที่ {target_date:%d/%m} อยู่ที่ประมาณ "
            f"${fmt_price(lo)} – ${fmt_price(hi)} ซึ่งกว้างกว่าการเปลี่ยนแปลงที่โมเดลคาดมาก")
    ups = [r["prob_up"] > 0.5 for r in result.values()]
    (st.success if all(ups) or not any(ups) else st.warning)(
        "ทั้ง 2 โมเดลเห็นทิศทางตรงกัน" if all(ups) or not any(ups) else "2 โมเดลเห็นทิศทางต่างกัน — สัญญาณไม่ชัดเจน")

    log_axis = st.checkbox("แกนราคาแบบ log", value=False)
    view = df.tail(180)
    fig = go.Figure(go.Scatter(x=view["time"], y=view["close"], name="ราคาปิด", line=dict(color="#4c78a8")))
    spikes = view[view["repaired"]]
    if len(spikes):
        fig.add_trace(go.Scatter(x=spikes["time"], y=spikes["close_raw"], mode="markers", name="จุดผิดปกติที่ซ่อม",
                                 marker=dict(symbol="x", color="#d64545", size=8)))
    for color, (name, r) in zip(["#f58518", "#54a24b"], result.items()):
        fig.add_trace(go.Scatter(x=[last["time"], target_date], y=[last["close"], r["pred_price"]],
                                 mode="lines+markers", name=MODEL_LABELS[name], line=dict(color=color, dash="dash")))
    fig.add_trace(go.Scatter(x=[target_date], y=[avg_price], mode="markers", name="ค่าเฉลี่ย ±1σ",
                             marker=dict(symbol="diamond", size=10, color="#333"),
                             error_y=dict(type="data", symmetric=False, array=[hi - avg_price], arrayminus=[avg_price - lo])))
    fig.update_layout(height=480, margin=dict(t=20), yaxis_title="USD", legend=dict(orientation="h", y=1.1))
    if log_axis:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, width="stretch")

with tab_perf:
    st.markdown(f"**{symbol}** — ประเมินแบบ walk-forward (ฝึกด้วยข้อมูล 50% แรก แล้วทดสอบ 4 ช่วงถัดไปตามเวลา เว้นช่วงกันข้อมูลรั่ว) "
                f"พยากรณ์ล่วงหน้า {horizon} วัน")
    rows = []
    for name, r in result.items():
        m = r["metrics"]
        rows.append({"โมเดล": MODEL_LABELS[name], "AUC (ทิศทาง)": m["auc"], "Accuracy": m["acc"],
                     "Baseline Acc": m["acc_baseline"], "Hit-rate (ถดถอย)": m["reg_hit"],
                     "MAE เทียบเดา 'ไม่เปลี่ยน'": m["mae_ratio"], "จำนวนตัวอย่างทดสอบ": m["n_test"]})
    st.dataframe(pd.DataFrame(rows).set_index("โมเดล").round(4), width="stretch")
    st.caption("AUC = 0.50 คือเดาสุ่ม • MAE เทียบเดา < 1 = ดีกว่าการเดาว่าราคาไม่เปลี่ยน")

    if "summary" in CONFIG:
        st.markdown(f"**ผลเทียบ 5 โมเดลบนทั้ง 4 เหรียญ** (พยากรณ์ 1 วัน, รันเมื่อ {CONFIG.get('created', '-')}) — "
                    "เลือก 2 อันดับแรกตาม AUC เฉลี่ย")
        sm = pd.DataFrame(CONFIG["summary"])
        sm["model"] = sm["model"].map(lambda m: ("✅ " if m in MODELS else "") + MODEL_LABELS.get(m, m))
        st.dataframe(sm.rename(columns={"model": "โมเดล", "auc": "AUC เฉลี่ย", "acc": "Accuracy",
                                        "acc_baseline": "Baseline Acc", "reg_hit": "Hit-rate", "mae_ratio": "MAE เทียบเดา"})
                       .set_index("โมเดล"), width="stretch")
        if "per_coin" in CONFIG:
            pc = pd.DataFrame(CONFIG["per_coin"]).pivot(index="symbol", columns="model", values="auc")
            pc.columns = [MODEL_LABELS.get(c, c) for c in pc.columns]
            st.markdown("**AUC แยกรายเหรียญ**")
            st.dataframe(pc.round(3), width="stretch")
        st.caption("ความต่างระหว่างโมเดลอันดับต้น ๆ น้อยมากเมื่อเทียบกับความแปรปรวนของข้อมูล "
                   "การเลือก 2 อันดับแรกจึงไม่ได้แปลว่าชนะโมเดลอื่นอย่างมีนัยสำคัญ")

with tab_dq:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("แถวในไฟล์", f"{info['rows_raw']:,}")
    d2.metric("ตัดแถวไม่สมบูรณ์", info["dropped_incomplete"], help="วันล่าสุดที่ราคาปิดยังว่าง")
    d3.metric("วันที่ขาดหาย", info["date_gaps"])
    d4.metric("จุดที่ซ่อม", info["n_repaired"])
    rep = df[df["repaired"]][["time", "close_raw", "close"]].rename(
        columns={"time": "วันที่", "close_raw": "ราคาเดิม", "close": "ราคาหลังซ่อม"})
    if len(rep):
        st.markdown("**จุดที่ถูกซ่อม** (ราคากระโดดแล้วกลับระดับเดิม แทนด้วยค่าประมาณระหว่างจุดข้างเคียง)")
        st.dataframe(rep, hide_index=True, width="stretch")
    else:
        st.caption("ไม่มีจุดที่ถูกซ่อม" if repair else "ปิดการซ่อมจุดผิดปกติอยู่")
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=df["time"], y=df["close_raw"], name="ราคาเดิม", line=dict(color="#aaa")))
    f2.add_trace(go.Scatter(x=df["time"], y=df["close"], name="หลังซ่อม", line=dict(color="#4c78a8")))
    f2.update_yaxes(type="log")
    f2.update_layout(height=380, margin=dict(t=20), yaxis_title="USD (log)", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(f2, width="stretch")

with tab_about:
    st.markdown("""
- ผลลัพธ์เป็นการประมาณทางสถิติ **ไม่ใช่คำแนะนำการลงทุน** และไม่รับประกันผลตอบแทน
- ข้อมูลเป็นราคา USD รายวันแบบรวมศูนย์ (รูปแบบ CoinGecko) ไม่ใช่ราคาของสัญญา Perpetual บน Binance Futures โดยตรง
  และไม่มีราคาสูง/ต่ำรายวัน จึงใช้ได้เฉพาะราคาปิดและปริมาณซื้อขาย
- โมเดลไม่รู้ข่าว สภาวะตลาดรวม (เช่น BTC) หรือเหตุการณ์ของโครงการ
- ความได้เปรียบเหนือการเดาอยู่ในระดับเล็ก (AUC เฉลี่ยราว 0.55) และบางเหรียญใกล้ 0.50 ผลต่างระหว่างโมเดลอันดับต้น ๆ อยู่ในระดับสัญญาณรบกวน
- เหรียญที่เพิ่งเปิดเทรด (เช่น Q, BR) มีข้อมูลน้อย ผลประเมินจึงไม่เสถียร
- ขั้นตอนซ่อมจุดผิดปกติเป็นดุลยพินิจ อาจแก้ผิดจุดได้ ดูรายการในแท็บ "คุณภาพข้อมูล" หรือปิดตัวเลือกนี้ได้
- ราคาที่ซ่อมแล้วไม่ใช่ราคาที่ซื้อขายได้จริง ผลประเมินจึงเป็นของข้อมูลหลังทำความสะอาด
""")
