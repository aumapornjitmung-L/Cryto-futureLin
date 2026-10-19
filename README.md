# 📈 พยากรณ์ Crypto จากไฟล์ CSV — QUSDT, BRUSDT, GTCUSDT, PHAUSDT

แอป Streamlit ใช้ **2 โมเดลที่ดีที่สุด** (เลือกจากการเทียบ 5 โมเดลแบบ walk-forward — ผลอยู่ใน `model_config.json`)
พยากรณ์ทิศทาง (ขึ้น/ลง) และราคาปิดล่วงหน้า 1–7 วัน

## โครงสร้าง
```
app.py              แอป Streamlit
crypto_ml.py        โหลด CSV, ซ่อมจุดผิดปกติ, ฟีเจอร์, โมเดล, walk-forward
benchmark.py        เทียบโมเดลทั้งหมด -> เขียน model_config.json (2 โมเดลที่ดีที่สุด)
model_config.json   ผลเทียบโมเดลและ 2 โมเดลที่เลือก
data/               QUSDT.csv, BRUSDT.csv, GTCUSDT.csv, PHAUSDT.csv
```

## รันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
push ทั้งโฟลเดอร์ขึ้น GitHub แล้วไปที่ https://share.streamlit.io → New app → เลือก repo, branch `main`, Main file: `app.py`

## อัปเดตข้อมูล
- ชั่วคราว: อัปโหลด CSV ใหม่ในแถบด้านซ้ายของแอป
- ถาวร: แทนที่ไฟล์ใน `data/` (รูปแบบคอลัมน์: `event_date, close_price_usd, market_cap_usd, volume_usd`) แล้ว commit
  จากนั้นรัน `python benchmark.py` ใหม่เพื่อเลือก 2 โมเดลที่ดีที่สุดจากข้อมูลล่าสุด

> ไม่ใช่คำแนะนำการลงทุน
