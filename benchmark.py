"""เทียบโมเดลทั้งหมดแบบ walk-forward บนทุกเหรียญ แล้วเลือก "2 โมเดลที่ดีที่สุด" เขียนลง model_config.json
รันได้ทั้งในเครื่องและ Colab (อัปโหลด crypto_ml.py และโฟลเดอร์ data/ ก่อน):   python benchmark.py
รันซ้ำทุกครั้งที่เปลี่ยน/อัปเดตข้อมูล"""
import json
from datetime import date

from crypto_ml import COINS, MODEL_LABELS, benchmark

HORIZON = 1  # พยากรณ์ล่วงหน้า 1 วัน

res, summary = benchmark("data", COINS, HORIZON, repair=True)

print("\n=== ผลรายเหรียญ ===")
print(res.round(4).to_string(index=False))
print("\n=== สรุปเฉลี่ยทุกเหรียญ (เรียงตาม AUC) ===")
print(summary.round(4).to_string())

top2 = list(summary.index[:2])
print("\nโมเดลที่เลือก:", [MODEL_LABELS[m] for m in top2])

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "models": top2, "horizon": HORIZON, "coins": COINS, "created": date.today().isoformat(),
        "summary": summary.round(4).reset_index().to_dict("records"),
        "per_coin": res.round(4).to_dict("records"),
    }, f, ensure_ascii=False, indent=2)
