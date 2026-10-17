# EvoGraph-R1 (rút gọn) — mã tái lập

Mã chạy lại thực nghiệm trong báo cáo. Kết quả và mô tả giải pháp nằm trong báo cáo, không đặt ở đây.

## Môi trường
- Python 3.10+, không cần GPU (gọi LLM qua API tương thích OpenAI, mặc định Groq).
- Máy dùng để chạy: CPU thường, cần Internet.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...        # khóa API của nhà cung cấp LLM
wandb login
```

## Dữ liệu
Tải MetaQA (https://github.com/yuyuz/MetaQA), cần hai tệp:
- `kb.txt` (định dạng `head|relation|tail`)
- `2-hop/vanilla/qa_test.txt`

## Chạy
Mỗi lệnh ghi một run lên W&B (project `evograph-reproduction`). Nhiễu của KG được sinh theo seed nên giống nhau giữa các cấu hình.

```bash
# Bốn cấu hình chính (cùng seed 42, 100 câu)
for v in static web edit full; do
  python run.py --kb data/kb.txt --qa data/2hop_qa_test.txt --variant $v --n 100 --wandb_group main-ablation
done

# Ablation theo mức khuyết của KG (--drop) cho static và full
for d in 0.0 0.25 0.5 0.75; do
  for v in static full; do
    python run.py --kb data/kb.txt --qa data/2hop_qa_test.txt --variant $v --drop $d --wandb_group ablation-drop
  done
done

# Ablation theo độ nhiễu của web (--web_noise)
for p in 0.0 0.1 0.3 0.5; do
  python run.py --kb data/kb.txt --qa data/2hop_qa_test.txt --variant full --web_noise $p --wandb_group ablation-webnoise
done
```

Tham số khác: `--model`, `--base_url`, `--api_key_env`, `--T`, `--k`, `--wrong`, `--web_cov`, `--seed` (xem `python run.py -h`).
