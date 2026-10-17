# EvoGraph-R1 (rút gọn) — mã tái lập

## Chạy trên Google Colab
Mở [`run_colab.ipynb`](https://colab.research.google.com/github/FluteInTheRain/evograph-reproduction/blob/main/run_colab.ipynb), chọn Runtime GPU T4 và chạy lần lượt các ô (Run All). Notebook clone repo, đăng nhập W&B (cần dán API key khi được hỏi), kiểm tra dữ liệu, khởi động LLM rồi chạy các thực nghiệm. Ô đầu có hai tham số: `BACKEND` (`"local"` hoặc `"gemini"`) và `N` (số câu hỏi, mặc định 100; đặt 5 để chạy thử nhanh).
- `local`: Qwen2.5-7B-Instruct-AWQ chạy bằng vLLM trên T4, không cần API key.
- `gemini`: cần khóa `GEMINI_API_KEY` trong Colab Secrets.

Mã chạy lại thực nghiệm trong báo cáo. Kết quả và mô tả giải pháp nằm trong báo cáo, không đặt ở đây.

## Môi trường
- Python 3.10+. `run.py` chỉ gọi LLM qua API tương thích OpenAI nên chính nó không cần GPU.
- Cấu hình đã dùng cho báo cáo: Google Colab, GPU T4, Qwen2.5-7B-Instruct-AWQ chạy bằng vLLM (cổng 8000). Mặc định của `run.py` trỏ tới máy chủ này (`--base_url http://localhost:8000/v1`).
- Chạy ngoài Colab: cần một máy chủ tương thích OpenAI ở địa chỉ trên, ví dụ trên máy có GPU:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq --dtype half --max-model-len 6144 --port 8000
```

  Dùng LLM khác thì đổi `--model`, `--base_url`, `--api_key_env` (biến môi trường chứa khóa; nếu không đặt thì dùng `EMPTY`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Dữ liệu
Hai tệp MetaQA (https://github.com/yuyuz/MetaQA) đã có sẵn trong `data/`; nguồn gốc:
- `kb.txt` (định dạng `head|relation|tail`)
- `2-hop/vanilla/qa_test.txt`

## Chạy
Mỗi lệnh ghi một run lên W&B (project `evograph-reproduction`).

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

Tham số khác: `--model`, `--base_url`, `--api_key_env`, `--max_tokens`, `--T`, `--k`, `--wrong`, `--web_cov`, `--seed` (xem `python run.py -h`).
