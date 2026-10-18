import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from gradio_client import Client

client = Client("http://127.0.0.1:7860")
job = client.submit(
    message="分散データベースにおけるデッドロック検出アルゴリズムについて、Wait-For Graph（待機グラフ）の観点から段階的に解説してください。",
    use_cascaded=True,
    gamma_val=0.05,
    show_trace=True,
    api_name="/chat_fn"
)

full_output = ""
for update in job:
    full_output = update
print("\n[Complex Reasoning Response]:")
print(full_output)
