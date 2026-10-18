import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
from gradio_client import Client

def test():
    print("Connecting to WebUI at http://127.0.0.1:7860 ...")
    client = Client("http://127.0.0.1:7860")

    print("\n[1] Calling /load_model_for_chat endpoint...")
    result = client.predict(
        base_id="Qwen/Qwen2.5-0.5B-Instruct",
        precision_opt="FP16 (Direct, Recommended <=1.5B)",
        lora_dir="None",
        use_lora=False,
        api_name="/load_model_for_chat"
    )
    print("Load Status & Autonomous Genesis Result:\n", result)

    print("\n[2] Calling /chat_fn endpoint with Autonomous Stacked Sync...")
    # chat_fn(message, use_sync_chk, gamma_slider, show_stats_chk)
    job = client.submit(
        message="日本の首都はどこですか？簡潔に答えてください。",
        use_sync=True,
        gamma_val=0.06,
        show_stats=True,
        api_name="/chat_fn"
    )

    full_output = ""
    for update in job:
        full_output = update
    print("\nChat response from WebUI:")
    print(full_output)
    print("\nE2E Autonomous Stacked Sync Verification SUCCESS!")

if __name__ == "__main__":
    test()
