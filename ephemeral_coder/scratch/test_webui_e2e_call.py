import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import time
from gradio_client import Client

def test_webui_api():
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
    print("Load status result:", result)

    print("\n[2] Calling /chat_fn endpoint with Stacked Terrains Active...")
    job = client.submit(
        message="What is a mutex in one short sentence?",
        use_stacked_terrains=True,
        terrain_domain="Distributed Systems & CS (分散システム・並行処理)",
        gamma_val=0.08,
        use_deepthink=False,
        use_lightweight=False,
        api_name="/chat_fn"
    )

    full_output = ""
    for update in job:
        full_output = update
    print("\nChat response from WebUI:")
    print(full_output)
    print("\nE2E Verification SUCCESS!")

if __name__ == "__main__":
    test_webui_api()
