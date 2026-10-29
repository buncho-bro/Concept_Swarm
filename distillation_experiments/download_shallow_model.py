import os
from huggingface_hub import snapshot_download

# ==============================================================================
# Ultra-Lightweight Model Downloader (For CPU-Only Shallow Extraction)
# 
# 巨大モデル（GLM-5.3等）の数百GBのウェイトを無視し、
# 概念蒸留（Shallow Mode）に必要な「初期ベクトル（Embeddings）」が含まれる
# 最初のウェイトファイルと、Tokenizer関連のファイルのみを部分ダウンロードします。
# ==============================================================================

def download_shallow_weights(repo_id: str, local_dir: str):
    print(f"🚀 Starting Partial Download for [{repo_id}]...")
    print(f"Target Directory: {local_dir}")
    print("-" * 50)
    
    os.makedirs(local_dir, exist_ok=True)
    
    # ダウンロードを許可するファイルのパターン（これ以外は無視されます）
    # ※ GLMアーキテクチャなどカスタムコード(trust_remote_code)が必要な場合があるため *.py も含めます。
    patterns = [
        "config.json",
        "configuration_*.py",
        "modeling_*.py",
        "tokenization_*.py",
        "tokenizer*",
        "model.safetensors.index.json",   # どのウェイトがどのファイルにあるかのマップ
        "model-00001-of-*.safetensors",   # 通常、Input Embeddingsはこの最初のファイルに入っています
        "pytorch_model-00001-of-*.bin"    # 古いフォーマット用
    ]
    
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=patterns,
            ignore_patterns=["*.msgpack", "*.h5", "model-0000[2-9]-of-*", "model-000[1-9][0-9]-of-*"] # 2番目以降の巨大ファイルを明示的に拒否
        )
        print("\n✅ Partial Download Complete!")
        print(f"Downloaded files are saved in: {download_path}")
        print("\n[NEXT STEP]")
        print("1. Launch concept_forge_webui.py")
        print(f"2. Enter '{local_dir}' into the [Teacher Model ID] field.")
        print("3. Check the [⚡ CPU-Only Shallow Extraction] box.")
        print("4. Start Distillation!")
        
    except Exception as e:
        print(f"\n❌ Error during download: {str(e)}")

if __name__ == "__main__":
    # ==========================================================================
    # 設定エリア：ダウンロードしたいモデルのリポジトリ名と保存先を指定してください。
    # ==========================================================================
    
    # 例: 巷で噂のGLM-5.3 (仮称のリポジトリ名なので、実際のものに書き換えてください)
    # TARGET_REPO = "THUDM/glm-4-9b-chat" 
    TARGET_REPO = "THUDM/glm-5.3-base" 
    
    # ローカルの保存先ディレクトリ
    SAVE_DIR = os.path.abspath(f"./models_shallow/{TARGET_REPO.split('/')[-1]}_shallow")
    
    print("Warning: Ensure the model repository name is correct.")
    print(f"Repository: {TARGET_REPO}")
    download_shallow_weights(TARGET_REPO, SAVE_DIR)
