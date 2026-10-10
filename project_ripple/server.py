"""
【ファイル概要】
server.py

Project Rippleの中核となるWebサーバー。物理シミュレーションUIとバックエンドのAI推論エンジンをリアルタイム（WebSocket等）で結びつけます。
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import uuid
import asyncio
import os

from configurator.env_setter import EnvironmentConfigurator
from engine.physics_core import PhysicsCoreEngine
from tools.web_search import WebSearchTool

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = None
configurator = None
search_tool = WebSearchTool()
sessions = {}

# スキャン用のパス定義
HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
# もし過去の概念蒸留モデルを特定ディレクトリに置いている場合はここに追加
CUSTOM_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct"
]

class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_web_search: bool
    use_eval: bool

class ModelChangeRequest(BaseModel):
    model_id: str

def get_available_models():
    models = set(CUSTOM_MODELS)
    if os.path.exists(HF_CACHE_DIR):
        for item in os.listdir(HF_CACHE_DIR):
            if item.startswith("models--"):
                # "models--Org--Model" -> "Org/Model"
                model_name = item.replace("models--", "").replace("--", "/")
                models.add(model_name)
    
    # Concept Forge の LoRA出力ディレクトリもスキャン
    lora_dir = r"C:\Users\lop3l\.gemini\antigravity\scratch\concept_core\distillation_experiment"
    if os.path.exists(lora_dir):
        for item in os.listdir(lora_dir):
            if item.startswith("lora_") and os.path.isdir(os.path.join(lora_dir, item)):
                # 絶対パスとして追加（UIに表示される）
                models.add(os.path.join(lora_dir, item).replace("\\", "/"))
                
    return sorted(list(models))

@app.on_event("startup")
async def startup_event():
    global engine, configurator
    engine = PhysicsCoreEngine()
    # 初期モデルはローカルに確実に存在するであろう軽量モデル
    initial_model = "Qwen/Qwen2.5-1.5B-Instruct"
    engine.load_model(initial_model)
    configurator = EnvironmentConfigurator(engine.model, engine.tokenizer)
    print("Server Started. Default model loaded.")

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/models")
async def get_models():
    return {
        "current_model": engine.current_model_id if engine else None,
        "available_models": get_available_models()
    }

@app.post("/api/change_model")
async def change_model(req: ModelChangeRequest):
    global configurator
    success = engine.load_model(req.model_id)
    if success:
        # モデルが変わったのでConfiguratorも再初期化
        configurator = EnvironmentConfigurator(engine.model, engine.tokenizer)
        return {"status": "success", "model": req.model_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/api/sessions")
async def get_sessions():
    return [{"id": k, "name": v["name"]} for k, v in sessions.items()]

@app.get("/api/session/{session_id}")
async def get_session_history(session_id: str):
    if session_id in sessions:
        return sessions[session_id]["history"]
    return []

def run_self_evaluation(task, output, config):
    if configurator:
        configurator.evaluate_performance(task, output, config)

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    await asyncio.sleep(1)
    
    if req.session_id not in sessions:
        sessions[req.session_id] = {"name": req.message[:15] + "...", "history": []}
    
    session = sessions[req.session_id]
    session["history"].append({"role": "user", "content": req.message})

    context = ""
    if req.use_web_search:
        search_results = search_tool.search(req.message)
        context = f"\n\n[External Context]:\n{search_results}\n"
    
    config = configurator.analyze_task(req.message)
    full_prompt = f"Task: {req.message}{context}"
    final_output = engine.generate_with_physics(full_prompt, config)
    
    session["history"].append({"role": "ai", "content": final_output, "config": config})
    
    if req.use_eval:
        background_tasks.add_task(run_self_evaluation, req.message, final_output, config)
    
    return {
        "reply": final_output,
        "config": config,
        "session_name": session["name"]
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
