"""
【ファイル概要】
app.py

プロジェクトの一部として作成されたPythonスクリプト。
"""

import gradio as gr
from configurator.env_setter import EnvironmentConfigurator
from engine.physics_core import PhysicsCoreEngine
from tools.web_search import WebSearchTool

engine = None
configurator = None
search_tool = WebSearchTool()

def initialize():
    global engine, configurator
    if engine is None:
        engine = PhysicsCoreEngine()
        configurator = EnvironmentConfigurator(engine.model, engine.tokenizer)
    return "✅ Model loaded and ready."

def run_ripple_chat(user_message, history, use_search):
    if history is None:
        history = []
        
    if engine is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ 物理エンジンが未初期化です。"})
        yield history, "idle"
        return

    history.append({"role": "user", "content": user_message})
    
    # 1. 🌧️ 雨フェーズ
    progress_msg = "### 🌧️ 外部情報の収集中..."
    history.append({"role": "assistant", "content": progress_msg})
    yield history, "rain" # 背景を雨にするシグナル
    
    context = ""
    if use_search:
        search_results = search_tool.search(user_message)
        context = f"\n\n[External Context from Web]:\n{search_results}\n"
    
    # 2. 🌊 水面フェーズ
    progress_msg = "### 🌊 波の干渉を計算中 (粘性 / 減衰)..."
    history[-1]["content"] = progress_msg
    yield history, "surface" # 背景を水面にするシグナル
    
    config = configurator.analyze_task(user_message)
    terrain = config.get("terrain_injection", "None")
    
    # 3. 🏔️ 水底フェーズ
    progress_msg = f"### 🏔️ 概念地形 `{terrain}` へ投影中...\n\n---\n\n"
    history[-1]["content"] = progress_msg
    yield history, "topography" # 背景を水底にするシグナル
    
    full_prompt = f"Task: {user_message}{context}"
    final_output = engine.generate_with_physics(full_prompt, config)
    
    # 最終出力
    history[-1]["content"] = final_output
    yield history, "idle" # 待機状態に戻す

# 視覚効果をもたらすカスタムCSS
custom_css = """
/* 基本のトランジション */
.gradio-container {
    transition: background 1.5s ease, background-color 1.5s ease;
}

/* 1. 雨の背景エフェクト */
body.bg-rain {
    background-color: #1a1a2e !important;
    background-image: 
        linear-gradient(to bottom, rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0) 100%);
    background-size: 10px 100px;
    animation: rainDrop 0.5s linear infinite;
}
@keyframes rainDrop {
    0% { background-position: 0px 0px, 20px -50px; }
    100% { background-position: 0px 1000px, 20px 950px; }
}

/* 2. 水面（波紋）の背景エフェクト */
body.bg-surface {
    background-color: #0f3460 !important;
    background-image: radial-gradient(circle at center, rgba(0, 191, 255, 0.3) 0%, transparent 60%);
    background-size: 200% 200%;
    animation: rippleEffect 3s ease-out infinite;
}
@keyframes rippleEffect {
    0% { background-position: center; background-size: 50% 50%; opacity: 0.8; }
    100% { background-position: center; background-size: 300% 300%; opacity: 0; }
}

/* 3. 水底（地形への投影）の背景エフェクト */
body.bg-topography {
    background-color: #000000 !important;
    background-image: radial-gradient(ellipse at bottom, #16213e 0%, #000000 100%);
}

/* チャットバブル自体は少し透けさせて背景を見せる */
.message-wrap {
    background: rgba(30, 30, 30, 0.7) !important;
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255,255,255,0.1);
}
"""

# クライアントサイドのJavaScript（フェーズを受け取ってbodyのクラスを変える）
phase_js = """
function(phase) {
    document.body.classList.remove('bg-rain', 'bg-surface', 'bg-topography', 'bg-idle');
    if (phase && phase !== 'idle') {
        document.body.classList.add('bg-' + phase);
    }
    return phase;
}
"""

with gr.Blocks(title="Project Ripple Platform", css=custom_css) as demo:
    gr.Markdown("# 🌊 Project Ripple Platform")
    gr.Markdown("タスク送信後、チャットの背景が **[🌧️雨] → [🌊水面] → [🏔️水底]** と物理プロセスに合わせて変化します。")
    
    with gr.Row():
        init_btn = gr.Button("1. Initialize Physics Engine", variant="secondary")
        init_status = gr.Markdown("Status: 🔴 Not Loaded")
    init_btn.click(initialize, inputs=[], outputs=[init_status])
    
    chatbot = gr.Chatbot(height=600, label="Ripple Simulation Terminal")
    
    with gr.Row():
        use_search = gr.Checkbox(label="🌍 雨を降らせる (Web Search)", value=False)
        
    with gr.Row():
        msg_input = gr.Textbox(label="Message", placeholder="タスクを入力してEnter...", scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        
    # 裏でフェーズ状態を保持し、変更時にJSを発火させるための隠しコンポーネント
    phase_state = gr.Textbox(visible=False, value="idle")
    phase_state.change(None, inputs=[phase_state], outputs=None, js=phase_js)
    
    msg_input.submit(
        run_ripple_chat, 
        inputs=[msg_input, chatbot, use_search], 
        outputs=[chatbot, phase_state]
    ).then(
        lambda: "", None, msg_input
    )
    
    send_btn.click(
        run_ripple_chat, 
        inputs=[msg_input, chatbot, use_search], 
        outputs=[chatbot, phase_state]
    ).then(
        lambda: "", None, msg_input
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870)
