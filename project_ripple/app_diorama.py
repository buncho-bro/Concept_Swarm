"""
【ファイル概要】
app_diorama.py

プロジェクトの一部として作成されたPythonスクリプト。
"""

import gradio as gr
from configurator.env_setter import EnvironmentConfigurator
from engine.physics_core import PhysicsCoreEngine
from tools.web_search import WebSearchTool
import json
import base64

# 画像のパス（先ほど生成した傘の少女の画像）
GIRL_IMAGE_PATH = r"C:\Users\lop3l\.gemini\antigravity\brain\126586e2-f977-4633-9d33-4d5f6ad332c5\pixel_girl_umbrella_1791363779023.jpg"

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        return ""

girl_b64 = get_base64_image(GIRL_IMAGE_PATH)

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
    if history is None: history = []
    
    if engine is None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ 物理エンジンが未初期化です。"})
        # 状態: 待機
        yield history, "idle"
        return

    history.append({"role": "user", "content": user_message})
    
    # --- 少女が箱庭の霧の中へ消えていく（思考開始フェーズ） ---
    yield history, "thinking"
    
    progress_msg = "### 🌧️ 雨: 情報収集中..."
    history.append({"role": "assistant", "content": progress_msg})
    yield history, "thinking"
    
    context = ""
    if use_search:
        search_results = search_tool.search(user_message)
        context = f"\n\n[External Context from Web]:\n{search_results}\n"
        
    progress_msg = "### 🌊 水面: 面的な波紋の干渉を計算中（可能性の抽出）..."
    history[-1]["content"] = progress_msg
    yield history, "thinking"
    
    config = configurator.analyze_task(user_message)
    terrain = config.get("terrain_injection", "None")
    
    progress_msg = f"### 🏔️ 水底: 概念地形 `{terrain}` へ投影中...\n\n---\n\n"
    history[-1]["content"] = progress_msg
    yield history, "thinking"
    
    full_prompt = f"Task: {user_message}{context}"
    final_output = engine.generate_with_physics(full_prompt, config)
    
    history[-1]["content"] = final_output
    # --- 少女が戻ってくる（待機フェーズ） ---
    yield history, "idle"

# --- 究極のピクセルアート・ジオラマUI CSS ---
custom_css = f"""
/* Gradio全体の背景を透明にして、後ろの箱庭を見せる */
body {{
    background-color: #111 !important;
    margin: 0;
    overflow: hidden;
}}

/* 箱庭コンテナ (3D空間) */
#diorama-container {{
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    z-index: -10;
    perspective: 1000px;
    background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    overflow: hidden;
}}

/* ピクセルアート風の雨 */
.pixel-rain {{
    position: absolute;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background-image: 
        linear-gradient(to bottom, transparent 0%, rgba(150, 200, 255, 0.3) 50%, transparent 100%);
    background-size: 8px 64px;
    image-rendering: pixelated;
    animation: rainDrop 0.6s linear infinite;
    z-index: 1;
}}
@keyframes rainDrop {{
    0% {{ background-position: 0px 0px, 32px -32px; }}
    100% {{ background-position: 0px 1000px, 32px 968px; }}
}}

/* 奥行きのある水面（面的な干渉） */
.water-surface {{
    position: absolute;
    bottom: -10%; left: -50%; width: 200%; height: 60%;
    background: radial-gradient(ellipse at center, rgba(0, 150, 255, 0.4) 0%, rgba(0, 50, 100, 0.8) 70%);
    transform: rotateX(75deg);
    transform-origin: top;
    box-shadow: inset 0 50px 100px rgba(0,0,0,0.8);
    animation: waveMotion 4s ease-in-out infinite alternate;
    z-index: 2;
}}
@keyframes waveMotion {{
    0% {{ transform: rotateX(75deg) translateY(0px); }}
    100% {{ transform: rotateX(75deg) translateY(20px); }}
}}

/* 奥にかかる霧（一定距離から先が見えなくなる） */
.fog-layer {{
    position: absolute;
    top: 40%; left: 0; width: 100vw; height: 60vh;
    background: linear-gradient(to bottom, rgba(26,26,46,1) 0%, rgba(26,26,46,0) 100%);
    z-index: 3;
    pointer-events: none;
}}

/* 傘の少女のキャラクターコンテナ */
#pixel-girl {{
    position: fixed;
    bottom: 20px; /* チャットウィンドウの縁の高さに合わせる */
    left: 20px;
    width: 200px;
    height: 200px;
    background-image: url('{girl_b64}');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center bottom;
    image-rendering: pixelated; /* ピクセルアートをくっきりさせる */
    z-index: 100;
    transition: all 2s cubic-bezier(0.4, 0, 0.2, 1);
    /* 背景色(青)を簡易的に透過するためのCSSフィルタトリック (完全ではないが雰囲気が出る) */
    mix-blend-mode: lighten;
}}

/* 少女が霧の中に消えていくアニメーションクラス */
.girl-thinking {{
    transform: scale(0.3) translateY(-100px) translateX(200px);
    opacity: 0;
    filter: blur(4px);
}}

/* チャットUIの調整（半透明のガラス効果） */
.gradio-container {{
    background: rgba(20, 20, 30, 0.6) !important;
    backdrop-filter: blur(8px);
    border: 2px solid rgba(100, 150, 255, 0.2);
    border-radius: 12px;
    box-shadow: 0 0 30px rgba(0, 150, 255, 0.1);
    margin-top: 20px !important;
}}
"""

html_diorama = """
<div id="diorama-container">
    <div class="pixel-rain"></div>
    <div class="water-surface"></div>
    <div class="fog-layer"></div>
</div>
<div id="pixel-girl"></div>
"""

phase_js = """
function(phase) {
    const girl = document.getElementById('pixel-girl');
    if(girl) {
        if(phase === 'thinking') {
            girl.classList.add('girl-thinking'); // 霧の中へ消える
        } else {
            girl.classList.remove('girl-thinking'); // 縁に戻ってくる
        }
    }
    return phase;
}
"""

with gr.Blocks(title="Project Ripple Diorama", css=custom_css) as demo:
    # カスタムHTMLで箱庭レイヤーと少女を注入
    gr.HTML(html_diorama)
    
    gr.Markdown("# 🌊 Project Ripple (Diorama View)")
    gr.Markdown("チャットウィンドウの奥に広がるピクセルアートの箱庭。送信すると、縁に座る少女が霧の中へデータ（波紋）を集めに潜っていきます。")
    
    with gr.Row():
        init_btn = gr.Button("1. Initialize Physics Engine", variant="secondary")
        init_status = gr.Markdown("Status: 🔴 Not Loaded")
    init_btn.click(initialize, inputs=[], outputs=[init_status])
    
    chatbot = gr.Chatbot(height=500, label="Ripple Topography Terminal")
    
    with gr.Row():
        use_search = gr.Checkbox(label="🌍 雨を降らせる (Web Search)", value=False)
        
    with gr.Row():
        msg_input = gr.Textbox(label="Message", placeholder="タスクを入力してEnter...", scale=4)
        send_btn = gr.Button("Send", variant="primary", scale=1)
        
    phase_state = gr.Textbox(visible=False, value="idle")
    phase_state.change(None, inputs=[phase_state], outputs=None, js=phase_js)
    
    msg_input.submit(
        run_ripple_chat, 
        inputs=[msg_input, chatbot, use_search], 
        outputs=[chatbot, phase_state]
    ).then(lambda: "", None, msg_input)
    
    send_btn.click(
        run_ripple_chat, 
        inputs=[msg_input, chatbot, use_search], 
        outputs=[chatbot, phase_state]
    ).then(lambda: "", None, msg_input)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870)
