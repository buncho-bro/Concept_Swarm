"""
【ファイル概要】
capture_ui.py

プロジェクトの一部として作成されたPythonスクリプト。
"""

from playwright.sync_api import sync_playwright
import time

def capture_ui():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        
        print("Navigating to local UI...")
        page.goto("http://127.0.0.1:8000")
        time.sleep(2)  # 初期ロード待ち
        
        # 待機状態のスクリーンショット
        page.screenshot(path="C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/ui_idle.png")
        print("Captured Idle state.")
        
        # タスクを入力して送信
        print("Simulating task execution...")
        page.fill("#user-input", "UIのアニメーションテスト")
        page.click("button")
        
        # 思考開始直後（雨と少女の潜行）
        time.sleep(1)
        page.screenshot(path="C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/ui_thinking_1.png")
        print("Captured Thinking state (Rain).")
        
        # 水面のアニメーション
        time.sleep(2)
        page.screenshot(path="C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/ui_thinking_2.png")
        print("Captured Thinking state (Surface Ripple).")
        
        # レスポンス完了待ち
        page.wait_for_selector(".msg:has-text('[投影完了')", timeout=30000)
        time.sleep(1)
        page.screenshot(path="C:/Users/lop3l/.gemini/antigravity/brain/126586e2-f977-4633-9d33-4d5f6ad332c5/ui_completed.png")
        print("Captured Completed state.")
        
        browser.close()

if __name__ == "__main__":
    capture_ui()
