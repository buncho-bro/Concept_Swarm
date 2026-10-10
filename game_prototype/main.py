import time
from latent_ai import LatentConsulAI

def clear_screen():
    print("\n" * 5)

def type_print(text, delay=0.03):
    for c in text:
        print(c, end='', flush=True)
        time.sleep(delay)
    print()

def main():
    clear_screen()
    print("==================================================")
    print("      代理戦争・国家指導者アカデミー (Prototype)  ")
    print("        ~ Powered by Latent Concept Swarm ~       ")
    print("==================================================\n")
    
    # 1. Player Nation Setup
    print("【国家作成フェーズ】")
    country_name = input("国家名を入力してください: ") or "アルカディア帝国"
    consul_name = input("執政官（AI）の名前を入力してください: ") or "レオン"
    doctrine = input("国家ドクトリン（40文字以内）: ") or "力こそ全て。全領土を支配せよ。"
    
    print("\n【執政官の性格パラメータ設定 (-1.0 〜 1.0)】")
    try:
        aggressive = float(input("好戦性 (-1.0:防衛/慎重 〜 1.0:戦争狂) [1.0]: ") or "1.0")
        rational = float(input("合理性 (-1.0:感情的/反発 〜 1.0:論理的/服従) [-0.5]: ") or "-0.5")
        coop = float(input("協調性 (-1.0:孤立主義 〜 1.0:同盟重視) [-0.8]: ") or "-0.8")
    except ValueError:
        aggressive, rational, coop = 1.0, -0.5, -0.8
        
    consul = LatentConsulAI(consul_name, doctrine, aggressive, rational, coop)
    
    # Dummy board state
    board_state = {
        "turn": 1,
        "resources": {"Food": 10, "Gold": 5, "Prod": 5, "Prestige": 0},
        "enemies": ["青の連邦 (北)", "赤の共和国 (東)"]
    }
    
    type_print(f"\n>>> 設定完了。あなたの執政官「{consul_name}」が着任しました。")
    time.sleep(1)
    
    # 2. Game Loop (1 Turn for Prototype)
    clear_screen()
    print(f"=== ターン {board_state['turn']} ===")
    print(f"【自国リソース】 食糧:{board_state['resources']['Food']} | 資金:{board_state['resources']['Gold']} | 生産:{board_state['resources']['Prod']}")
    print(f"【周辺状況】 北には軍拡中の「青の連邦」、東には豊かな「赤の共和国」がいます。")
    print("-" * 50)
    
    # Phase 2: AI Situation Report
    type_print(f"\n[執政官 {consul_name} の報告]")
    type_print(f"「我が主よ。東の『赤の共和国』は資金が潤沢ですが軍備が手薄です。今すぐ軍を向ければ落とせるでしょう。いかがなさいますか？」")
    
    # Phase 3: Player Advice
    print("\n[Phase 3: プレイヤー指導（アドバイス）フェーズ]")
    advice = input(f"{country_name}の指導者として指示を与えてください: ")
    if not advice:
        advice = "今は内政をして資金を貯めろ。戦争はまだ早い。"
        print(f"(入力なし -> デフォルト指示: '{advice}')")
        
    # Phase 4: AI Decision (Physics computation)
    print("\n[Phase 4: AI行動決定・実行フェーズ]")
    print(">>> 潜在空間にて『指導者の意図ベクトル』と『執政官の性格アトラクター』を干渉計算中...")
    time.sleep(1.5)
    
    actions, reasoning, intent_vector = consul.decide_action(advice, board_state)
    
    print("\n" + "=" * 40)
    type_print(f"執政官 {consul_name} の最終決断:", delay=0.05)
    type_print(f"  Action 1: {actions[0]}")
    type_print(f"  Action 2: {actions[1]}")
    print("=" * 40)
    
    print("\n[AIの思考ログ（なぜその手を選んだか）]")
    type_print(reasoning, delay=0.04)
    
    print("\n[デバッグ情報: 潜在ベクトル座標]")
    print(f"  [軍事, 内政, 外交] = [{intent_vector[0]:.2f}, {intent_vector[1]:.2f}, {intent_vector[2]:.2f}]")
    
    print("\n================== ターン終了 ==================")

if __name__ == "__main__":
    main()
