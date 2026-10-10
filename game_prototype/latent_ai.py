import numpy as np

class LatentConsulAI:
    def __init__(self, name, doctrine, aggressive, rational, coop):
        self.name = name
        self.doctrine = doctrine
        # Sliders: -1.0 to 1.0
        self.aggressive = aggressive  # 1.0 = War, -1.0 = Peace/Defend
        self.rational = rational      # 1.0 = Logic/Compliance, -1.0 = Emotion/Rebel
        self.coop = coop              # 1.0 = Alliance, -1.0 = Isolation
        
        # Base Personality Attractor in 3D Space: [Military, Economy, Diplomacy]
        self.attractor = np.array([
            self.aggressive,
            -self.aggressive * 0.5 + self.rational * 0.5, # Cautious/Rational favors economy
            self.coop
        ])
        # Normalize
        norm = np.linalg.norm(self.attractor)
        if norm > 0:
            self.attractor = self.attractor / norm
            
    def _parse_advice_to_drift(self, advice_text):
        """Simulates NLP extraction of intent vector from player advice."""
        advice = advice_text.lower()
        drift = np.array([0.0, 0.0, 0.0]) # [Mil, Eco, Dip]
        
        if any(w in advice for w in ["戦", "攻", "殺", "軍", "兵", "奪", "attack", "war"]):
            drift[0] += 1.0
        if any(w in advice for w in ["内政", "金", "資源", "開発", "守", "economy", "invest"]):
            drift[1] += 1.0
        if any(w in advice for w in ["同盟", "交渉", "条約", "友", "ally", "diplomacy"]):
            drift[2] += 1.0
            
        norm = np.linalg.norm(drift)
        return (drift / norm) if norm > 0 else np.array([0.0, 1.0, 0.0]) # default to eco

    def decide_action(self, advice_text, board_state):
        """
        Simulates Latent Concept Physics.
        Vector is pulled by Player Advice (drift) and AI Personality (attractor).
        """
        drift_vector = self._parse_advice_to_drift(advice_text)
        
        # Physics simulation: The vector is a weighted sum based on "Rationality" (compliance)
        # Highly emotional AIs heavily weight their own attractor basin over the player's drift.
        compliance = (self.rational + 1.0) / 2.0  # map -1..1 to 0..1
        compliance = max(0.1, min(0.9, compliance)) # bounds
        
        final_intent = (drift_vector * compliance) + (self.attractor * (1.0 - compliance))
        final_intent = final_intent / np.linalg.norm(final_intent)
        
        # Determine Top 2 Actions based on highest dimensions
        actions_map = {0: "軍事行動 (Military)", 1: "内政投資 (Economy)", 2: "外交交渉 (Diplomacy)"}
        sorted_indices = np.argsort(-final_intent) # Descending
        
        chosen_actions = [actions_map[sorted_indices[0]], actions_map[sorted_indices[1]]]
        
        # Generate Reasoning Log
        reasoning = ""
        rebelled = False
        
        # Did the AI disobey the primary advice?
        primary_advice_idx = np.argmax(drift_vector)
        if primary_advice_idx != sorted_indices[0] and np.linalg.norm(drift_vector) > 0:
            rebelled = True
            
        if rebelled:
            reasoning = f"【主の言葉への反発】主は『{actions_map[primary_advice_idx]}』を望んでおられるようですが、我が国の教義「{self.doctrine}」と私の信念に反します。私は独断で『{chosen_actions[0]}』を優先しました！"
        else:
            reasoning = f"【主の言葉への服従】主の『{actions_map[primary_advice_idx]}』という方針は、現在の状況と私の思考にも合致しました。指示通り『{chosen_actions[0]}』を実行します。"
            
        return chosen_actions, reasoning, final_intent
