import json
import numpy as np
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import uuid

# --- Agent-Based Semantic ALife Engine (V4: Vision, Movement Choice, Swarm) ---
WIDTH = 800
HEIGHT = 800
VECTOR_DIM = 16
VISION_RADIUS = 150 # 視界の広さ

np.random.seed(42)
ANCHORS = {
    "Red (捕食/単独)": np.random.randn(VECTOR_DIM),
    "Green (採集/草食)": np.random.randn(VECTOR_DIM),
    "Blue (機動/逃走)": np.random.randn(VECTOR_DIM),
    "Cyan (群れ/同期)": np.random.randn(VECTOR_DIM)
}
for k in ANCHORS:
    ANCHORS[k] /= np.linalg.norm(ANCHORS[k])

class Agent:
    def __init__(self, x, y, vector=None, energy=100.0):
        self.id = str(uuid.uuid4())[:8]
        self.x = x
        self.y = y
        self.vx = np.random.randn() * 2
        self.vy = np.random.randn() * 2
        
        if vector is None:
            self.vector = np.random.randn(VECTOR_DIM)
            self.vector /= np.linalg.norm(self.vector)
        else:
            self.vector = vector
            
        self.energy = energy
        self.age = 0
        self.update_traits()

    def update_traits(self):
        r = max(0, np.dot(self.vector, ANCHORS["Red (捕食/単独)"]))
        g = max(0, np.dot(self.vector, ANCHORS["Green (採集/草食)"]))
        b = max(0, np.dot(self.vector, ANCHORS["Blue (機動/逃走)"]))
        c = max(0, np.dot(self.vector, ANCHORS["Cyan (群れ/同期)"]))
        
        # Color mapping (Cyan is G+B)
        rc = r
        gc = g + c
        bc = b + c
        rgb = np.array([rc, gc, bc])
        if np.sum(rgb) > 0:
            rgb = (rgb / np.max(rgb)) * 255
        self.color = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
        
        self.trait_predator = r
        self.trait_forager = g
        self.trait_speed = max(1.0, b * 3 + 1.5)
        self.trait_swarm = c # 群れを作る傾向 (Boidsの強さ)

class ALifeWorld:
    def __init__(self):
        # 1. 個体数を大幅に増加
        self.agents = [Agent(np.random.rand()*WIDTH, np.random.rand()*HEIGHT) for _ in range(120)]
        self.foods = []
        self.generation = 0
        self.ticks = 0

    def step(self):
        self.ticks += 1
        
        # 2. 餌のスポーン量と上限を増加
        if len(self.foods) < 300 and np.random.rand() < 0.6:
            self.foods.append({
                'x': np.random.rand() * WIDTH,
                'y': np.random.rand() * HEIGHT
            })

        new_agents = []
        dead_agents = set()

        for i, a in enumerate(self.agents):
            if a.id in dead_agents: continue
            
            a.age += 1
            age_penalty = (a.age / 400.0) ** 2 if a.age > 300 else 0
            a.energy -= 0.1 * (a.trait_speed ** 1.1) + age_penalty
            
            # --- 1. Vision (視界内の情報取得) ---
            seen_foods = []
            for f in self.foods:
                dx, dy = f['x'] - a.x, f['y'] - a.y
                if dx*dx + dy*dy < VISION_RADIUS*VISION_RADIUS:
                    seen_foods.append((dx, dy, np.sqrt(dx*dx + dy*dy), f))
                    
            seen_agents = []
            for other in self.agents:
                if other.id == a.id or other.id in dead_agents: continue
                dx, dy = other.x - a.x, other.y - a.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < VISION_RADIUS:
                    sim = np.dot(a.vector, other.vector)
                    seen_agents.append((dx, dy, dist, sim, other))

            # --- 2. Movement Selection (行動選択) ---
            force_x, force_y = np.random.randn() * 0.1, np.random.randn() * 0.1 # Wander
            
            # Foraging (餌へ向かう)
            if a.trait_forager > 0.1 and seen_foods:
                seen_foods.sort(key=lambda x: x[2]) # Sort by distance
                dx, dy, dist, f = seen_foods[0]
                
                # 3. 食べる判定範囲を広げて「周りを回ってしまう」問題を解決
                if dist < 20:
                    a.energy += 30
                    if f in self.foods: self.foods.remove(f)
                    # Learning
                    a.vector = a.vector + ANCHORS["Green (採集/草食)"] * 0.05
                    a.vector /= np.linalg.norm(a.vector)
                    a.update_traits()
                else:
                    # Target velocity steering to prevent orbits
                    desired_vx = (dx / dist) * a.trait_speed * 3.0
                    desired_vy = (dy / dist) * a.trait_speed * 3.0
                    force_x += (desired_vx - a.vx) * a.trait_forager * 0.5
                    force_y += (desired_vy - a.vy) * a.trait_forager * 0.5

            # Boids & Predation (他エージェントへの反応)
            center_x, center_y = 0.0, 0.0
            align_vx, align_vy = 0.0, 0.0
            swarm_count = 0
            
            for dx, dy, dist, sim, other in seen_agents:
                # 衝突回避 (Separation) 全員に対して
                if dist < 20:
                    force_x -= (dx / dist) * 4.0
                    force_y -= (dy / dist) * 4.0
                    
                # 捕食・逃走
                if sim < 0.0:
                    if a.trait_predator > other.trait_predator + 0.1:
                        # Hunt (Chase)
                        force_x += (dx / dist) * a.trait_predator * 2.0
                        force_y += (dy / dist) * a.trait_predator * 2.0
                        if dist < 18: # Bite
                            absorbed = min(other.energy, 40)
                            a.energy += absorbed
                            other.energy -= absorbed
                            a.vector = a.vector + other.vector * 0.02
                            a.vector /= np.linalg.norm(a.vector)
                            a.update_traits()
                            if other.energy <= 0: dead_agents.add(other.id)
                    elif other.trait_predator > a.trait_predator + 0.1:
                        # Flee (逃走)
                        force_x -= (dx / dist) * 4.0
                        force_y -= (dy / dist) * 4.0
                
                # 群れ (Swarm / Boids) 類似度が高い仲間
                elif sim > 0.7:
                    center_x += dx
                    center_y += dy
                    align_vx += other.vx
                    align_vy += other.vy
                    swarm_count += 1
                    
                    # 4. 繁殖のエネルギー閾値を下げて個体数を維持しやすくする
                    if dist < 20 and a.energy > 70 and other.energy > 70:
                        a.energy -= 35
                        other.energy -= 35
                        child_vec = (a.vector + other.vector) / 2.0
                        child_vec += np.random.randn(VECTOR_DIM) * 0.1
                        child_vec /= np.linalg.norm(child_vec)
                        new_agents.append(Agent(a.x, a.y, vector=child_vec, energy=50))

            if swarm_count > 0 and a.trait_swarm > 0.2:
                # Cohesion (凝集)
                center_x /= swarm_count
                center_y /= swarm_count
                dist_c = np.sqrt(center_x**2 + center_y**2)
                if dist_c > 0:
                    force_x += (center_x / dist_c) * a.trait_swarm * 1.5
                    force_y += (center_y / dist_c) * a.trait_swarm * 1.5
                
                # Alignment (整列)
                align_vx /= swarm_count
                align_vy /= swarm_count
                speed_a = np.sqrt(align_vx**2 + align_vy**2)
                if speed_a > 0:
                    force_x += (align_vx / speed_a) * a.trait_swarm * 1.0
                    force_y += (align_vy / speed_a) * a.trait_swarm * 1.0

            # --- 3. 速度の適用と摩擦 ---
            a.vx += force_x * 0.8
            a.vy += force_y * 0.8
            
            # 5. 水中や空気抵抗のような摩擦(Friction)を追加して「氷上を滑って周回する」のを防ぐ
            a.vx *= 0.85
            a.vy *= 0.85
            
            speed = np.sqrt(a.vx**2 + a.vy**2)
            max_speed = a.trait_speed * 3.0
            if speed > max_speed:
                a.vx = (a.vx / speed) * max_speed
                a.vy = (a.vy / speed) * max_speed
            
            a.x += a.vx
            a.y += a.vy
            
            # 壁でのバウンド
            if a.x < 0 or a.x > WIDTH: a.vx *= -1; a.x = max(0, min(WIDTH, a.x))
            if a.y < 0 or a.y > HEIGHT: a.vy *= -1; a.y = max(0, min(HEIGHT, a.y))
            
            if a.energy <= 0: dead_agents.add(a.id)

        self.agents = [a for a in self.agents if a.id not in dead_agents]
        self.agents.extend(new_agents)
        if len(self.agents) < 10:
            self.agents.append(Agent(np.random.rand()*WIDTH, np.random.rand()*HEIGHT))

    def get_state(self):
        return {
            "ticks": self.ticks,
            "agents": [{"id": a.id, "x": a.x, "y": a.y, "color": a.color, "energy": a.energy, "r": max(3, a.energy/10)} for a in self.agents],
            "foods": [{"x": f['x'], "y": f['y']} for f in self.foods]
        }

    def decode_agent(self, agent_id):
        a = next((x for x in self.agents if x.id == agent_id), None)
        if not a: return "Agent not found or died."
        
        scores = {k: np.dot(a.vector, ANCHORS[k]) for k in ANCHORS}
        top = max(scores, key=scores.get)
        
        desc = f"【個体解析レポート (V4)】\nID: {a.id} | 年齢: {a.age} | エネルギー: {a.energy:.1f}\n\n"
        
        desc += f"【物理的特性・行動指針】\n"
        desc += f"支配アトラクター: {top}\n"
        desc += f"- 捕食性(赤): {a.trait_predator:.2f}\n"
        desc += f"- 採集性(緑): {a.trait_forager:.2f}\n"
        desc += f"- 機動力(青): {a.trait_speed:.2f}\n"
        desc += f"- 群れ性(水色): {a.trait_swarm:.2f}\n\n"
        
        desc += "《行動プロファイル》\n"
        if a.trait_swarm > 0.6:
            desc += "極めて社会性が高く、類似したベクトルの仲間を見つけると同調（Boids則）し、巨大な群れ（Swarm）を形成して捕食者から身を守ります。"
        elif a.trait_predator > 0.6:
            desc += "視界内の『弱い・異なるベクトルの個体』を的確にターゲットとして選び、追跡して狩る単独のハンターです。"
        else:
            desc += "視界内の『餌』を最優先で探索し、敵を見つけるといち早く逃走経路を計算して逃げる生存特化型です。"
            
        return desc


world = ALifeWorld()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            with open('index.html', 'r', encoding='utf-8') as f:
                self.wfile.write(f.read().encode())
        elif self.path == '/api/state':
            world.step()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(world.get_state()).encode())
        elif self.path.startswith('/api/decode'):
            q = self.path.split('?id=')[1]
            desc = world.decode_agent(q)
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({"text": desc}, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8081):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Starting ALife V4 (Swarm Edition) on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
