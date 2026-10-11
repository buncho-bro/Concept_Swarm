import json
import numpy as np
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

# --- Semantic ALife Engine ---
GRID_SIZE = 15
VECTOR_DIM = 16

# Define some "Concept Anchors" in our mock 16D latent space
np.random.seed(42)
ANCHORS = {
    "Red (捕食/攻撃)": np.random.randn(VECTOR_DIM),
    "Green (光合成/協調)": np.random.randn(VECTOR_DIM),
    "Blue (探索/遊牧)": np.random.randn(VECTOR_DIM)
}
# Normalize anchors
for k in ANCHORS:
    ANCHORS[k] /= np.linalg.norm(ANCHORS[k])

class ALifeGrid:
    def __init__(self):
        # Grid of vectors
        self.grid = np.random.randn(GRID_SIZE, GRID_SIZE, VECTOR_DIM)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.grid[i, j] /= np.linalg.norm(self.grid[i, j])
        # Energy levels (0.0 to 1.0)
        self.energy = np.ones((GRID_SIZE, GRID_SIZE)) * 0.5
        self.generation = 0

    def step(self):
        new_grid = self.grid.copy()
        new_energy = self.energy.copy()
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                # Pick a random neighbor
                ni = (i + np.random.randint(-1, 2)) % GRID_SIZE
                nj = (j + np.random.randint(-1, 2)) % GRID_SIZE
                
                v_self = self.grid[i, j]
                v_neigh = self.grid[ni, nj]
                
                sim = np.dot(v_self, v_neigh) # Cosine similarity
                
                if sim > 0.6:
                    # Symbiosis / Mating: Vectors pull closer, energy increases slightly
                    avg = (v_self + v_neigh) / 2.0
                    avg /= np.linalg.norm(avg)
                    new_grid[i, j] = avg
                    new_energy[i, j] = min(1.0, new_energy[i, j] + 0.1)
                elif sim < -0.2:
                    # Conflict / Predation
                    if self.energy[i, j] > self.energy[ni, nj]:
                        # Self wins: absorb neighbor's energy, neighbor vector gets overwritten by self
                        new_energy[i, j] = min(1.0, new_energy[i, j] + new_energy[ni, nj] * 0.5)
                        new_grid[ni, nj] = v_self # Overwrite/Reproduce into that cell
                        new_energy[ni, nj] = 0.1
                    else:
                        # Self loses
                        new_energy[i, j] -= 0.2
                else:
                    # Neutral drift (Mutation)
                    drift = np.random.randn(VECTOR_DIM) * 0.05
                    v_new = v_self + drift
                    new_grid[i, j] = v_new / np.linalg.norm(v_new)
                    new_energy[i, j] = max(0.0, new_energy[i, j] - 0.05) # Starvation

                # Death -> Spontaneous generation of random noise
                if new_energy[i, j] <= 0:
                    v_new = np.random.randn(VECTOR_DIM)
                    new_grid[i, j] = v_new / np.linalg.norm(v_new)
                    new_energy[i, j] = 0.5

        self.grid = new_grid
        self.energy = new_energy
        self.generation += 1

    def get_color_map(self):
        """Map vectors to RGB by their similarity to the 3 anchors"""
        colors = []
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                v = self.grid[i, j]
                r = max(0, np.dot(v, ANCHORS["Red (捕食/攻撃)"]))
                g = max(0, np.dot(v, ANCHORS["Green (光合成/協調)"]))
                b = max(0, np.dot(v, ANCHORS["Blue (探索/遊牧)"]))
                
                # Normalize color
                rgb = np.array([r, g, b])
                if np.sum(rgb) > 0:
                    rgb = (rgb / np.max(rgb)) * 255 * self.energy[i, j] # Dim if low energy
                else:
                    rgb = np.array([0, 0, 0])
                row.append(f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})")
            colors.append(row)
        return colors

    def decode_cell(self, i, j):
        """Mock LLM Translation: Translate the latent vector back to text"""
        v = self.grid[i, j]
        scores = {k: np.dot(v, ANCHORS[k]) for k in ANCHORS}
        top_concept = max(scores, key=scores.get)
        
        description = "【Output Bridge 翻訳ログ】\n"
        description += f"細胞座標 ({i}, {j}) | 世代: {self.generation} | 活性エネルギー: {self.energy[i, j]:.2f}\n"
        description += f"最も支配的な概念アトラクター: {top_concept} (スコア: {scores[top_concept]:.2f})\n\n"
        
        if "Red" in top_concept:
            description += "《解析結果》\nこの生命体は極めて攻撃的な『捕食者』のベクトルへ進化しました。異なるベクトルの細胞を見つけるとエネルギーを奪って自らのコピー（繁殖）を強制するアルゴリズムを持っています。"
        elif "Green" in top_concept:
            description += "《解析結果》\nこの生命体は『協調・群生』のベクトルへ進化しました。似たベクトルの細胞と隣り合うことで互いのエネルギーを分け合い、巨大なコロニーを形成して捕食者から身を守ります。"
        else:
            description += "《解析結果》\nこの生命体は『探索・遊牧』のベクトルへ進化しました。定住を好まず、常に概念空間のドリフト（突然変異）を繰り返すことで、環境への適応を模索しています。"
            
        return description

# --- Server Logic ---
alife = ALifeGrid()

class RequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            with open('index.html', 'r', encoding='utf-8') as f:
                self.wfile.write(f.read().encode())
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            data = {
                "generation": alife.generation,
                "colors": alife.get_color_map()
            }
            self.wfile.write(json.dumps(data).encode())
        elif self.path.startswith('/api/step'):
            steps = 1
            if '?n=' in self.path:
                steps = int(self.path.split('?n=')[1])
            for _ in range(steps):
                alife.step()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            data = {
                "generation": alife.generation,
                "colors": alife.get_color_map()
            }
            self.wfile.write(json.dumps(data).encode())
        elif self.path.startswith('/api/decode'):
            query = self.path.split('?')[1]
            params = dict(qc.split('=') for qc in query.split('&'))
            i = int(params['i'])
            j = int(params['j'])
            desc = alife.decode_cell(i, j)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"text": desc}, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Starting ALife Semantic Engine on http://localhost:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
