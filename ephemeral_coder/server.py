import sys
import json
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from engine.memory_cluster import MemoryCluster
from engine.reference_cluster import ReferenceCluster
from engine.main_ai import MainAI

PORT = 8085
memory_cluster = MemoryCluster()
reference_cluster = ReferenceCluster(memory_cluster)
main_ai = MainAI(memory_cluster, reference_cluster)

class EphemeralCoderHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self._send_cors()
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                self.wfile.write(f.read().encode())
                
        elif self.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(memory_cluster.get_state(), ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode('utf-8'))

        if self.path == '/api/chat/plan':
            # ユーザーの入力からメインAIが設計書とタスク分割を作成（承認要求）
            user_prompt = req.get('prompt', '')
            spec = main_ai.generate_plan(user_prompt)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(spec, ensure_ascii=False).encode('utf-8'))

        elif self.path == '/api/chat/approve':
            # ユーザーが承認ボタンを押下して実行命令を発行
            spec_id = req.get('spec_id', '')
            result = main_ai.execute_approved_plan(spec_id)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self._send_cors()
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('', PORT)
    httpd = ThreadingHTTPServer(server_address, EphemeralCoderHandler)
    print(f"==================================================")
    print(f" Ephemeral Latent Coding Engine (ELCE v2.0)       ")
    print(f" Hierarchical Main/Worker/Reference/Memory Swarm  ")
    print(f" Running at: http://localhost:{PORT}/             ")
    print(f"==================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
