import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from engine.memory import MemoryEngine
from engine.reference import ReferenceRouter

PORT = 8085
memory = MemoryEngine()
router = ReferenceRouter(memory)

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
            self.wfile.write(json.dumps(memory.get_full_state(), ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req = json.loads(post_data.decode('utf-8'))
            user_prompt = req.get('prompt', '')
            
            # レファレンスAIへ処理委譲
            result = router.handle_user_request(user_prompt)
            
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
    httpd = HTTPServer(server_address, EphemeralCoderHandler)
    print(f"==================================================")
    print(f" Ephemeral Latent Coding Engine (ELCE) Web Server ")
    print(f" Running at: http://localhost:{PORT}/             ")
    print(f"==================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
