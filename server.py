#!/usr/bin/env python3
import json
import subprocess
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# API key should be set via environment variable
# export GROQ_API_KEY='your_key_here'
if 'GROQ_API_KEY' not in os.environ:
    print("ERROR: GROQ_API_KEY environment variable not set")
    sys.exit(1)

class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            message = data.get('message', '')

            # Run chatbot
            try:
                result = subprocess.run(
                    [sys.executable, 'fmcg_chat.py'],
                    input=f"{message}\nquit\n",
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                response = result.stdout.strip() if result.stdout else "No response"
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'response': response}).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'response': f'Error: {str(e)}'}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == '__main__':
    server = HTTPServer(('localhost', 9000), ChatHandler)
    print("🚀 FMCG Chat Server running on http://localhost:9000")
    print("📖 Open index.html in browser")
    print("⏹ Press Ctrl+C to stop")
    server.serve_forever()
