#!/usr/bin/env python3
"""
ASCEC Input Generator Web Server with 3D Generation Proxy.

This server provides:
1. Static file serving for the web interface
2. A proxy endpoint for novoprolabs SMILES-to-3D API (which uses RDKit)

Usage:
    python serve_web.py [port]
    
Default port is 8080. Open http://localhost:8080 in your browser.

Note: The proxy uses only Python standard library - no extra packages needed!
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import json
import urllib.request
import urllib.parse
from functools import partial


class ASCECHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with API proxy for 3D generation."""
    
    def do_POST(self):
        """Handle POST requests - proxy to novoprolabs for 3D generation."""
        if self.path == '/api/smiles-to-3d':
            self.handle_smiles_to_3d()
        else:
            self.send_error(404, 'Not Found')
    
    def handle_smiles_to_3d(self):
        """Proxy SMILES to novoprolabs for 3D MOL generation."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            smiles = data.get('smiles', '')
            fmt = data.get('format', 'mol')  # mol, sdf, or pdb
            
            if not smiles:
                self.send_json_response({'success': False, 'error': 'Missing SMILES'}, 400)
                return
            
            # Step 1: Request 3D generation from novoprolabs
            post_data = urllib.parse.urlencode({
                'sr': 'smiles2pdb',
                'fmt': fmt,
                'sq': smiles
            }).encode('utf-8')
            
            req = urllib.request.Request(
                'https://www.novoprolabs.com/plus/ppc.php',
                data=post_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = response.read().decode('utf-8')
            
            # Parse response: [1, "filename.mol"] or [0, "error"]
            try:
                parsed = json.loads(result)
                if not isinstance(parsed, list) or len(parsed) < 2:
                    raise ValueError("Invalid response format")
                
                status_code = parsed[0]
                filename = parsed[1]
                
                if status_code != 1:
                    self.send_json_response({
                        'success': False, 
                        'error': f'3D generation failed: {filename}'
                    }, 500)
                    return
                
            except (json.JSONDecodeError, ValueError) as e:
                self.send_json_response({
                    'success': False,
                    'error': f'Invalid response from 3D server: {result[:100]}'
                }, 500)
                return
            
            # Step 2: Download the generated 3D file
            file_url = f'https://novopro.cn/plus/tmp/{filename}'
            
            with urllib.request.urlopen(file_url, timeout=30) as response:
                mol_data = response.read().decode('utf-8')
            
            # Verify it's a valid MOL/SDF file
            if 'V2000' not in mol_data and 'V3000' not in mol_data:
                self.send_json_response({
                    'success': False,
                    'error': 'Invalid MOL file received'
                }, 500)
                return
            
            # Check if it has 3D coordinates (Z values vary)
            is_3d = '3D' in mol_data[:100]
            
            self.send_json_response({
                'success': True,
                'mol_block': mol_data,
                'is_3d': is_3d,
                'method': 'RDKit (novoprolabs)'
            })
            
        except urllib.error.URLError as e:
            self.send_json_response({
                'success': False,
                'error': f'Network error: {str(e)}'
            }, 503)
        except Exception as e:
            self.send_json_response({
                'success': False,
                'error': str(e)
            }, 500)
    
    def send_json_response(self, data, status=200):
        """Send a JSON response with CORS headers."""
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(response)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Change to the web directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = script_dir
    
    if not os.path.exists(web_dir):
        print(f"Error: Web directory not found at {web_dir}")
        sys.exit(1)
    
    os.chdir(web_dir)
    
    # Create handler with web_dir set
    handler = partial(ASCECHandler, directory=web_dir)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}/ascec_input_generator.html"
        server_line = f"http://localhost:{port}"
        
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║              ASCEC Input Generator Web Server                  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║   Server running at: {server_line:<42}║
║                                                                ║
║   3D Generation: PubChem + RDKit proxy (novoprolabs)           ║
║   No extra Python packages required!                           ║
║                                                                ║
║   API Endpoints:                                               ║
║     POST /api/smiles-to-3d  - Generate 3D from SMILES          ║
║                                                                ║
║   Open in browser:                                             ║
║   {url:<61}║
║                                                                ║
║   Press Ctrl+C to stop the server                              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
        
        # Try to open browser automatically
        try:
            webbrowser.open(url)
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
