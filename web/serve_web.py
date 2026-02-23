#!/usr/bin/env python3
"""
Simple HTTP server to serve the ASCEC Input Generator web form.

Usage:
    python serve_web.py [port]
    
Default port is 8080. Open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
import os
import sys
import webbrowser
from functools import partial

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    # Change to the web directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.join(script_dir)
    
    if not os.path.exists(web_dir):
        print(f"Error: Web directory not found at {web_dir}")
        sys.exit(1)
    
    os.chdir(web_dir)
    
    # Custom handler to serve index.html by default
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=web_dir)
    
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
