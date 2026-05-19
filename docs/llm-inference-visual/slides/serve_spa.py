#!/usr/bin/env python3
"""SPA-aware HTTP server for nano-vllm slides dist/ output.

Usage: python3 serve_spa.py [port] [directory]

Serves files from the given directory (default: ./dist).
For any path that doesn't correspond to a real file, serves index.html
instead — required for client-side routing (e.g., /1, /2, /overview).
"""
import http.server
import os
import sys


class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    def do_GET(self):
        path = self.translate_path(self.path)
        # If the requested path doesn't exist as a file, serve index.html
        if not os.path.exists(path) or os.path.isdir(path):
            self.path = "/index.html"
        super().do_GET()


if __name__ == "__main__":
    SERVE_DIR = sys.argv[2] if len(sys.argv) > 2 else "dist"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    # Bind to all interfaces so LAN access works too
    server = http.server.HTTPServer(("0.0.0.0", port), SPAHandler)
    print(f"SPA server running at http://localhost:{port}")
    print(f"Serving directory: {os.path.abspath(SERVE_DIR)}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()
