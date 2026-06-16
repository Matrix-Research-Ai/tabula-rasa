"""Run the Tabula Rasa AI server with proper path setup."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import egefalos.tabula_rasa as tr

mgr = tr.SkillManager()
tr.manager = mgr
server = tr.ThreadedHTTPServer(('0.0.0.0', 8002), tr.TabulaRasaHandler)
print(f'[*] Tabula Rasa AI on port 8002 | known: {mgr.known_skills}')
import os, sys as _sys
_sys.stdout.flush()
try:
    server.serve_forever()
except KeyboardInterrupt:
    print('Shutting down...')
    server.server_close()
