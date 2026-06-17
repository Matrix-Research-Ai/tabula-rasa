"""Server that dumps detect_skill info then starts."""
import sys, json
sys.path.insert(0, 'src'); sys.path.insert(0, '.')
import egefalos.tabula_rasa as tr

info = {
    'module_file': tr.__file__,
    'explanation_ops': tr.SKILL_REGISTRY['explanation_question']['ops'],
    'conversation_ops': tr.SKILL_REGISTRY['conversation']['ops'],
    'detect_how_are_you': tr.detect_skill('how are you?'),
    'detect_how_gravity': tr.detect_skill('how does gravity work?'),
}

# Write to known path
import os
path = os.path.join(os.path.dirname(tr.__file__), '..', 'server_dump.json')
with open(os.path.normpath(path), 'w') as f:
    json.dump(info, f, indent=2)

tr.manager = tr.SkillManager()
tr.TabulaRasaHandler.manager = tr.manager
server = tr.ThreadedHTTPServer(('0.0.0.0', 8002), tr.TabulaRasaHandler)
server.serve_forever()
