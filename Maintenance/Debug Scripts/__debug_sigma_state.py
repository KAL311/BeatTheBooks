import json
from pathlib import Path
state = json.loads(Path('model_state.json').read_text(encoding='utf-8'))
print(json.dumps(state.get('total_sigma_calibration'), indent=2))
