"""Build a self-contained Worker using the same model and UI as Python."""
from pathlib import Path
import json
import sys
root=Path(__file__).parent
out=Path(sys.argv[1])
out.mkdir(parents=True,exist_ok=True)
worker=(root/'hosted/worker.mjs').read_text()
for name,file in [('MODEL','serving-model.json'),('HTML','web/index.html'),('STUDY','delivery-results/report.html'),('RESEARCH','robust-detection-results.json')]:
    content=(root/file).read_text()
    worker+='\nconst '+name+'='+ (content if file.endswith('.json') else json.dumps(content))+';\n'
worker+='\nexport default createHandler(MODEL,HTML,STUDY,RESEARCH);\n'
(out/'index.js').write_text(worker)
