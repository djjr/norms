"""Build index.html by pasting src/engine.js into src/template.html.

The engine lives in its own file so it can be run and tested headless in Node
(see scripts/). The page is shipped as one self-contained file.
"""
from pathlib import Path

root = Path(__file__).parent
template = (root / "src" / "template.html").read_text()
engine = (root / "src" / "engine.js").read_text()
assert "/*ENGINE*/" in template, "template is missing the /*ENGINE*/ placeholder"
(root / "index.html").write_text(template.replace("/*ENGINE*/", engine))
print("wrote index.html")
