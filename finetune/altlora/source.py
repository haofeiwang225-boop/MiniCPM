"""Load only optimizer_new.py; do not shadow installed transformers/PEFT."""
from pathlib import Path
import hashlib
import importlib.util
import sys
ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/'third_party/altlora_official/AltLoRA-main'
def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def vendor_manifest():
 return {str(p.relative_to(VENDOR)):sha256(p) for p in sorted(VENDOR.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def official_class():
 name='_minicpm_official_altlora_optimizer'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,VENDOR/'optimizer_new.py');module=importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module);sys.modules[name]=module
 return sys.modules[name].altlora
