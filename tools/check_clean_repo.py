"""Check tracked paths and token patterns without printing candidate secrets."""
from pathlib import Path
import subprocess,re,csv,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]

def checked_weight_paths():
 manifest=json.loads((ROOT/'weights/manifest.json').read_text('utf-8'))
 if manifest.get('schema')!='project_trained_weights_v1':raise ValueError('Unknown weights schema')
 frozen=json.loads((ROOT/'artifacts/frozen_runs.json').read_text('utf-8'))['files']
 expected={}
 for source,entry in frozen.items():
  if not source.endswith('/best.pt'):continue
  parts=source.split('/')
  method='v1' if parts[1]=='verification_v1' else parts[4]
  expected[f"weights/{method}/{parts[3]}.pt"]=(source,entry)
 rows=manifest['files']
 if len(expected)!=18 or len(rows)!=18 or {r['path'] for r in rows}!=set(expected):raise ValueError('Expected exactly 18 pinned weight files')
 for row in rows:
  source,entry=expected[row['path']]
  if (row['source_relative_path']!=source or row['sha256']!=entry['sha256'] or row['bytes']!=entry['bytes']):raise ValueError('Weight manifest source mismatch')
  file=ROOT/row['path']
  if file.stat().st_size!=entry['bytes'] or hashlib.sha256(file.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Weight checksum mismatch')
 return set(expected)

def main():
 q=subprocess.run(['git','-C',str(ROOT),'ls-files','-z'],check=True,capture_output=True)
 files=[Path(p) for p in q.stdout.decode('utf-8').split('\0') if p]
 if not files:raise RuntimeError('Stage the proposed clean files before checking')
 allowed_weights=checked_weight_paths()
 bad=[];large=[];secret=[];personal_rows=[];machine_paths=[]
 banned_dirs={'.venv','node_modules','__pycache__','.aws','.ssh','.codex','.agents','outputs','local_artifacts','reviews','deliverables','.qa'}
 banned_suffix={'.npy','.npz','.pt','.pth','.ckpt','.pem','.key','.pfx','.p12','.log','.pyc'}
 patterns={
  'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
  'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
  'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
  'openai_token':rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b',
 }
 for rel in files:
  p=ROOT/rel
  if any(x in banned_dirs for x in rel.parts) or (p.suffix.lower() in banned_suffix and rel.as_posix() not in allowed_weights) or p.name.startswith('.env'):bad.append(rel.as_posix())
  if p.stat().st_size>2*1024**2:large.append(rel.as_posix())
  data=p.read_bytes()
  for kind,pat in patterns.items():
   if re.search(pat,data):secret.append({'file':rel.as_posix(),'category':kind})
  text=data.decode('utf-8',errors='replace')
  if re.search(r'[A-Za-z]:[\\/]Users[\\/]',text):machine_paths.append(rel.as_posix())
  if p.suffix=='.csv':
   with p.open(encoding='utf-8-sig',newline='') as f:columns=next(csv.reader(f),[])
   if any(c in {'participant_id','claimed_id','source_id','recording_id','subject_id','genuine_score'} for c in columns):personal_rows.append(rel.as_posix())
 result={'tracked_files':len(files),'tracked_bytes':sum((ROOT/p).stat().st_size for p in files),'audited_trained_weights':len(allowed_weights),'forbidden_artifacts':bad,'oversized_files':large,'credential_pattern_hits':secret,'participant_level_tables':personal_rows,'machine_user_paths':machine_paths,'scope':'Pattern and allowlist audit, not a proof that every possible secret representation is detectable.'}
 print(json.dumps(result,ensure_ascii=True,indent=2))
 return 1 if any([bad,large,secret,personal_rows,machine_paths]) else 0
if __name__=='__main__':sys.exit(main())
