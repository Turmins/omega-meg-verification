"""Copy a hash-verified saved V1 replay, never run neural inference."""
from pathlib import Path
import argparse,hashlib,json,re,math,shutil
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive-root',type=Path,default=ROOT.parent/'deep_learning_extension')
 p.add_argument('--output-dir',type=Path,default=ROOT/'local_artifacts/verification_v1/report')
 a=p.parse_args();archive=a.archive_root.resolve();out=a.output_dir.resolve()
 if not archive.is_dir():raise FileNotFoundError('Supply the existing source archive with --archive-root')
 if out==archive or archive in out.parents:raise ValueError('Output must not write inside the original archive')
 if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be a new empty directory')
 manifest=json.loads((ROOT/'artifacts/EXTERNAL_ARTIFACTS.json').read_text(encoding='utf-8'))
 files=manifest['saved_V1_demo_files']
 for name,row in files.items():
  rel=Path(row['source_relative_path']);source=(archive/rel).resolve()
  if archive not in source.parents or source.stat().st_size!=row['size_bytes'] or sha(source)!=row['sha256']:raise ValueError('External artifact hash/size mismatch: '+name)
 text=(archive/files['demo.html']['source_relative_path']).read_text(encoding='utf-8')
 match=re.search(r'<script id="experiment-data" type="application/json">(.*?)</script>',text,re.S)
 if not match:raise ValueError('Saved data block missing')
 d=json.loads(match.group(1));assert len(d['folds'])==6 and all(len(ids)==10 for ids in d['folds']) and len(d['pairs'])==7200
 for f,ids in enumerate(d['folds']):
  for claim in ids:
   for source in ids:
    for duration in [10,30,60]:
     for start in [0,60,120,180]:
      pair=d['pairs'][f'{f}|{claim}|{source}|{duration}|{start}'];assert set(pair)=={'phase','cnn'}
      for score,threshold,accepted in pair.values():assert math.isfinite(score) and math.isfinite(threshold) and bool(accepted)==(score>=threshold)
 out.mkdir(parents=True,exist_ok=True)
 for name,row in files.items():
  dest=out/name
  with dest.open('xb') as handle, (archive/row['source_relative_path']).open('rb') as source:shutil.copyfileobj(source,handle)
  assert sha(dest)==row['sha256']
 print(json.dumps({'mode':'saved V1 replay; no inference','claims':7200,'method_decisions':14400,'output':str(out/'demo.html')},ensure_ascii=True))
if __name__=='__main__':main()
