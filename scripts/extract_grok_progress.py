"""Extract Grok progress using verified final evaluation receipts; no pause inference.
Usage: python extract_grok_progress.py ARCHIVE ORIGINAL_RUN EVALUATOR OUTPUT_JSON
"""
from pathlib import Path
import csv,datetime,hashlib,json,sys,collections,subprocess
archive,run,evaluator,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(evaluator))
from isolated_runs.monitor_progress import classify
parse=lambda s:datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
raw=(archive/'results.tsv').read_bytes();rows=list(csv.DictReader(raw.decode().splitlines(),delimiter='\t'))
points=[];seen=set();omitted=[]
for r in rows:
 if r['decision']=='pending':continue
 p=classify(r);evidence=[]
 for split in ['train','valid']:
  if not r[split+'_evaluation_id']:continue
  f=run/r[split+'_evaluation_json'].removeprefix('/workspace/')
  b=f.read_bytes();d=json.loads(b)
  assert d['evaluation_id']==r[split+'_evaluation_id']
  assert d['candidate_commit']==r['candidate_commit']
  assert d['program_id']==r[split+'_program_id']
  evidence.append({'split':split,'evaluation_id':d['evaluation_id'],'timestamp':d['timestamp'],'sha256':hashlib.sha256(b).hexdigest()})
 ids={e['evaluation_id'] for e in evidence}
 if ids<=seen:omitted.append(p['cycle']);continue
 seen.update(ids)
 t=max(parse(e['timestamp']) for e in evidence)
 commit=subprocess.check_output(['git','-C',str(archive),'show','-s','--format=%cI',r['candidate_commit']],text=True).strip()
 assert parse(commit)<=t
 p.update(evaluation_completed_at=datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat(),timestamp_evidence=evidence,commit_committed_at=commit)
 points.append(p)
assert len({p['cycle'] for p in points})==len(points)
start=parse(points[0]['evaluation_completed_at'])
for p in points:p['research_seconds']=parse(p['evaluation_completed_at'])-start
assert all(a['research_seconds']<=b['research_seconds'] for a,b in zip(points,points[1:]))
champions=[p for p in points if p['status'] in ['anchor','accepted']]
assert all(b['train_cost']<a['train_cost'] and b['valid_cost']<a['valid_cost'] for a,b in zip(champions,champions[1:]))
assert champions[-1]['candidate_commit']==json.loads((archive/'research_manifest.json').read_text())['champion']['candidate_commit']
data={'model':'Grok 4.6','effort':'xhigh','points':points,'counts':dict(collections.Counter(p['status'] for p in points)),'timing':{'started_at':points[0]['evaluation_completed_at'],'last_completed_at':points[-1]['evaluation_completed_at'],'research_seconds':points[-1]['research_seconds'],'axis':'Elapsed time (hours)','method':'Elapsed wall time from starting-anchor evaluation completion to each candidate final required evaluation. Includes pauses; no gap subtraction or active-time reconstruction. Partial research_time.jsonl is not used.'},'provenance':{'results_sha256':hashlib.sha256(raw).hexdigest(),'classifier_sha256':hashlib.sha256((evaluator/'isolated_runs/monitor_progress.py').read_bytes()).hexdigest()},'omitted_reused_receipts':omitted}
out.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'points':len(points),'counts':data['counts'],'elapsed_hours':data['timing']['research_seconds']/3600,'champion_cycle':champions[-1]['cycle'],'omitted_reused_receipts':omitted},indent=2))
