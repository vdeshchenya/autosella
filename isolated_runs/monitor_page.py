"""Self-contained operator UI: no remote assets or browser storage."""

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Autoresearch progress</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:14px system-ui,sans-serif;background:#10151c;color:#e8edf4;margin:24px auto;padding:0 22px;max-width:1420px}h1{font-size:28px;margin-bottom:8px}h2{font-size:22px;margin:0 0 6px}h3{font-size:17px;margin:20px 0 8px}p{line-height:1.5}button,select{font:inherit;background:#263346;color:inherit;border:1px solid #60738c;border-radius:6px;padding:8px;cursor:pointer}button:hover{background:#354966}button[aria-pressed=true]{border-color:#98c9ff;background:#354966}a{color:#98c9ff}small,.muted{color:#aab7c7}article{border:1px solid #354458;border-radius:12px;padding:20px;margin:22px 0}dl{display:grid;grid-template-columns:170px minmax(0,1fr);gap:8px}dt{color:#aab7c7}dd{margin:0;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto;background:#080d13;padding:16px;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #354458;vertical-align:top}th{color:#c0cee0}.table-scroll{overflow:auto;max-height:440px;border:1px solid #354458;border-radius:8px}.table-scroll table{font:12px ui-monospace,monospace}.table-scroll td{white-space:pre;max-width:600px}.table-scroll th{position:sticky;top:0;background:#1b2634;white-space:nowrap}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}.table-view:empty{display:none}.plot{background:#141d29;border:1px solid #354458;border-radius:10px;padding:12px}.plot svg{width:100%;height:auto;display:block}.legend{display:flex;gap:10px 18px;flex-wrap:wrap;font-size:12px;color:#b6c3d4;margin:4px 12px}.legend span{display:flex;gap:6px;align-items:center}.legend svg{width:25px;height:14px}.plot-note{font-size:12px;color:#b6c3d4;margin:12px}.warning,#message{color:#f5cd82}.empty{padding:25px 16px;color:#b6c3d4}.tooltip{position:fixed;z-index:10;pointer-events:auto;max-height:calc(100vh - 24px);overflow:auto;overscroll-behavior:contain;box-sizing:border-box;width:min(560px,calc(100vw - 24px));padding:12px 14px;font-size:12px;line-height:1.5;color:#e8edf4;white-space:pre-line;overflow-wrap:anywhere;border:1px solid #53657c;border-radius:8px;background:#172332;box-shadow:0 6px 20px #0005}.plot [tabindex]:focus{outline:none}.plot [tabindex]:focus-visible .marker-hit,.plot [tabindex]:hover .marker-hit{stroke:#e8edf4;stroke-width:1.2;fill:#ffffff12}.disclosure{margin-top:15px}.disclosure summary{cursor:pointer;color:#b6c3d4}.eval-scroll{overflow:auto;max-height:360px}.eval-scroll th:nth-child(-n+2),.eval-scroll td:nth-child(-n+2){white-space:nowrap}.eval-scroll td{overflow-wrap:anywhere}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid #98c9ff;outline-offset:2px}@media(max-width:640px){body{padding:0 12px;margin:18px auto}article{padding:12px}dl{grid-template-columns:120px minmax(0,1fr)}.plot{padding:4px}.toolbar{gap:6px}}
.page-header{display:flex;align-items:baseline;justify-content:space-between;gap:16px}.viewer-help{font-size:12px;color:#aab7c7}.viewer-help summary{cursor:pointer}.demo-banner{padding:12px 16px;margin:14px 0;border:1px solid #907631;border-radius:8px;background:#292519;color:#f4d58a}.run-header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap}.run-header p{margin:0;color:#b6c3d4}.run-status{margin:10px 0 0;color:#b6c3d4;font-size:12px}.run-status:empty{display:none}.gate:empty{display:none}.gate{color:#f5cd82;overflow-wrap:anywhere}.metrics{display:flex;gap:20px;align-items:baseline;flex-wrap:wrap;margin:4px 0 8px}.metrics strong{font-size:22px;font-weight:600}.metrics span{color:#aab7c7;font-size:12px}.outcome-details{font-size:12px;color:#b6c3d4;margin:10px 0}.outcome-details summary,.plot-help summary{cursor:pointer}.plot-help{font-size:12px;color:#aab7c7;margin:8px 12px}.legend{margin:12px 0}.tooltip[hidden]{display:none}.table-toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}.table-toolbar p{margin:0}.table-meta{font-size:12px;color:#aab7c7;margin:8px 0}.table-meta summary{cursor:pointer}.table-scroll td.summary-text{white-space:normal;min-width:160px;max-width:300px}.table-scroll .numeric{text-align:right;font-variant-numeric:tabular-nums;min-width:100px}.logs-section>summary{font-size:15px;color:#e8edf4}.logs-section pre{max-height:340px}.logs-section{margin-top:18px}.plot h3{margin-top:0}@media(max-width:640px){.page-header{display:block}.metrics{gap:12px}.run-header{gap:4px}.run-header p{font-size:12px}}
.champion-costs{display:flex;gap:8px 24px;align-items:baseline;flex-wrap:wrap;margin:10px 0 16px;font-size:12px;color:#aab7c7}.champion-costs strong{font-size:18px;color:#e8edf4;font-weight:600;font-variant-numeric:tabular-nums}.champion-costs p{margin:0}
.run-picker{margin:18px 0;border:1px solid #354458;border-radius:10px;padding:12px 16px}.run-picker summary{cursor:pointer;font-weight:600}.run-options{display:flex;gap:8px 18px;flex-wrap:wrap;max-height:240px;overflow:auto;margin:12px 0}.run-option{display:flex;align-items:center;gap:8px;cursor:pointer;overflow-wrap:anywhere}.run-option input{width:17px;height:17px;accent-color:#98c9ff;flex-shrink:0}.run-option input:focus-visible{outline:2px solid #98c9ff;outline-offset:3px}.run-picker .toolbar{margin-bottom:0}[hidden]{display:none!important}
</style></head><body><header class="page-header"><h1>Autoresearch progress</h1><span class="muted" title="Refreshes every 5 seconds">Updated <span id="updated">connecting</span></span></header>
<div id="demo-banner" class="demo-banner" hidden>GENERATED DATA · Demonstration only. No models or molecular calculations are running. <a href="http://127.0.0.1:8765/" target="_blank" rel="noopener">Open real runs</a></div>
<details class="viewer-help"><summary>About this viewer</summary><p><span id="gateway-mode">Checking evaluation data source…</span> Lifecycle and Docker state are separate. Known credentials are redacted; diagnostic text is untrusted. All table exports preserve the full evidence columns.</p></details>
<details id="run-picker" class="run-picker" open><summary>Displayed runs · <span id="run-count">loading</span></summary><div class="toolbar"><button id="runs-all" type="button">All</button><button id="runs-none" type="button">None</button></div><div id="run-options" class="run-options" role="group" aria-label="Displayed runs"></div></details>
<div id="message" role="status"></div><p id="run-empty" class="empty" hidden></p><main id="runs"></main><script>
const cards=new Map(), names=['results','generalizable','non_generalizable'];let busy=false,refreshAgain=false;
let availableRuns=[],displayNames={},runSelection=null,selectorReady=false,optionsSignature='';
const notes={'full-log':{label:'Full log',file:'full_log.md'},backlog:{label:'Backlog',file:'backlog.md'}};
const summaryColumns=['cycle','description','decision','validity_reason','train_mean_rel_steps','valid_mean_rel_steps','train_improvement_vs_anchor','valid_improvement_vs_anchor','train_mean_rel_energy','valid_mean_rel_energy','train_internal_error_count','valid_internal_error_count'];
const columnLabels={cycle:'Cycle',description:'Change',decision:'Decision',validity_reason:'Reason',train_mean_rel_steps:'Train cost',valid_mean_rel_steps:'Valid cost',train_improvement_vs_anchor:'Train gain',valid_improvement_vs_anchor:'Valid gain',train_mean_rel_energy:'Train energy ratio',valid_mean_rel_energy:'Valid energy ratio',train_internal_error_count:'Train errors',valid_internal_error_count:'Valid errors'};
function age(value){const seconds=Math.max(0,(Date.now()-Date.parse(value))/1000);if(!Number.isFinite(seconds))return 'unknown';if(seconds<60)return 'just now';if(seconds<3600)return Math.floor(seconds/60)+'m ago';if(seconds<86400)return Math.floor(seconds/3600)+'h ago';return Math.floor(seconds/86400)+'d ago'}
function tableValue(record,field,all){const raw=record[field]??'';if(all)return raw;if(raw==='')return '—';const split=field.startsWith('train_')?'train':field.startsWith('valid_')?'valid':null;if(split&&field!==split+'_internal_error_count'&&(record.decision==='pending'||Number(record[split+'_internal_error_count'])>0))return 'unscored';if(split&&field.endsWith('mean_rel_steps')&&Number(raw)===1000)return 'unscored';if(split&&field.endsWith('mean_rel_energy')&&Number(raw)===-1)return 'unscored';return raw}

const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const svg=(tag,attrs={})=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));return e};
const get=async(path)=>{const r=await fetch(path,{cache:'no-store',credentials:'omit'});if(!r.ok)throw Error('Local view unavailable ('+r.status+')');return r};
const query=(name,extra)=>new URLSearchParams({name,...extra});
function row(dl,label,value){dl.append(el('dt',label),el('dd',value===null||value===undefined?'unknown':String(value)))}
function counts(p,terminal=false){if(!p)return 'Counts not exposed';if(p.availability!=='available')return 'Counts '+p.availability+(p.reason?' ('+p.reason.replaceAll('_',' ')+')':'');let s=p.completed+' / '+p.total+' completed · '+p.pending+' pending';if(p.infrastructure_retries!==null&&p.infrastructure_retries!==undefined)s+=' · '+p.infrastructure_retries+' retries';if(p.stale===true&&!terminal)s+=' · old snapshot'+(p.stale_after_seconds?' (older than '+p.stale_after_seconds+' s)':'');if(p.updated_at)s+=' · updated '+p.updated_at;return s}
const labels={accepted:'Accepted experiment',anchor:'Starting anchor',no_improvement:'No cost improvement',validity_reject:'Rejected: validity gate',generalization_reject:'Rejected: generalization gate',error:'Numerical error (no score)',pending:'Pending (unscored)',unverified:'Awaiting consistent evidence'};
const colors={no_improvement:'#8c98a8',validity_reject:'#eda93a',generalization_reject:'#ef7973',error:'#9eacbc',pending:'#c4b5fd',unverified:'#f5cd82'};
function cost(v){if(v===null||v===undefined)return 'unscored';const [mantissa,exponent]=Number(v).toPrecision(7).split('e');const m=mantissa.includes('.')?mantissa.replace(/0+$/,'').replace(/\.$/,''):mantissa;return m+(exponent===undefined?'':'e'+exponent)}
function energyRatio(v){return v===null||v===undefined||!Number.isFinite(v)||v===-1?'unavailable':String(Number(v.toPrecision(12)))}
function forceCost(p,split){if(p.status==='pending')return 'pending (unscored)';if(p.status==='unverified')return 'unscored';const value=p[split+'_cost'];if(split==='valid'&&!p.valid_evaluation_id&&(value===null||value===undefined))return 'not evaluated';if(p.status==='error'||value===null||value===undefined||!Number.isFinite(value)||value<0||value===1000)return 'unscored';return cost(value)}
function description(p){return 'Cycle '+p.cycle+' · '+labels[p.status]+'\nSubmitted · training: '+submissionTime(p.train_submitted_at)+(p.valid_evaluation_id?'\nSubmitted · validation: '+submissionTime(p.valid_submitted_at):'')+'\nTraining force-call cost / reference: '+forceCost(p,'train')+'\nValidation force-call cost / reference: '+forceCost(p,'valid')+'\nEnergy recovery ratio · training: '+energyRatio(p.train_mean_rel_energy)+' · validation: '+energyRatio(p.valid_mean_rel_energy)+'\n'+p.description+'\n'+p.validity_reason+'\nCandidate: '+p.candidate_commit+' · anchor: '+p.anchor_commit+'\nRelease: '+p.release_id+'\nTrain evaluation: '+p.train_evaluation_id+'\nValid evaluation: '+(p.valid_evaluation_id||'not recorded')+'\nRecorded: '+p.timestamp}
function marker(status,x,y,color){const a={stroke:colors[status]||color,'stroke-width':1.7,fill:'none'};if(status==='error')return svg('path',{...a,d:`M${x} ${y-5}v10`});if(status==='validity_reject')return svg('path',{...a,d:`M${x-5} ${y-4}l10 0l-5 9Z`});if(status==='generalization_reject')return svg('path',{...a,d:`M${x-4} ${y-4}l8 8m0 -8l-8 8`});if(status==='anchor'||status==='pending')return svg('path',{...a,d:`M${x} ${y-6}l6 6l-6 6l-6 -6Z`});if(status==='unverified')return svg('rect',{...a,x:x-4,y:y-4,width:8,height:8});return svg('circle',{cx:x,cy:y,r:status==='accepted'?5:2.8,fill:colors[status]||color,stroke:status==='accepted'?'#141d29':'none','stroke-width':1})}
function renderPlot(target,p,color){const signature=JSON.stringify(p);if(target.dataset.signature===signature)return;target.dataset.signature=signature;const guideOpen=target.querySelector('.outcome-details')?.open,helpOpen=target.querySelector('.plot-help')?.open;target.replaceChildren();target.append(el('h3','AR progress'));
const frame=el('div');frame.className='plot';target.append(frame);const c=p.counts;
const metrics=el('div');metrics.className='metrics';
for(const [value,label]of [[p.points.filter(d=>d.status!=='anchor').length,'experiments'],[c.accepted||0,'accepted'],...(c.pending?[[c.pending,'pending']]:[]),...(c.error?[[c.error,'numerical errors']]:[]),...(c.unverified?[[c.unverified,'need evidence']]:[])]){const item=el('div');item.append(el('strong',value),document.createTextNode(' '),el('span',label));metrics.append(item)}frame.append(metrics);
const champion=p.champions.at(-1);if(champion){const summary=el('div');summary.className='champion-costs';summary.append(el('p','Current verified champion · cycle '+champion.cycle));for(const [split,label]of [['train','Training force-call cost / reference'],['valid','Validation force-call cost / reference']]){const item=el('div');item.append(el('strong',forceCost(champion,split)),document.createTextNode(' '),el('span',label));summary.append(item)}frame.append(summary)}
const guide=el('details');guide.className='outcome-details';guide.open=!!guideOpen;guide.append(el('summary','Outcomes and legend'),el('p',Object.entries(c).map(([key,n])=>n+' '+labels[key].toLowerCase()).join(' · ')||'No recorded outcomes'));
if(p.points.length){const high=Math.max(1.05,...p.champions.flatMap(d=>[d.train_cost,d.valid_cost]).filter(v=>v!==null&&Number.isFinite(v)).map(v=>v+.025));const above=p.points.filter(d=>Number.isFinite(d.train_cost)&&d.train_cost>high&&!['error','pending','unverified'].includes(d.status));const rails=[['above','Above range'],['error','Errors'],['pending','Pending'],['unverified','Unverified']].filter(([key])=>key==='above'?above.length:c[key]);const railY=Object.fromEntries(rails.map(([key],i)=>[key,18+i*20]));const W=1000,H=340,L=78,R=26,T=18+rails.length*20,B=46;const xMax=Math.max(1,...p.points.map(d=>d.cycle));const x=v=>L+(v/(xMax+0.5))*(W-L-R);const measured=p.points.flatMap(d=>[d.train_cost,...(['accepted','anchor'].includes(d.status)?[d.valid_cost]:[])]).filter(v=>v!==null&&Number.isFinite(v));const visible=measured.filter(v=>v<=high);let low=visible.length?Math.min(...visible):0;const pad=Math.max((high-low)*.1,.025);low=Math.max(0,low-pad);const y=v=>H-B-(v-low)/(high-low)*(H-B-T);const chart=svg('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':'Current-run AR progress, cost by cycle'});const text=(x,y,value,attrs={})=>{const t=svg('text',{x,y,fill:'#aab7c7','font-size':12,...attrs});t.textContent=value;chart.append(t)};
for(let i=0;i<=4;i++){const v=low+(high-low)*i/4,py=y(v);chart.append(svg('path',{d:`M${L} ${py}H${W-R}`,stroke:'#2a394d','stroke-width':1}));text(L-9,py+4,v.toFixed(high-low<.1?4:high-low<2?3:1),{'text-anchor':'end'})}for(let i=0;i<=Math.min(5,xMax);i++){const v=Math.round(i*xMax/Math.min(5,xMax));text(x(v),H-B+21,v,{'text-anchor':'middle'})}text((L+W-R)/2,H-5,'Cycle',{'text-anchor':'middle'});text(15,(T+H-B)/2,'Force-call cost / reference',{'text-anchor':'middle',transform:`rotate(-90 15 ${(T+H-B)/2})`});for(const [st,label]of rails){const py=railY[st];text(L-9,py+4,label,{'text-anchor':'end','font-size':10});chart.append(svg('path',{d:`M${L} ${py}H${W-R}`,stroke:'#263346','stroke-width':.5}))}
for(const [field,dash]of[['train_cost',null],['valid_cost','6 4']]){for(let i=0;i<p.champions.length;i++){const a=p.champions[i],b=p.champions[i+1];let end=b?b.cycle:xMax;const uncertainty=p.points.find(d=>d.status==='unverified'&&d.cycle>a.cycle&&d.cycle<=end);if(uncertainty)end=uncertainty.cycle;let d=`M${x(a.cycle)} ${y(a[field])}H${x(end)}`;if(b&&!uncertainty)d+=`V${y(b[field])}`;chart.append(svg('path',{d,stroke:color,fill:'none','stroke-width':field==='train_cost'?2.3:1.8,...(dash?{'stroke-dasharray':dash}:{})}))}}
const tooltip=el('div');tooltip.className='tooltip';tooltip.hidden=true;tooltip.setAttribute('aria-live','polite');let hideTimer;const hideTooltip=()=>{clearTimeout(hideTimer);tooltip.hidden=true;tooltip.textContent=''};tooltip.addEventListener('pointerenter',()=>clearTimeout(hideTimer));tooltip.addEventListener('pointerleave',hideTooltip);for(const d of p.points){const py=Object.hasOwn(railY,d.status)?railY[d.status]:d.train_cost>high?railY.above:y(d.train_cost);const group=svg('g',{tabindex:0,role:'graphics-symbol','aria-label':description(d)}),hit=svg('circle',{cx:x(d.cycle),cy:py,r:7,fill:'transparent',class:'marker-hit'});group.append(hit,marker(d.status,x(d.cycle),py,color));const show=(event)=>{clearTimeout(hideTimer);tooltip.hidden=false;tooltip.textContent=hoverDescription(d);tooltip.scrollTop=0;if(hit.getBoundingClientRect){const box=hit.getBoundingClientRect(),tip=tooltip.getBoundingClientRect(),px=event?.clientX??box.right,py=event?.clientY??box.bottom;tooltip.style.left=Math.max(12,Math.min(px+16,window.innerWidth-tip.width-12))+'px';tooltip.style.top=Math.max(12,Math.min(py+16,window.innerHeight-tip.height-12))+'px'}},hide=(event)=>{if(event?.type==='pointerleave'){hideTimer=setTimeout(hideTooltip,180)}else hideTooltip()};group.addEventListener('pointerenter',show);group.addEventListener('pointerleave',hide);group.addEventListener('focus',show);group.addEventListener('blur',hide);chart.append(group)}frame.append(chart);if(above.length){const note=el('p',above.length+' experiment'+(above.length===1?'':'s')+' above the plotted cost range; hover the top markers for exact values.');note.className='plot-note';frame.append(note)}const legend=el('div');legend.className='legend';for(const st of ['no_improvement','validity_reject','generalization_reject','accepted','anchor','error','pending','unverified']){const item=el('span'),icon=svg('svg',{viewBox:'0 0 25 14'});icon.append(marker(st,12,7,color));item.append(icon,document.createTextNode(labels[st]));legend.append(item)}for(const [label,dash]of[['Champion, training',null],['Champion, validation','6 4']]){const item=el('span'),icon=svg('svg',{viewBox:'0 0 25 14'});icon.append(svg('path',{d:'M0 7H25',stroke:color,'stroke-width':2,...(dash?{'stroke-dasharray':dash}:{})}));item.append(icon,document.createTextNode(label));legend.append(item)}guide.append(legend);frame.append(tooltip)}else{const empty=el('p',p.message);empty.className='empty';frame.append(empty)}
frame.append(guide);const help=el('details');help.className='plot-help';help.open=!!helpOpen;help.append(el('summary','How to read this chart'));const note=el('p',p.source+'. Lower cost ratio is better. Starting anchors are separate from accepted experiments; pending records and error penalties have no plotted score.');note.className='plot-note';help.append(note);frame.append(help);if(p.warnings.length){const details=el('details'),sum=el('summary',p.warnings.length+' data notice'+(p.warnings.length===1?'':'s'));details.className='disclosure warning';details.append(sum,...p.warnings.map(w=>el('p',w)));frame.append(details)}}

function submissionTime(value){
 if(!value)return 'unavailable';
 const date=new Date(value);if(!Number.isFinite(date.getTime()))return 'unavailable';
 return new Intl.DateTimeFormat('en-GB',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',timeZoneName:'short'}).format(date);
}
function hoverDescription(p){
 const note=p.description||'';
 return 'Cycle '+p.cycle+' · '+labels[p.status]+'\nSubmitted · training: '+submissionTime(p.train_submitted_at)+(p.valid_evaluation_id?'\nSubmitted · validation: '+submissionTime(p.valid_submitted_at):'')+'\nTraining force-call cost / reference: '+forceCost(p,'train')+'\nValidation force-call cost / reference: '+forceCost(p,'valid')+'\nEnergy recovery ratio · training: '+energyRatio(p.train_mean_rel_energy)+' · validation: '+energyRatio(p.valid_mean_rel_energy)+(note?'\n\n'+note:'');
}

function evaluationDetails(r,historyOpen=false){
 const section=el('div'),items=[...r.evaluations,...r.recovery.map(x=>({...x,split:'recovery'}))];
 const isActive=j=>['running','pending','queued','awaiting recovery'].includes(j.status);
 const active=items.filter(isActive),past=items.filter(j=>!isActive(j));
 const table=rows=>{
  const wrapper=el('div'),t=el('table'),head=el('tr');wrapper.className='eval-scroll';
  for(const label of ['Split','Status','Evaluation','Molecule outcomes'])head.append(el('th',label));t.append(head);
  for(const j of rows){const tr=el('tr');for(const v of [j.split||'',j.status,j.evaluation_id||'',counts(j.progress,j.status==='complete')+(j.infrastructure_error?' · infrastructure error recorded':'')])tr.append(el('td',v));t.append(tr)}
  wrapper.append(t);return wrapper;
 };
 if(active.length){section.append(el('p','Unfinished evaluations'),table(active))}
 if(past.length){const history=el('details');history.className='evaluation-history disclosure';history.open=historyOpen;history.append(el('summary','Evaluation history ('+past.length+')'),el('p',r.evaluation_source),table([...past].reverse()));section.append(history)}
 return section;
}

function activitySummary(r){
 const parts=[];
 if(r.gateway?.enabled&&!r.gateway.available){
  parts.push('Live evaluation status temporarily unavailable');
  if(r.gateway.observed_at)parts.push('Last gateway update '+age(r.gateway.observed_at));
 }else{
  const active=r.evaluations.filter(j=>['running','pending','queued'].includes(j.status));
  for(const j of active.slice(-2))parts.push((j.split||'Evaluation')+': '+(j.progress?.availability==='available'?j.progress.completed+'/'+j.progress.total+' outcomes · '+j.progress.pending+' pending':j.status));
  if(active.length>2)parts.push((active.length-2)+' more active evaluations (see run details)');
 }
 if(r.last_activity)parts.push('Last saved activity '+age(r.last_activity));
 return parts.join(' · ');
}

function makeCard(name,index){
 const article=el('article');article.dataset.run=name;
 const heading=el('h2',displayNames[name]||name),header=el('div'),execution=el('p'),brief=el('p'),gate=el('p'),activity=el('p'),meta=el('div'),operations=el('details'),plot=el('section'),bar=el('div'),view=el('section'),logs=el('details');
 header.className='run-header';header.append(heading,brief);activity.className='run-status';gate.className='gate';bar.className='toolbar';view.className='table-view';operations.className='disclosure';logs.className='disclosure logs-section';
 operations.append(el('summary','Run details and evaluation status'),execution,meta);article.append(header,plot,activity,gate,bar,view,operations,logs);
 const c={name,heading,article,execution,brief,gate,activity,meta,plot,bar,view,logs,selected:null,allEvidence:false,tableSignature:null,tableRequest:0,availableLogs:[],color:index%2?'#46cfa4':'#68b0ff'};
 for(const key of [...names,...Object.keys(notes)]){const b=el('button',notes[key]?.label||key+'.tsv');b.type='button';b.setAttribute('aria-pressed','false');if(notes[key])b.title=notes[key].file;b.onclick=()=>{c.selected=c.selected===key?null:key;c.tableSignature=null;for(const button of bar.children)button.setAttribute('aria-pressed',String(button===b&&c.selected===key));view.replaceChildren();if(c.selected)loadView(c);else c.tableRequest++};bar.append(b)}
 const access=el('p'),controls=el('div'),sel=el('select'),link=el('a','Open log'),pre=el('pre');controls.className='toolbar';sel.setAttribute('aria-label','Log for '+name);link.target='_blank';link.rel='noopener';controls.append(sel,link);logs.append(el('summary','Logs'),access,controls,pre);Object.assign(c,{logAccess:access,logControls:controls,sel,link,pre});sel.onchange=()=>loadLog(c);logs.ontoggle=()=>{if(logs.open)loadLog(c)};return c
}
function renderTable(c,t){
 const old=c.view.querySelector('.table-scroll'),scroll=old?[old.scrollLeft,old.scrollTop]:[0,0];
 c.view.replaceChildren();const bar=el('div');bar.className='table-toolbar';bar.append(el('p',t.name+' · '+t.rows.length+' rows'+(t.availability!=='available'?' · '+t.availability:'')));
 const toggle=el('button',c.allEvidence?'Summary columns':'All evidence (51 columns)');toggle.type='button';toggle.setAttribute('aria-pressed',String(c.allEvidence));toggle.onclick=()=>{c.allEvidence=!c.allEvidence;renderTable(c,t)};if(t.columns.length)bar.append(toggle);
 if(t.raw!==null){const download=el('a','Download TSV');download.href='/table?'+query(c.name,{table:c.selected,download:'1'});bar.append(download)}c.view.append(bar);
 const meta=el('details');meta.className='table-meta';meta.append(el('summary','File details'),el('p',t.availability+' · '+t.source+(t.size_bytes!==null?' · '+t.size_bytes+' bytes':'')+(t.modified_at?' · modified '+t.modified_at:'')));
 if(t.raw!==null){const raw=el('a','Open raw TSV');raw.href='/table?'+query(c.name,{table:c.selected});raw.target='_blank';raw.rel='noopener';meta.append(raw)}c.view.append(meta);
 for(const warning of t.warnings){const p=el('p',warning);p.className='warning';c.view.append(p)}
 if(!t.columns.length){c.view.append(el('p',t.availability==='missing'?'This run has not written this table yet.':'No parsed table is available.'));return}
 const columns=c.allEvidence?t.columns:summaryColumns.filter(field=>t.columns.includes(field));
 const wrapper=el('div'),table=el('table'),thead=el('thead'),head=el('tr'),tbody=el('tbody');wrapper.className='table-scroll';wrapper.tabIndex=0;wrapper.setAttribute('aria-label',c.selected+' table, scroll for all columns and records');
 for(const field of columns){const th=el('th',c.allEvidence?field:(columnLabels[field]||field));th.scope='col';th.title=field;head.append(th)}thead.append(head);
 for(const record of t.rows){const tr=el('tr');for(const field of columns){const td=el('td',tableValue(record,field,c.allEvidence));td.title=record[field]||'';if(!c.allEvidence)td.className=['description','validity_reason'].includes(field)?'summary-text':field.startsWith('train_')||field.startsWith('valid_')?'numeric':'';tr.append(td)}tbody.append(tr)}
 table.append(thead,tbody);wrapper.append(table);c.view.append(wrapper);wrapper.scrollLeft=scroll[0];wrapper.scrollTop=scroll[1]
}
function renderNote(c,n){
 const old=c.view.querySelector('pre'),scroll=old?[old.scrollLeft,old.scrollTop]:[0,0];c.view.replaceChildren();
 const bar=el('div');bar.className='table-toolbar';bar.append(el('p',notes[c.selected].file));
 if(n.text!==null){const link=el('a','Open raw notes');link.href='/log?'+query(c.name,{log:c.selected});link.target='_blank';link.rel='noopener';bar.append(link)}c.view.append(bar);
 if(n.text===null){c.view.append(el('p','No readable '+notes[c.selected].file+' in this view yet. This panel refreshes automatically as the run writes notes.'));return}
 const boundary=n.text.indexOf('\n'),heading=boundary<0?'':n.text.slice(0,boundary),body=boundary<0?n.text:n.text.slice(boundary+1);
 const meta=el('p',heading+' Refreshes every 5 seconds.');meta.className='table-meta';c.view.append(meta);
 if(!body.trim()){c.view.append(el('p',notes[c.selected].file+' is empty. Notes will appear here as the run writes them.'));return}
 const pre=el('pre',body);pre.tabIndex=0;pre.setAttribute('aria-label',notes[c.selected].label+' notes');c.view.append(pre);pre.scrollLeft=scroll[0];pre.scrollTop=scroll[1]
}
async function loadView(c){
 if(c.displayed===false)return;
 const selected=c.selected,id=++c.tableRequest;if(!selected)return;
 try{const note=Object.hasOwn(notes,selected),t=note?{text:c.availableLogs.includes(selected)?await(await get('/log?'+query(c.name,{log:selected}))).text():null}:await(await get('/api/table?'+query(c.name,{table:selected}))).json();if(c.selected!==selected||id!==c.tableRequest)return;const signature=JSON.stringify(t);if(signature===c.tableSignature)return;c.tableSignature=signature;if(note)renderNote(c,t);else renderTable(c,t)}
 catch(e){if(c.selected===selected&&id===c.tableRequest){c.view.replaceChildren(el('p',e.message));c.tableSignature=null}}
}
async function loadLog(c){if(c.displayed===false||!c.sel.value)return;const key=c.sel.value,revision=c.visibilityRevision;c.link.href='/log?'+query(c.name,{log:key});try{const text=await(await get(c.link.href)).text();if(c.displayed===false||c.visibilityRevision!==revision||c.sel.value!==key)return;if(c.pre.textContent!==text){const [left,top]=[c.pre.scrollLeft,c.pre.scrollTop];c.pre.textContent=text;c.pre.scrollLeft=left;c.pre.scrollTop=top}}catch(e){if(c.displayed!==false&&c.visibilityRevision===revision)c.pre.textContent=e.message}}
async function updateCard(c,r){c.heading.textContent=r.display_name||r.name||c.name;c.execution.textContent=r.execution;c.brief.textContent=r.model+' · '+r.lifecycle;c.brief.title=r.provider+' · effort '+r.effort;
c.gate.textContent=/blocked|invalid|error/i.test(r.gate)?r.gate:'';
c.activity.textContent=activitySummary(r);c.activity.title=r.last_activity||'';const frag=document.createDocumentFragment(),dl=el('dl');row(dl,'Reasoning effort',r.effort);row(dl,'Run identity',r.run_id);row(dl,'Container',r.container.status+(r.container.exit_code!==null&&r.container.exit_code!==undefined?' (exit '+r.container.exit_code+')':''));row(dl,'Current candidate',r.current_candidate);row(dl,'Last saved activity',r.last_activity);if(!c.gate.textContent)row(dl,'Research gate',r.gate);row(dl,'Starting anchors',r.anchors_ready?'Both ready; same release':r.anchors_state==='not_started'?'Not started':r.anchors_state==='unavailable'?'Not recorded in verified export':r.anchors_state==='invalid'?'Invalid; research blocked':'Pending');for(const split of ['train','valid']){const a=r.anchors[split];row(dl,split+' anchor',a.evaluation_id?a.status+' · '+a.evaluation_id:a.status==='unavailable'?'Not recorded in verified export':a.status==='submitting'?'Submitting (ID not confirmed)':a.started?'No saved anchor result':'Not started');if(a.is_valid===0){let why=a.invalid_reason.replaceAll('_',' ');if(a.mean_rel_energy!==undefined)why+=' · mean relative energy '+a.mean_rel_energy;if(a.internal_error_count!==undefined)why+=' · '+a.internal_error_count+' internal errors';row(dl,split+' invalid reason',why)}if(a.progress)row(dl,split+' progress',counts(a.progress,a.status==='complete'))}if(r.ralph_iteration!==null)row(dl,'Ralph iteration',r.ralph_iteration);if(r.purpose==='verification'){row(dl,'Provider invocations',r.provider_invocations);row(dl,'Verification complete',r.verification_complete?'yes (local evidence)':'not recorded')}if(r.gateway.available)row(dl,'Gateway state',r.gateway.state+' · '+r.gateway.observed_at);frag.append(dl,el('small',r.last_activity_source));if(r.evaluations.length||r.recovery.length)frag.append(evaluationDetails(r,!!c.meta.querySelector('.evaluation-history')?.open));c.meta.replaceChildren(frag);renderPlot(c.plot,r.research_progress,c.color);for(const [i,key]of names.entries()){const t=r.tables[key];c.bar.children[i].title=t.availability+' · '+t.row_count+' complete rows'}c.logAccess.textContent=r.log_access;c.availableLogs=r.logs;const visibleLogs=r.logs.filter(key=>!names.includes(key));const current=c.sel.value,logSignature=JSON.stringify(visibleLogs);if(c.logSignature!==logSignature){c.logSignature=logSignature;c.sel.replaceChildren();for(const key of visibleLogs){const o=el('option',key==='docker'?'Live Docker stdout / stderr':key);o.value=key;c.sel.append(o)}if(visibleLogs.includes(current))c.sel.value=current;else if(visibleLogs.includes('docker'))c.sel.value='docker';else if(visibleLogs.length)c.sel.value=visibleLogs.at(-1)}c.logControls.hidden=!visibleLogs.length;c.pre.hidden=!visibleLogs.length;if(!visibleLogs.length)c.logAccess.textContent+=' · No readable logs in this view.';await Promise.all([c.logs.open?loadLog(c):Promise.resolve(),c.selected?loadView(c):Promise.resolve()])}
function readRunSelection(){
 const params=new URLSearchParams(window.location.hash.slice(1));
 if(!params.has('runs'))return null;
 try{const selected=JSON.parse(params.get('runs'));return Array.isArray(selected)&&selected.every(name=>typeof name==='string')?new Set(selected):null}catch{return null}
}
function runIsSelected(name){return runSelection===null||runSelection.has(name)}
function refreshSelection(){refreshAgain=true;void refresh()}
function setRunSelection(selected){
 runSelection=selected;
 const params=new URLSearchParams(window.location.hash.slice(1));
 if(selected===null)params.delete('runs');else params.set('runs',JSON.stringify([...selected]));
 const hash=params.toString();window.history.replaceState(null,'',window.location.pathname+window.location.search+(hash?'#'+hash:''));
 displaySelectedRuns();refreshSelection()
}
function initializeRunSelector(){
 if(selectorReady)return;selectorReady=true;runSelection=readRunSelection();
 document.getElementById('runs-all').onclick=()=>setRunSelection(null);
 document.getElementById('runs-none').onclick=()=>setRunSelection(new Set());
 window.addEventListener('hashchange',()=>{runSelection=readRunSelection();displaySelectedRuns();refreshSelection()})
}
function displaySelectedRuns(){
 const options=document.getElementById('run-options'),signature=JSON.stringify([availableRuns,displayNames]);
 if(signature!==optionsSignature){
  optionsSignature=signature;options.replaceChildren();
  for(const name of availableRuns){
   const label=el('label'),input=el('input');label.className='run-option';input.type='checkbox';input.value=name;
   input.onchange=()=>{const selected=new Set(runSelection===null?availableRuns:runSelection);if(input.checked)selected.add(name);else selected.delete(name);setRunSelection(selected)};
   label.append(input,el('span',displayNames[name]||name));options.append(label)
  }
 }
 for(const label of options.children)label.firstElementChild.checked=runIsSelected(label.firstElementChild.value);
 const main=document.getElementById('runs');let reordered=false;
 for(const [name,c]of cards)if(!availableRuns.includes(name)){c.displayed=false;c.visibilityRevision++;c.tableRequest++;c.article.remove();cards.delete(name)}
 for(const [index,name]of availableRuns.entries()){
  let c=cards.get(name);const displayed=runIsSelected(name);
  if(!c&&displayed){c=makeCard(name,index);c.visibilityRevision=0;cards.set(name,c);reordered=true}
  if(!c)continue;
  if(c.displayed!==displayed){c.displayed=displayed;c.visibilityRevision++;c.tableRequest++}
  c.article.hidden=!displayed
 }
 if(reordered)for(const name of availableRuns){const c=cards.get(name);if(c)main.append(c.article)}
 const count=availableRuns.filter(runIsSelected).length;
 document.getElementById('run-count').textContent=count+' of '+availableRuns.length;
 document.getElementById('runs-all').disabled=!availableRuns.length;
 document.getElementById('runs-none').disabled=!availableRuns.length;
 const empty=document.getElementById('run-empty');empty.hidden=!!count;
 empty.textContent=availableRuns.length?'No runs selected. Choose runs above to display them.':'No recognized runs in the configured private root.'
}
async function refresh(){
 if(busy)return;busy=true;refreshAgain=false;
 try{
  initializeRunSelector();
  const data=await(await get('/api/runs')).json();
  document.getElementById('gateway-mode').textContent=data.gateway_status_enabled?'Gateway ledger polling is enabled (every 30 seconds); snapshots show their observation time.':'Gateway polling is disabled; evaluations show saved local receipts.';
  document.getElementById('demo-banner').hidden=!data.demo;
  availableRuns=data.runs;displayNames=data.display_names||{};displaySelectedRuns();
  const outcomes=await Promise.allSettled(availableRuns.filter(runIsSelected).map(async name=>{
   const c=cards.get(name),revision=c.visibilityRevision;
   const current=()=>cards.get(name)===c&&c.displayed&&c.visibilityRevision===revision;
   try{const r=await(await get('/api/run?'+query(name,{}))).json();if(current())await updateCard(c,r)}catch(e){return {error:e,current}}
  }));
  document.getElementById('message').textContent=outcomes.flatMap(r=>r.status==='rejected'?[r.reason.message]:r.value?.current()?[r.value.error.message]:[]).join(' · ');
  document.getElementById('updated').textContent=new Date().toLocaleTimeString()
 }catch(e){document.getElementById('message').textContent=e.message}
 finally{busy=false;if(refreshAgain)void refresh()}
}
refresh();setInterval(refresh,5000);
</script></body></html>'''
