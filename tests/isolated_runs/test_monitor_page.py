from pathlib import Path
import json
import shutil
import subprocess

import pytest

from isolated_runs.monitor_page import PAGE


def test_browser_script_and_unscored_summary_values(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    path = tmp_path / "page.js"
    path.write_text(script)
    subprocess.run([node, "--check", str(path)], check=True, capture_output=True)
    function = script.split("function tableValue(", 1)[1].split("\nconst el=", 1)[0]
    # Test only the pure formatter; no browser or server is needed.
    function = "function tableValue(" + function
    cases = [
        ({"decision":"pending","train_mean_rel_steps":"0.82","train_internal_error_count":"0"}, "train_mean_rel_steps", False, "unscored"),
        ({"decision":"invalid","train_mean_rel_steps":"1000","train_internal_error_count":"1"}, "train_mean_rel_steps", False, "unscored"),
        ({"decision":"invalid","train_mean_rel_steps":"1000","train_internal_error_count":"1"}, "train_mean_rel_steps", True, "1000"),
        ({"decision":"discard","train_mean_rel_steps":"125","train_internal_error_count":"0"}, "train_mean_rel_steps", False, "125"),
        ({"decision":"invalid","train_internal_error_count":"2"}, "train_internal_error_count", False, "2"),
        ({"decision":"discard","valid_mean_rel_steps":""}, "valid_mean_rel_steps", False, "—"),
    ]
    check = function + "\nconst cases=" + json.dumps(cases) + ";\nfor(const [r,f,a,want] of cases){const got=tableValue(r,f,a);if(got!==want)throw Error(JSON.stringify({r,f,a,want,got}));}"
    subprocess.run([node, "-e", check], check=True, capture_output=True)


def test_notes_buttons_refresh_safely_and_ignore_stale_requests(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    # Exercise the actual page handlers without a browser dependency or network.
    script = script.rsplit("refresh();setInterval(refresh,5000);", 1)[0]
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.textContent='';this.scrollTop=0;this.scrollLeft=0}
 append(...children){this.children.push(...children)}
 replaceChildren(...children){this.children=children}
 setAttribute(key,value){this.attrs[key]=value}
 querySelector(tag){for(const child of this.children){if(child.tag===tag)return child;const found=child.querySelector?.(tag);if(found)return found}return null}
}
global.document={createElement:tag=>new Element(tag)};
let requests=[],reply='Persisted log.\n';
global.fetch=async(path,options)=>{requests.push(path);assert.equal(options.credentials,'omit');return {ok:true,text:async()=>reply,json:async()=>({rows:[]})}};
function content(node){return [node.textContent,...node.children.map(content)].join('\n')}
'''
    cases = r'''
(async()=>{
 const c=makeCard('trial-a',0),full=c.bar.children[3],backlog=c.bar.children[4];
 assert.equal(full.textContent,'Full log');assert.equal(backlog.textContent,'Backlog');
 full.onclick();await loadView(c);
 assert.match(content(c.view),/No readable full_log.md/);assert.equal(requests.length,0);
 c.availableLogs=['full-log','backlog'];await loadView(c);
 assert.match(content(c.view),/full_log.md is empty/);
 reply='Persisted log.\n# Progress\n<script>unsafe()</script>\n';await loadView(c);
 const pre=c.view.querySelector('pre');assert.equal(pre.textContent,'# Progress\n<script>unsafe()</script>\n');assert.equal(pre.children.length,0);
 assert.match(requests.at(-1),/log=full-log/);pre.scrollTop=42;pre.scrollLeft=7;
 await loadView(c);assert.equal(c.view.querySelector('pre'),pre);
 reply+='Fresh notes\n';await loadView(c);
 assert.equal(c.view.querySelector('pre').scrollTop,42);assert.equal(c.view.querySelector('pre').scrollLeft,7);
 backlog.onclick();await loadView(c);assert.match(requests.at(-1),/log=backlog/);
 assert.equal(full.attrs['aria-pressed'],'false');assert.equal(backlog.attrs['aria-pressed'],'true');
 backlog.onclick();assert.equal(c.selected,null);assert.equal(c.view.children.length,0);
 // Late note responses must not reopen a closed panel or overwrite another selection.
 let resolve;global.fetch=()=>new Promise(done=>resolve=done);
 full.onclick();full.onclick();resolve({ok:true,text:async()=> 'Persisted log.\nlate'});
 await new Promise(setImmediate);assert.equal(c.view.children.length,0);
 full.onclick();const resolveFull=resolve;backlog.onclick();const resolveBacklog=resolve;
 resolveBacklog({ok:true,text:async()=> 'Persisted log.\nnew backlog'});await new Promise(setImmediate);
 resolveFull({ok:true,text:async()=> 'Persisted log.\nold full log'});await new Promise(setImmediate);
 assert.equal(c.view.querySelector('pre').textContent,'new backlog');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    path = tmp_path / "notes.js"
    path.write_text(harness + script + cases)
    subprocess.run([node, str(path)], check=True, capture_output=True)


def test_energy_tooltip_is_visible_only_on_marker_interaction(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = PAGE.split("<script>", 1)[1].split("</script>", 1)[0]
    script = script.rsplit("refresh();setInterval(refresh,5000);", 1)[0]
    harness = r'''
const assert=require('node:assert/strict');
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.attrs={};this.events={};this.textContent='';this.hidden=false}
 append(...children){this.children.push(...children)}
 replaceChildren(...children){this.children=children}
 setAttribute(key,value){this.attrs[key]=value}
 addEventListener(key,handler){this.events[key]=handler}
 querySelector(selector){for(const child of this.children){if(selector.startsWith('.')?child.className===selector.slice(1):child.tag===selector)return child;const found=child.querySelector?.(selector);if(found)return found}return null}
}
global.document={createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>Object.assign(new Element('text'),{textContent:text})};
'''
    cases = r'''
const completedProgress={availability:'available',completed:460,total:460,pending:0,stale:true,stale_after_seconds:300};
assert.ok(!counts(completedProgress,true).includes('old snapshot'));
assert.ok(counts(completedProgress).includes('old snapshot'));
const historyView=evaluationDetails({evaluations:[{status:'complete',split:'train',progress:completedProgress}],recovery:[],evaluation_source:'Fixture'});
assert.equal(historyView.children.length,1);
assert.equal(historyView.children[0].tag,'details');
assert.equal(historyView.children[0].open,false);
assert.equal(historyView.children[0].children[0].textContent,'Evaluation history (1)');
const mixedView=evaluationDetails({evaluations:[{status:'pending',split:'train'},{status:'complete',split:'valid'}],recovery:[],evaluation_source:'Fixture'},true);
assert.equal(mixedView.children[0].textContent,'Unfinished evaluations');
assert.equal(mixedView.querySelector('details').open,true);
const historical=Array.from({length:128},()=>({split:'train',status:'queued'}));
const unavailable=activitySummary({evaluations:historical,gateway:{enabled:true,available:false},last_activity:null});
assert.equal(unavailable,'Live evaluation status temporarily unavailable');
const many=activitySummary({evaluations:historical,gateway:{enabled:true,available:true},last_activity:null});
assert.equal((many.match(/train: queued/g)||[]).length,2);
assert.match(many,/126 more active evaluations/);
const point={cycle:2,status:'accepted',train_cost:.9,valid_cost:.95,train_mean_rel_energy:1.000234,valid_mean_rel_energy:1.000056,
 description:'Measured completed splits',validity_reason:'valid',candidate_commit:'c',anchor_commit:'b',release_id:'fixture',train_evaluation_id:'train-fixture',valid_evaluation_id:'valid-fixture',timestamp:'2026-09-10'};
point.train_submitted_at='2026-09-08T10:01:02Z';point.valid_submitted_at='2026-09-08T10:03:04Z';
assert.ok(hoverDescription(point).includes('Submitted · training: '+submissionTime(point.train_submitted_at)));
assert.ok(hoverDescription(point).includes('Submitted · validation: '+submissionTime(point.valid_submitted_at)));
assert.equal(submissionTime(null),'unavailable');
assert.equal(submissionTime('broken'),'unavailable');
const target=new Element('section');renderPlot(target,{points:[point],champions:[point],counts:{accepted:1},warnings:[],source:'Test fixture'},'#68b0ff');
const tooltip=target.querySelector('.tooltip'),graphMarker=target.querySelector('g');
const textOf=node=>[node.textContent,...node.children.map(textOf)].join(' ');
const championText=textOf(target.querySelector('.champion-costs'));
assert.match(championText,/Current verified champion · cycle 2/);
assert.match(championText,/0.9\s+Training force-call cost \/ reference/);
assert.match(championText,/0.95\s+Validation force-call cost \/ reference/);
assert.equal(tooltip.hidden,true);assert.equal(tooltip.textContent,'');
assert.equal(graphMarker.querySelector('title'),null); // No competing browser tooltip.
assert.ok(hoverDescription({...point,description:'x'.repeat(1000)}).endsWith('x'.repeat(1000)));
assert.equal(target.querySelector('title'),null); // The parent chart must not add a native tooltip either.
assert.ok(!hoverDescription(point).includes('train-fixture'));
for(const [show,hide] of [['pointerenter','pointerleave'],['focus','blur']]){
 graphMarker.events[show]();assert.equal(tooltip.hidden,false);
 assert.match(tooltip.textContent,/Training force-call cost \/ reference: 0.9\nValidation force-call cost \/ reference: 0.95/);
 assert.match(tooltip.textContent,/Energy recovery ratio · training: 1.000234 · validation: 1.000056/);
 graphMarker.events[hide]();assert.equal(tooltip.hidden,true);assert.equal(tooltip.textContent,'');
}
graphMarker.events.pointerenter();
graphMarker.events.pointerleave({type:'pointerleave'});
assert.equal(tooltip.hidden,false); // Time to move from the marker into the scrollable card.
tooltip.events.pointerenter(); // Cancels the delayed hide.
tooltip.events.pointerleave();
assert.equal(tooltip.hidden,true);
assert.match(description({...point,train_mean_rel_energy:.97,valid_mean_rel_energy:null}),/training: 0.97 · validation: unavailable/);
assert.equal(energyRatio(.999999998),'0.999999998'); // Preserve differences near the 1e-9 energy gate.
for(const unavailable of [null,undefined,-1,NaN,Infinity])assert.equal(energyRatio(unavailable),'unavailable');
assert.equal(forceCost({...point,status:'no_improvement',valid_cost:null,valid_evaluation_id:''},'valid'),'not evaluated');
assert.equal(forceCost({...point,status:'pending',valid_cost:null,valid_evaluation_id:''},'valid'),'pending (unscored)');
assert.equal(forceCost({...point,status:'unverified',valid_cost:null,valid_evaluation_id:''},'valid'),'unscored');
assert.equal(forceCost({...point,status:'validity_reject',valid_cost:.97},'valid'),'0.97');
for(const badCost of [null,undefined,1000,-1,NaN,Infinity])assert.equal(forceCost({...point,valid_cost:badCost},'valid'),'unscored');
const later={...point,cycle:3,status:'no_improvement',train_cost:9,valid_cost:null,valid_evaluation_id:''};
renderPlot(target,{points:[point,later],champions:[point],counts:{accepted:1,no_improvement:1},warnings:[],source:'Test fixture'},'#68b0ff');
assert.equal(textOf(target.querySelector('.champion-costs')),championText); // Never use a later unaccepted candidate.
const next={...point,cycle:4,train_cost:.8,valid_cost:.85};
renderPlot(target,{points:[point,later,next],champions:[point,next],counts:{accepted:2,no_improvement:1},warnings:[],source:'Test fixture'},'#68b0ff');
assert.match(textOf(target.querySelector('.champion-costs')),/cycle 4.*0.8\s+Training.*0.85\s+Validation/);
renderPlot(target,{points:[],champions:[],counts:{},warnings:[],source:'Test fixture',message:'No verified cycles'},'#68b0ff');
assert.equal(target.querySelector('.champion-costs'),null);
'''
    path = tmp_path / "tooltip.js"
    path.write_text(harness + script + cases)
    subprocess.run([node, str(path)], check=True, capture_output=True)
