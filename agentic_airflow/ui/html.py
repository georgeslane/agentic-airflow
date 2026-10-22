"""Self-contained HTML renderer for the DAG graph (no external assets).

:func:`render_html` returns a full page; :func:`render_embed` returns a
fragment (``<div>`` + ``<script>``) that mounts into any host application.
All data is JSON-embedded with ``</`` escaped and rendered through
``textContent``, so model outputs can never inject markup.
"""

from __future__ import annotations

import html
import json
import secrets
from typing import Any

_CSS = """
:root{--bg:#f7f7f8;--panel:#fff;--fg:#1c1c1e;--muted:#6b6b70;--line:#c9c9ce;--edge:#9a9aa0;
--pending:#d7d7dc;--queued:#c5d6f2;--running:#5b8def;--success:#43a86d;--failed:#d9534f;--skipped:#ededf0;
--waiting:#f0a63a;--upstream_failed:#e9a19f;--accent:#5b8def;--evaluator:#f3eefc}
@media (prefers-color-scheme:dark){:root{--bg:#141417;--panel:#1e1e23;--fg:#ececf1;--muted:#a0a0a8;--line:#3a3a42;--edge:#6c6c76;
--pending:#3a3a42;--queued:#2c3f63;--skipped:#2a2a30;--evaluator:#2b2535}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg)}
.aa-root{display:grid;grid-template-columns:minmax(0,1fr) 340px;grid-template-rows:auto minmax(0,1fr);height:100vh}
.aa-header{grid-column:1/3;display:flex;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);background:var(--panel)}
.aa-header h1{font-size:16px;margin:0;font-weight:600}.aa-header .aa-status{padding:2px 8px;border-radius:999px;font-size:12px;color:#fff;background:var(--pending)}
.aa-header .aa-meta{color:var(--muted);font-size:12px;margin-left:auto}
.aa-canvas{overflow:auto;padding:16px}.aa-canvas svg{display:block;min-width:100%}
.aa-panel{border-left:1px solid var(--line);background:var(--panel);padding:14px;overflow:auto}
.aa-panel h2{font-size:14px;margin:0 0 8px}.aa-panel h3{font-size:12px;margin:14px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.aa-panel pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);padding:8px;border-radius:6px;font-size:12px;max-height:260px;overflow:auto;margin:0}
.aa-node{cursor:pointer}.aa-node rect{stroke:var(--line);stroke-width:1.2;rx:8}.aa-node.selected rect{stroke:var(--accent);stroke-width:2.5}
.aa-node text{font-size:12px;fill:var(--fg);pointer-events:none}.aa-node .aa-kind{font-size:10px;fill:var(--muted)}
.aa-edge{fill:none;stroke:var(--edge);stroke-width:1.4}.aa-arrow{fill:var(--edge)}
.aa-legend{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:var(--muted);margin-top:8px}.aa-legend span i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px}
.aa-badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;color:#fff;background:var(--pending);margin-right:4px}
.aa-inst{padding:6px 0;border-top:1px solid var(--line)}.aa-inst small{color:var(--muted)}
@media (max-width:720px){.aa-root{grid-template-columns:1fr;grid-template-rows:auto minmax(0,1fr) 40vh}.aa-header{grid-column:1}.aa-panel{border-left:0;border-top:1px solid var(--line)}}
"""

_JS = r"""
(function(){
  const COLORS={pending:'var(--pending)',queued:'var(--queued)',running:'var(--running)',success:'var(--success)',failed:'var(--failed)',
    skipped:'var(--skipped)',upstream_failed:'var(--upstream_failed)',awaiting_tool_approval:'var(--waiting)',awaiting_user:'var(--waiting)'};
  const TEXT_ON_DARK=new Set(['running','success','failed','awaiting_tool_approval','awaiting_user']);
  const cfg=window.__AA_CONFIG__;const root=document.getElementById(cfg.containerId);
  let graph=cfg.graph, selected=null;
  const el=(tag,attrs,parent)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(parent)parent.appendChild(n);return n;};
  const h=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
  const fmtTime=t=>t?new Date(t*1000).toLocaleTimeString():'';
  const dur=(a,b)=>a&&b?((b-a).toFixed(1)+'s'):'';
  function build(){
    root.textContent='';root.className='aa-root';
    const header=h('div','aa-header');const title=h('h1',null,graph.dag.id);header.appendChild(title);
    const st=h('span','aa-status',graph.run?graph.run.status:'definition');st.style.background=graph.run?(COLORS[graph.run.status==='paused'?'awaiting_user':graph.run.status==='running'?'running':graph.run.status==='success'?'success':graph.run.status==='failed'?'failed':'pending']):'var(--pending)';header.appendChild(st);
    const meta=h('span','aa-meta',graph.run?('run '+graph.run.run_id+(graph.run.started_at?' · started '+fmtTime(graph.run.started_at):'')):(graph.dag.schedule.length?'schedule: '+graph.dag.schedule.map(s=>s.expression||s.name||s.dag_id).join(', '):'no schedule'));
    header.appendChild(meta);root.appendChild(header);
    const canvas=h('div','aa-canvas');root.appendChild(canvas);
    const W=140,H=48;const svg=el('svg',{viewBox:`0 0 ${graph.layout.width+40} ${graph.layout.height+40}`,width:graph.layout.width+40,height:graph.layout.height+40});
    const defs=el('defs',{},svg);const m=el('marker',{id:'aa-arrow',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'7',markerHeight:'7',orient:'auto-start-reverse'},defs);el('path',{d:'M0,0 L10,5 L0,10 z','class':'aa-arrow'},m);
    const pos={};graph.nodes.forEach(n=>pos[n.id]=n.position);
    graph.edges.forEach(e=>{const a=pos[e.source],b=pos[e.target];if(!a||!b)return;const x1=a.x+W+20,y1=a.y+H/2+20,x2=b.x+20,y2=b.y+H/2+20;const c=(x2-x1)/2;
      el('path',{d:`M${x1},${y1} C${x1+c},${y1} ${x2-c},${y2} ${x2},${y2}`,'class':'aa-edge','marker-end':'url(#aa-arrow)'},svg);});
    graph.nodes.forEach(n=>{const g=el('g',{'class':'aa-node'+(selected===n.id?' selected':''),transform:`translate(${n.position.x+20},${n.position.y+20})`},svg);
      const status=n.status||'pending';const r=el('rect',{width:W,height:H,fill:n.evaluator_group&&!n.status?'var(--evaluator)':COLORS[status]||COLORS.pending},g);
      if(n.evaluator_group&&n.status)r.setAttribute('stroke-dasharray','4 3');
      const label=el('text',{x:10,y:20},g);label.textContent=n.id.length>20?n.id.slice(0,19)+'…':n.id;if(TEXT_ON_DARK.has(status))label.setAttribute('fill','#fff');
      const kind=el('text',{x:10,y:36,'class':'aa-kind'},g);kind.textContent=n.kind.replace('agentic_','🤖 ')+(n.fan_out?' ⇶':'')+(n.instances&&n.instances.length>1?' ×'+n.instances.length:'');if(TEXT_ON_DARK.has(status))kind.setAttribute('fill','rgba(255,255,255,.85)');
      g.addEventListener('click',()=>{selected=n.id;build();});});
    canvas.appendChild(svg);
    const legend=h('div','aa-legend');['pending','running','success','failed','skipped','awaiting_tool_approval'].forEach(s=>{const sp=h('span');const i=h('i');i.style.background=COLORS[s];sp.appendChild(i);sp.appendChild(document.createTextNode(s.replace(/_/g,' ')));legend.appendChild(sp);});canvas.appendChild(legend);
    const panel=h('div','aa-panel');root.appendChild(panel);
    const node=graph.nodes.find(n=>n.id===selected);
    if(!node){panel.appendChild(h('h2',null,'Workflow objective'));const p=h('pre',null,graph.dag.prompt);panel.appendChild(p);
      if(graph.run&&graph.run.summary){panel.appendChild(h('h3',null,'Executive summary'));panel.appendChild(h('pre',null,graph.run.summary));}
      if(graph.run&&graph.run.error){panel.appendChild(h('h3',null,'Run error'));panel.appendChild(h('pre',null,graph.run.error));}
      panel.appendChild(h('h3',null,'Hint'));panel.appendChild(h('div',null,'Click a node to inspect its configuration, instances and outputs.'));return;}
    panel.appendChild(h('h2',null,node.id));const badges=h('div');[node.kind,node.trigger_rule,node.fan_out?'fan_out':null,node.follow_mapping?'follow_mapping':null,node.group?'group: '+node.group:null].filter(Boolean).forEach(b=>badges.appendChild(h('span','aa-badge',b)));panel.appendChild(badges);
    if(node.prompt){panel.appendChild(h('h3',null,'Prompt'));panel.appendChild(h('pre',null,node.prompt));}
    panel.appendChild(h('h3',null,'Configuration'));panel.appendChild(h('pre',null,JSON.stringify(Object.fromEntries(Object.entries(node.config).filter(([k,v])=>v!==null&&v!==true&&v!==false||k==='inherit_skills'&&v===false)),null,1)));
    if(node.instances&&node.instances.length){panel.appendChild(h('h3',null,'Instances ('+node.instances.length+')'));node.instances.forEach(i=>{const d=h('div','aa-inst');const b=h('span','aa-badge',i.status.replace(/_/g,' '));b.style.background=COLORS[i.status]||COLORS.pending;d.appendChild(b);d.appendChild(document.createTextNode(i.id+' '));
      const s=h('small',null,[i.attempt>1?'attempt '+i.attempt:null,i.steps?i.steps+' steps':null,i.tool_calls?i.tool_calls+' tool calls':null,dur(i.started_at,i.finished_at)].filter(Boolean).join(' · '));d.appendChild(s);
      if(i.selected)d.appendChild(h('div',null,'selected → '+i.selected.join(', ')));if(i.reason)d.appendChild(h('div',null,i.reason));if(i.error){const e=h('pre',null,i.error);e.style.color='var(--failed)';d.appendChild(e);}if(i.output){d.appendChild(h('pre',null,i.output));}panel.appendChild(d);});}
  }
  build();
  if(cfg.refreshUrl){const tick=async()=>{try{const headers={};if(cfg.token)headers['Authorization']='Bearer '+cfg.token;const r=await fetch(cfg.refreshUrl,{headers});if(r.ok){graph=await r.json();build();}}catch(e){}
      if(!graph.run||['running','paused','pending'].includes(graph.run.status))setTimeout(tick,cfg.refreshMs||2000);else setTimeout(tick,(cfg.refreshMs||2000)*5);};setTimeout(tick,cfg.refreshMs||2000);}
})();
"""


def _json_for_script(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/").replace("<!--", "<\\!--")


def render_embed(graph: dict[str, Any], *, container_id: str = "agentic-airflow-graph", refresh_url: str | None = None, refresh_ms: int = 2000, token: str | None = None, nonce: str | None = None) -> str:
    """Return an HTML fragment that renders ``graph`` inside a host page."""
    config = {"containerId": container_id, "graph": graph, "refreshUrl": refresh_url, "refreshMs": refresh_ms, "token": token}
    nonce_attr = f' nonce="{html.escape(nonce)}"' if nonce else ""
    return (
        f'<div id="{html.escape(container_id)}"></div>\n'
        f'<style{nonce_attr}>{_CSS}</style>\n'
        f'<script{nonce_attr}>window.__AA_CONFIG__={_json_for_script(config)};</script>\n'
        f'<script{nonce_attr}>{_JS}</script>'
    )


def render_html(graph: dict[str, Any], *, title: str | None = None, refresh_url: str | None = None, refresh_ms: int = 2000, token: str | None = None, nonce: str | None = None) -> str:
    """Return a complete, self-contained HTML page for ``graph``."""
    nonce = nonce or secrets.token_urlsafe(12)
    title = title or f"{graph.get('dag', {}).get('id', 'DAG')} · agentic-airflow"
    csp = f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<meta http-equiv=\"Content-Security-Policy\" content=\"{html.escape(csp)}\">\n"
        f"<title>{html.escape(title)}</title>\n</head>\n<body>\n"
        + render_embed(graph, refresh_url=refresh_url, refresh_ms=refresh_ms, token=token, nonce=nonce)
        + "\n</body>\n</html>\n"
    )
