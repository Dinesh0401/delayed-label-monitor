// Build embeds Python-trained parameters and UI; D1 stores records across sessions.
export function probability(x, model) {
  const z=model.intercept+x.reduce((s,v,i)=>s+v*model.coefficients[i],0);
  return 1/(1+Math.exp(-z));
}
function fail(code,detail){throw {code,detail}}
function timestamp(value){if(typeof value!=='string'||!/(Z|[+-]\d\d:\d\d)$/.test(value)||!Number.isFinite(Date.parse(value)))fail(422,'Use a timezone-aware timestamp');return new Date(value).toISOString()}
export function normalizeOrder(b,model){
  if(typeof b.id!=='string'||!/^[A-Za-z0-9_-]{1,80}$/.test(b.id))fail(422,'Order ID must contain 1–80 letters, numbers, underscores or hyphens');
  const purchased=timestamp(b.purchased),due=timestamp(b.due),d=new Date(purchased),days=(Date.parse(due)-d.getTime())/86400000;
  if(!model.states.includes(b.state)||!(days>0&&days<=180))fail(422,'Choose a supported state and a deadline within 180 days after purchase');
  if(d.getTime()>Date.now())fail(422,'Purchase cannot be in the future');
  const values={promise_days:days,dow:(d.getUTCDay()+6)%7,month:d.getUTCMonth()+1,hour:d.getUTCHours(),['customer_state_'+b.state]:1};
  const p=probability(model.features.map(f=>values[f]||0),model);
  return {id:b.id,state:b.state,purchased,due,probability:p,predicted:Number(p>=model.threshold),model_version:model.version,created:new Date().toISOString()};
}
export function createHandler(model,html,study,research,study2='Research report unavailable'){
 const json=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
 return {async fetch(request,env){
  const url=new URL(request.url),path=url.pathname;
  try {
   if(request.method==='GET'&&path==='/')return new Response(html,{headers:{'Content-Type':'text/html;charset=utf-8'}});
   if(request.method==='GET'&&(path==='/delivery-study.html'||path==='/delivery-study'))return new Response(study,{headers:{'Content-Type':'text/html;charset=utf-8'}});
   if(request.method==='GET'&&path==='/research-study.html')return new Response(study2,{headers:{'Content-Type':'text/html;charset=utf-8'}});
   if(request.method==='GET'&&path==='/research.json')return json(research);
   if(request.method==='GET'&&path==='/api/model'){const {coefficients,intercept,features,...card}=model;return json(card)}
   if(!path.startsWith('/api/'))return json({detail:'Not found'},404);
   if(!env.DB)fail(503,'Persistent database is not configured');
   if(request.method==='POST'){
    const origin=request.headers.get('Origin');if(origin&&origin!==url.origin)fail(403,'Cross-origin writes are not allowed');
    if(!request.headers.get('Content-Type')?.startsWith('application/json'))fail(415,'Send JSON');
    const raw=await request.text();if(raw.length>65536)fail(413,'Request is too large');let b;try{b=JSON.parse(raw)}catch{fail(400,'Invalid JSON')}
    if(path==='/api/batch-orders'){
     if(!Array.isArray(b))fail(422,'Expected an array of orders');
     let inserted=0;
     for(const item of b){
      try{
       const o=normalizeOrder(item,model);
       const res=await env.DB.prepare('INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)').bind(o.id,o.state,o.purchased,o.due,o.probability,o.predicted,o.model_version,o.created).run();
       const changes=res?.changes??res?.meta?.changes??0;
       if(changes>0)inserted++;
      }catch(_){}
     }
     return json({received:b.length,inserted});
    }
    if(path==='/api/batch-outcomes'){
     if(!Array.isArray(b))fail(422,'Expected an array of outcomes');
     let updated=0;const now=Date.now(),nowIso=new Date().toISOString();
     for(const item of b){
      try{
       if(typeof item.id!=='string')continue;
       const delivered=timestamp(item.delivered);
       if(Date.parse(delivered)>now)continue; // Strictly reject future deliveries
       const row=await env.DB.prepare('SELECT purchased,delivered FROM orders WHERE id=?').bind(item.id).first();
       if(!row||row.delivered||Date.parse(delivered)<Date.parse(row.purchased))continue;
       await env.DB.prepare('UPDATE orders SET delivered=?,reported=? WHERE id=? AND delivered IS NULL').bind(delivered,nowIso,item.id).run();
       updated++;
      }catch(_){}
     }
     return json({received:b.length,updated});
    }
    if(!b||typeof b!=='object')fail(422,'Expected an object');
    if(path==='/api/orders'){
     const o=normalizeOrder(b,model);
     await env.DB.prepare('INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,NULL,NULL)').bind(o.id,o.state,o.purchased,o.due,o.probability,o.predicted,o.model_version,o.created).run();
     const row=await env.DB.prepare('SELECT * FROM orders WHERE id=?').bind(o.id).first();
     if(row.state!==o.state||row.purchased!==o.purchased||row.due!==o.due)fail(409,'Order ID already exists with different input');
     return json(row);
    }
    if(path==='/api/outcomes'){
     if(typeof b.id!=='string')fail(422,'Order ID is required');const delivered=timestamp(b.delivered);
     const row=await env.DB.prepare('SELECT * FROM orders WHERE id=?').bind(b.id).first();
     if(!row)fail(404,'Order not found');
     if(delivered<row.purchased||Date.parse(delivered)>Date.now())fail(422,'Delivery must be between purchase and now');
     await env.DB.prepare('UPDATE orders SET delivered=?,reported=? WHERE id=? AND delivered IS NULL').bind(delivered,new Date().toISOString(),b.id).run();
     const saved=await env.DB.prepare('SELECT * FROM orders WHERE id=?').bind(b.id).first();
     if(saved.delivered!==delivered)fail(409,'This order already has a different outcome');return json(saved);
    }
   }
   if(request.method==='GET'&&path==='/api/diagnostics')return json(diagnose((await env.DB.prepare('SELECT * FROM orders').all()).results));
   if(request.method==='GET'&&path==='/api/orders')return json((await env.DB.prepare('SELECT * FROM orders ORDER BY created DESC,id LIMIT 200').all()).results);
   if(request.method==='GET'&&path==='/api/export'){
    const rows=(await env.DB.prepare('SELECT * FROM orders ORDER BY created ASC').all()).results;
    return json({orders:rows,total:rows.length,scope:'hosted_export'});
   }
   if(request.method==='GET'&&path==='/api/pipeline-status'){
    const rows=(await env.DB.prepare('SELECT COUNT(*) as n,COUNT(delivered) as resolved FROM orders').all()).results[0];
    const staleThreshold=new Date(Date.now()-14*86400000).toISOString();
    const staleRows=(await env.DB.prepare('SELECT COUNT(*) as stale FROM orders WHERE delivered IS NULL AND due < ?').bind(staleThreshold).all()).results[0];
    return json({as_of:new Date().toISOString(),live_model_version:model.version,total_orders:rows.n,resolved_orders:rows.resolved,pending_orders:rows.n-rows.resolved,stale_orders:staleRows.stale,hosted_runtime:'Cloudflare Worker (D1)'});
   }
   if(request.method==='GET'&&path==='/api/metrics'){
    const {results}=await env.DB.prepare('SELECT state,COUNT(*) n,COUNT(delivered) resolved,SUM(CASE WHEN delivered IS NOT NULL AND predicted=(delivered>=due) THEN 1 ELSE 0 END) correct FROM orders GROUP BY state ORDER BY state').all();
    return json(results.map(r=>({state:r.state,total:r.n,pending:r.n-r.resolved,observed:r.resolved?r.correct/r.resolved:null,lower:r.correct/r.n,upper:(r.correct+r.n-r.resolved)/r.n})));
   }
   return json({detail:'Not found'},404);
  }catch(e){return json({detail:e.detail||'Server error; please retry'},e.code||500)}
 }};
}

export function diagnose(rows,now=new Date()){
 const DAY=86400000,cutoff=Math.floor(now.getTime()/DAY)*DAY,groups=new Map(),output=[];
 for(const r of rows){if(Date.parse(r.created)>=cutoff)continue;const key=r.state+'|'+r.model_version;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)}
 for(const [key,events] of [...groups.entries()].sort()){
  const first=Math.floor(events.reduce((minimum,r)=>Math.min(minimum,Date.parse(r.created)),Infinity)/DAY)*DAY,day=Math.floor((cutoff-first)/DAY);
  const base={state:events[0].state,model_version:events[0].model_version,day,as_of:new Date(cutoff).toISOString()};
  if(day<67){output.push({...base,status:'collecting_reference',days_until_first_look:67-day});continue}
  if(day>=150){output.push({...base,status:'horizon_complete'});continue}
  function cohort(lo,hi){const chosen=events.filter(r=>Date.parse(r.created)>=first+lo*DAY&&Date.parse(r.created)<first+hi*DAY),known=chosen.filter(r=>r.reported&&Date.parse(r.reported)<=cutoff);return [known.filter(r=>Boolean(r.predicted)!==(Date.parse(r.delivered)>=Date.parse(r.due))).length,chosen.length-known.length,chosen.length]}
  const ref=cohort(0,30),cur=cohort(day-37,day-9);
  if(!ref[2]||!cur[2]){output.push({...base,status:'insufficient_data'});continue}
  const isMultiscale=day>=88,curMature=isMultiscale?cohort(day-58,day-30):null;
  const scales=isMultiscale&&curMature&&curMature[2]>0?2:1;
  const tail=.05/(2*27*120*scales),radius=n=>Math.sqrt(Math.log(1/tail)/(2*n));
  const upper=Math.min(1,(ref[0]+ref[1])/ref[2]+radius(ref[2]));
  const lowerShort=Math.max(0,cur[0]/cur[2]-radius(cur[2]));
  const lowerMature=scales===2?Math.max(0,curMature[0]/curMature[2]-radius(curMature[2])):0;
  const alarmShort=lowerShort>upper+.05,alarmMature=scales===2&&lowerMature>upper+.05;
  const alarm=alarmShort||alarmMature;
  const scale=alarmShort&&alarmMature?'both':(alarmShort?'short':(alarmMature?'mature':'none'));
  output.push({...base,status:alarm?'harm_evidence':'no_confirmed_harm',alarm,scale_triggered:scale,multiscale_active:scales===2,reference_upper:upper,current_lower:Math.max(lowerShort,lowerMature),current_upper:Math.min(1,(cur[0]+cur[1])/cur[2]+radius(cur[2])),margin:.05,reference_n:ref[2],current_n:cur[2],pending:cur[1]});
 }
 return {as_of:new Date(cutoff).toISOString(),groups:output,policy:'Daily multiscale fixed cohorts, 5 percentage-point harm margin; no automatic promotion',assumptions:'Independent bounded losses, fixed model, fixed 27 states; one 120-look budget per model version. Reusing versions or repeated restarts invalidates the budget.'};
}
