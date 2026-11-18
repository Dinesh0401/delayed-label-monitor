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
export function createHandler(model,html,study,research){
 const json=(data,status=200)=>new Response(JSON.stringify(data),{status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
 return {async fetch(request,env){
  const url=new URL(request.url),path=url.pathname;
  try {
   if(request.method==='GET'&&path==='/')return new Response(html,{headers:{'Content-Type':'text/html;charset=utf-8'}});
   if(request.method==='GET'&&(path==='/delivery-study.html'||path==='/delivery-study'))return new Response(study,{headers:{'Content-Type':'text/html;charset=utf-8'}});
   if(request.method==='GET'&&path==='/research.json')return json(research);
   if(request.method==='GET'&&path==='/api/model'){const {coefficients,intercept,features,...card}=model;return json(card)}
   if(!path.startsWith('/api/'))return json({detail:'Not found'},404);
   if(!env.DB)fail(503,'Persistent database is not configured');
   if(request.method==='POST'){
    const origin=request.headers.get('Origin');if(origin&&origin!==url.origin)fail(403,'Cross-origin writes are not allowed');
    if(!request.headers.get('Content-Type')?.startsWith('application/json'))fail(415,'Send JSON');
    const raw=await request.text();if(raw.length>4096)fail(413,'Request is too large');let b;try{b=JSON.parse(raw)}catch{fail(400,'Invalid JSON')}
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
   if(request.method==='GET'&&path==='/api/orders')return json((await env.DB.prepare('SELECT * FROM orders ORDER BY created DESC,id LIMIT 200').all()).results);
   if(request.method==='GET'&&path==='/api/metrics'){
    const {results}=await env.DB.prepare('SELECT state,COUNT(*) n,COUNT(delivered) resolved,SUM(CASE WHEN delivered IS NOT NULL AND predicted=(delivered>=due) THEN 1 ELSE 0 END) correct FROM orders GROUP BY state ORDER BY state').all();
    return json(results.map(r=>({state:r.state,total:r.n,pending:r.n-r.resolved,observed:r.resolved?r.correct/r.resolved:null,lower:r.correct/r.n,upper:(r.correct+r.n-r.resolved)/r.n})));
   }
   return json({detail:'Not found'},404);
  }catch(e){return json({detail:e.detail||'Server error; please retry'},e.code||500)}
 }};
}
