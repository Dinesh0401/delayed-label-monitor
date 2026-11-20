import assert from 'node:assert/strict';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {adapter} from './db-adapter.mjs';
import {createHandler,probability,diagnose} from './worker.mjs';
const model=JSON.parse(readFileSync('serving-model.json'));
for(const f of JSON.parse(readFileSync('inference-fixtures.json')))assert.ok(Math.abs(probability(f.features,model)-f.expected)<1e-12);
const db=new DatabaseSync(':memory:');db.exec(readFileSync('hosted/drizzle/0000_delivery_orders.sql','utf8'));
const app=createHandler(model,'html','study',{}),env={DB:adapter(db)};
async function req(path,body){const r=await app.fetch(new Request('https://example.test/api/'+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{}),env);return {status:r.status,data:await r.json()}}
const order={id:'TEST',state:'SP',purchased:'2026-01-01T10:00:00Z',due:'2026-01-10T00:00:00Z'};
const first=await req('orders',order);assert.equal(first.status,200);assert.deepEqual((await req('orders',order)).data,first.data);
assert.equal((await req('orders',{...order,state:'AM'})).status,409);
assert.equal((await req('orders',{...order,id:'OTHER',due:order.purchased})).status,422);
assert.equal((await req('metrics')).data[0].pending,1);
assert.equal((await req('outcomes',{id:'TEST',delivered:'2025-01-01T00:00:00Z'})).status,422);
const outcome={id:'TEST',delivered:'2026-01-09T00:00:00Z'};
const saved=await req('outcomes',outcome);assert.equal(saved.status,200);assert.deepEqual((await req('outcomes',outcome)).data,saved.data);
assert.equal((await req('outcomes',{...outcome,delivered:'2026-01-11T00:00:00Z'})).status,409);
const m=(await req('metrics')).data[0];assert.equal(m.pending,0);assert.equal(m.lower,m.upper);
assert.equal((await req('orders')).data.length,1);

// Test batch orders, batch outcomes, export and pipeline-status
const batchOrders=[
  {id:'BATCH-1',state:'SP',purchased:'2026-01-01T10:00:00Z',due:'2026-01-10T00:00:00Z'},
  {id:'BATCH-2',state:'RJ',purchased:'2026-01-02T10:00:00Z',due:'2026-01-12T00:00:00Z'}
];
const bRes=await req('batch-orders',batchOrders);
assert.equal(bRes.status,200);assert.equal(bRes.data.inserted,2);

// Retry duplicate batch orders must report inserted: 0
const bRetryRes=await req('batch-orders',batchOrders);
assert.equal(bRetryRes.status,200);assert.equal(bRetryRes.data.inserted,0);

// Batch outcomes rejecting future deliveries (e.g. 2099)
const bOutFuture=[{id:'BATCH-2',delivered:'2099-01-01T00:00:00Z'}];
const bOutFutRes=await req('batch-outcomes',bOutFuture);
assert.equal(bOutFutRes.status,200);assert.equal(bOutFutRes.data.updated,0);

// Valid batch outcome
const bOutValid=[{id:'BATCH-1',delivered:'2026-01-08T12:00:00Z'}];
const bOutValRes=await req('batch-outcomes',bOutValid);
assert.equal(bOutValRes.status,200);assert.equal(bOutValRes.data.updated,1);

// Export route
const expRes=await req('export');
assert.equal(expRes.status,200);assert.equal(expRes.data.total,3);

// Pipeline status route
const pipeRes=await req('pipeline-status');
assert.equal(pipeRes.status,200);assert.equal(pipeRes.data.total_orders,3);assert.equal(pipeRes.data.resolved_orders,2);

console.log('Worker: model parity, SQLite-backed workflow, retries, invalid dates, conflicts, batch endpoints and bounds passed.');
const diagnosis=await req('diagnostics');assert.equal(diagnosis.status,200);assert.ok(Array.isArray(diagnosis.data.groups));
const collected=diagnose([{state:'SP',model_version:'v1',created:'2026-01-01T10:00:00Z',reported:null}],new Date('2026-01-02T00:00:00Z'));
assert.equal(collected.groups[0].status,'collecting_reference');assert.equal(collected.groups[0].days_until_first_look,66);
const mature=[];for(let day=0;day<100;day++){for(let i=0;i<100;i++){const created=new Date(Date.UTC(2026,0,1+day,1)).toISOString();mature.push({state:'SP',model_version:'v1',created,reported:created,delivered:created,due:new Date(Date.UTC(2026,0,1+day+(day<40?1:-1),1)).toISOString(),predicted:0})}}
assert.equal(diagnose(mature,new Date('2026-04-11T00:00:00Z')).groups[0].status,'harm_evidence');
console.log('Daily diagnostic warmup and strong-harm fixtures passed.');db.close();
