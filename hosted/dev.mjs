import {createServer} from 'node:http';
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
import {adapter} from './db-adapter.mjs';
import worker from '../dist/server/index.js';
const sqlite=new DatabaseSync('/tmp/delayed-label-preview.sqlite');
sqlite.exec(readFileSync('drizzle/0000_delivery_orders.sql','utf8').replace('CREATE TABLE','CREATE TABLE IF NOT EXISTS'));
createServer(async(req,res)=>{try{const chunks=[];for await(const chunk of req)chunks.push(chunk);const response=await worker.fetch(new Request('http://'+req.headers.host+req.url,{method:req.method,headers:req.headers,...(['GET','HEAD'].includes(req.method)?{}:{body:Buffer.concat(chunks)})}),{DB:adapter(sqlite)});res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()))}catch(e){res.writeHead(500);res.end('Preview error')}}).listen(Number(process.argv.includes('--port')?process.argv[process.argv.indexOf('--port')+1]:(process.env.PORT||3000)),'0.0.0.0',()=>console.log('Preview ready'));
