const $=id=>document.getElementById(id),pct=x=>x===null?'—':(100*x).toFixed(1)+'%';
let day=0,events=[],history=[],seed=42,timer,running=!matchMedia('(prefers-reduced-motion: reduce)').matches;
function random(){seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
function summarize(group){const all=events.filter(e=>e.group===group&&e.day>day-14),known=all.filter(e=>e.arrival<=day),c=known.filter(e=>!e.error).length;return {n:all.length,k:known.length,c,rate:known.length?1-c/known.length:null}}
function log(text){const li=document.createElement('li');li.textContent='D'+String(day).padStart(3,'0')+'  '+text;$('events').prepend(li);while($('events').children.length>5)$('events').lastChild.remove()}
function tick(){day++;const scenario=$('scenario').value,shift=day>=20;
 for(let g=0;g<2;g++){for(let i=0;i<(g?3:12);i++){const harm=g&&shift&&['harm','both'].includes(scenario),error=random()<(harm?.42:.1),slow=g&&shift&&['delay','both'].includes(scenario);events.push({day,group:g,error,arrival:day+(random()<(error?.3:.8)?1:slow?15:6)});}}
 const newly=events.filter(e=>e.arrival===day);log('15 predictions · '+newly.length+' outcomes received');
 if(day===20&&scenario!=='stable')log('Scenario change applied to smaller group');
 const known=events.filter(e=>e.arrival<=day),correct=known.filter(e=>!e.error).length;
 $('total').textContent=events.length;$('pending').textContent=events.length-known.length;$('accuracy').textContent=pct(known.length?correct/known.length:null);$('day').textContent=day;$('clock').textContent='Day '+day;$('new').textContent=15;$('inflight').textContent=events.length-known.length;$('done').textContent=known.length;
 const summaries=[summarize(0),summarize(1)];history.push(summaries.map(s=>s.rate));if(history.length>90)history.shift();
 $('groups').replaceChildren();summaries.forEach((s,g)=>{const row=document.createElement('div');row.className='grouprow';const lower=s.c/s.n,upper=(s.c+s.n-s.k)/s.n;row.innerHTML='<div><b>'+(g?'Smaller group':'Majority')+'</b><span>'+s.n+' recent predictions</span></div><div><b>'+pct(s.k?s.c/s.k:null)+'</b><span>Observed accuracy</span></div><div><b>'+(s.n-s.k)+'</b><span>Pending</span></div><div><span>'+pct(lower)+' – '+pct(upper)+' possible accuracy</span><div class="track"><div class="fill" style="left:'+lower*100+'%;width:'+(upper-lower)*100+'%"></div></div></div>';$('groups').append(row)});
 $('message').textContent=shift&&scenario!=='stable'?'The selected change is active in the smaller group. Compare the evidence coverage and accuracy range.':'Gathering outcomes across two groups with different sample sizes.';
 const s=summaries[1];$('decision').textContent=s.k<10?'Collect more evidence · fewer than 10 resolved outcomes in the smaller group.':s.rate>.25?'Investigate · observed small-group error exceeds the illustrative 25% threshold.':'Continue observation · delayed outcomes can still change this conclusion.';
 draw();
}
function draw(){const canvas=$('plot'),w=canvas.clientWidth||600,h=230,dpr=devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);
 c.font='10px monospace';[0,.25,.5,.75,1].forEach(v=>{const y=12+(1-v)*190;c.strokeStyle='#29323d';c.beginPath();c.moveTo(35,y);c.lineTo(w,y);c.stroke();c.fillStyle='#718096';c.fillText(Math.round(v*100)+'%',0,y+3)});
 ['#91e7bf','#baa4ff'].forEach((color,g)=>{c.beginPath();let started=false;history.forEach((s,i)=>{if(s[g]===null){started=false;return}const x=35+i/Math.max(1,history.length-1)*(w-45),y=12+(1-s[g])*190;if(!started)c.moveTo(x,y);else c.lineTo(x,y);started=true});c.strokeStyle=color;c.lineWidth=2.5;c.shadowColor=color;c.shadowBlur=7;c.stroke();c.shadowBlur=0});
}
function schedule(){clearInterval(timer);if(running)timer=setInterval(tick,Number($('speed').value));$('play').textContent=running?'Pause':'Resume';document.body.classList.toggle('paused',!running)}
function reset(){day=0;events=[];history=[];seed=42;$('events').replaceChildren();tick();schedule()}
$('play').onclick=()=>{running=!running;schedule()};$('step').onclick=()=>{running=false;schedule();tick()};$('reset').onclick=reset;$('scenario').onchange=reset;$('speed').onchange=schedule;window.addEventListener('resize',draw);
fetch('detection-results.json').then(r=>{if(!r.ok)throw Error();return r.json()}).then(data=>{for(const r of data.rows){const tr=document.createElement('tr');for(const value of [r.scenario,r.method==='censored'?'Fixed-delay Bayesian':'Arrived labels',r.false_alarm_runs+' / '+r.runs,r.eligible_runs?r.detected_runs+' / '+r.eligible_runs:'—',r.mean_detection_delay===null?'—':r.mean_detection_delay.toFixed(2)+' days']){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('results').append(tr)}}).catch(()=>{$('results').textContent='Results unavailable. Reload to retry.'});
reset();
