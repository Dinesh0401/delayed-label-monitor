const byId = id => document.getElementById(id);
const percent = value => Math.round(value * 100) + '%';
let total = 0, resolved = 0, correct = 0, running = !matchMedia('(prefers-reduced-motion: reduce)').matches, timer;
function card(id, text, style = '') {
  const item = document.createElement('div');
  item.className = 'card ' + style;
  item.textContent = text;
  byId(id).replaceChildren(item);
}
function render() {
  byId('total').textContent = total;
  byId('pending').textContent = total - resolved;
  byId('accuracy').textContent = resolved ? percent(correct / resolved) : '—';
  const low = total ? correct / total : 0;
  const high = total ? (correct + total - resolved) / total : 1;
  byId('range').textContent = percent(low) + ' – ' + percent(high);
  byId('interval').style.left = (100 * low) + '%';
  byId('interval').style.width = (100 * (high - low)) + '%';
  byId('bounds').setAttribute('aria-label', 'Final accuracy can be between ' + percent(low) + ' and ' + percent(high));
  byId('explain').textContent = total > resolved
    ? 'One outcome is still missing. Observed accuracy only includes the results received so far.'
    : 'All outcomes in this batch have arrived. The possible range has narrowed to the measured accuracy.';
}
function advance() {
  let message;
  if (total === resolved) {
    total++;
    card('prediction', 'Order ' + total + ' · predicted on time');
    card('pendingCard', 'Order ' + total + ' · waiting…', 'waiting');
    message = 'Order ' + total + ' predicted · outcome pending';
  } else {
    resolved++;
    const ok = resolved % 4 !== 3;
    correct += Number(ok);
    card('pendingCard', 'No pending outcomes');
    card('outcome', 'Order ' + resolved + (ok ? ' · on time ✓' : ' · arrived late'), ok ? '' : 'wrong');
    message = 'Order ' + resolved + (ok ? ' confirmed correct' : ' confirmed incorrect');
  }
  const entry = document.createElement('li');
  entry.textContent = message;
  byId('log').prepend(entry);
  while (byId('log').children.length > 4) byId('log').lastChild.remove();
  render();
}
function schedule() {
  clearInterval(timer);
  if (running) timer = setInterval(advance, Number(byId('speed').value));
  document.body.classList.toggle('paused', !running);
  byId('play').textContent = running ? 'Pause stream' : 'Resume stream';
  byId('status').textContent = running ? 'Stream running' : 'Paused · use Next event to explore';
}
byId('play').onclick = () => { running = !running; schedule(); };
byId('next').onclick = () => { running = false; schedule(); advance(); };
byId('reset').onclick = () => {
  total = resolved = correct = 0;
  byId('log').replaceChildren();
  byId('outcome').replaceChildren();
  advance();
  schedule();
};
byId('speed').onchange = schedule;
advance();
schedule();
