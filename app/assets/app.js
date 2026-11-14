// Fridge Guardian v1 web page. Polls /api/state once per second.

const $ = (id) => document.getElementById(id);
let state = null;
let labelsFilled = false;

const LEVEL_TEXT = {
  empty: ['Fridge is empty', 'Show food to the camera to start tracking.'],
  fresh: ['All fresh', 'Everything is well within its shelf life.'],
  eat_soon: ['Eat soon', ''],
  expired: ['Past shelf life', ''],
  door: ['Door open', 'Close the door to save energy.'],
};

async function api(path, params = {}, method = 'POST') {
  const qs = new URLSearchParams(params).toString();
  const res = await fetch(`/api/${path}${qs ? '?' + qs : ''}`, { method });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function fmtDuration(days) {
  const abs = Math.abs(days);
  if (abs >= 1) return `${abs.toFixed(abs >= 10 ? 0 : 1)} d`;
  const h = abs * 24;
  if (h >= 1) return `${h.toFixed(h >= 10 ? 0 : 1)} h`;
  return `${Math.max(1, Math.round(h * 60))} min`;
}

function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtAdded(ts, ageDays) {
  const d = new Date(ts * 1000);
  if (ageDays < 0.9) return fmtTime(ts).slice(0, 5);
  return d.toLocaleString([], { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

let render = function (s) {
  state = s;
  const sum = s.summary;

  // Header chips
  const t = s.sensors.temp_c, h = s.sensors.humidity;
  $('chipTemp').textContent = t == null ? 'No temp sensor' : `${t.toFixed(1)} °C`;
  $('chipHum').textContent = h == null ? '—' : `${h.toFixed(0)} % RH`;
  $('chipHum').style.display = h == null ? 'none' : '';
  const door = $('chipDoor');
  if (s.sensors.lux == null) {
    door.textContent = 'No door sensor';
    door.classList.remove('open');
  } else {
    door.textContent = s.door_open ? `Door open ${s.door_open_s}s` : 'Door closed';
    door.classList.toggle('open', s.door_open);
  }

  // Banner
  const [title, defaultText] = LEVEL_TEXT[sum.level] || ['—', ''];
  $('banner').className = `banner ${sum.level}`;
  $('bannerLabel').textContent = title;
  let text = defaultText;
  if (sum.eat_first && (sum.level === 'eat_soon' || sum.level === 'expired')) {
    const f = sum.eat_first;
    text = f.days_left <= 0
      ? `Eat or check the ${f.label} first: ${fmtDuration(f.days_left)} past its shelf life.`
      : `Eat the ${f.label} first: ${fmtDuration(f.days_left)} left.`;
  }
  $('bannerText').textContent = text;
  $('cFresh').textContent = sum.counts.fresh;
  $('cSoon').textContent = sum.counts.eat_soon;
  $('cExpired').textContent = sum.counts.expired;

  // Mode
  $('modeIn').classList.toggle('active', s.mode === 'in');
  $('modeOut').classList.toggle('active', s.mode === 'out');
  $('modeOut').classList.toggle('out', s.mode === 'out');
  $('modeOut').textContent = s.mode === 'out' ? `Taking out (${Math.ceil(s.mode_s_left)}s)` : 'Taking out';

  // What the camera sees
  const seen = Object.entries(s.camera || {});
  $('seen').textContent = 'Camera sees: ' + (seen.length
    ? seen.map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(', ')
    : 'nothing');

  // Items
  const list = $('items');
  list.innerHTML = s.items.map((it) => {
    const first = sum.eat_first && sum.eat_first.id === it.id && it.state !== 'fresh';
    const left = it.days_left <= 0 ? `${fmtDuration(it.days_left)} over` : `${fmtDuration(it.days_left)} left`;
    return `<li class="item ${it.state}">
      <div class="name">${esc(it.label)}${first ? '<span class="badge eat_first">eat first</span>' : ''}</div>
      <div class="left">${left}</div>
      <button class="remove" data-id="${it.id}" title="Remove">✕</button>
      <div class="bar"><i style="width:${Math.round(it.used_fraction * 100)}%"></i></div>
      <div class="meta">In fridge ${fmtDuration(it.age_days)} of
        <button class="shelf" data-id="${it.id}" data-days="${it.shelf_days}">${it.shelf_days} d</button>
        · added ${fmtAdded(it.added_ts, it.age_days)}</div>
    </li>`;
  }).join('');
  $('emptyMsg').style.display = s.items.length ? 'none' : '';

  // Events
  $('events').innerHTML = s.events.slice(0, 15)
    .map((e) => `<li><time>${fmtTime(e.ts)}</time><span>${esc(e.text)}</span></li>`).join('');

  // Label suggestions (once)
  if (!labelsFilled && s.shelf_life_table) {
    $('labels').innerHTML = Object.keys(s.shelf_life_table).sort()
      .map((l) => `<option value="${esc(l)}">`).join('');
    labelsFilled = true;
  }
}

async function refresh() {
  try {
    const res = await fetch('/api/state');
    render(await res.json());
    $('chipConn').textContent = 'Live';
    $('chipConn').className = 'chip conn ok';
  } catch (e) {
    $('chipConn').textContent = 'Board offline';
    $('chipConn').className = 'chip conn bad';
  }
}

// ---- Actions ---------------------------------------------------------------
$('modeIn').onclick = () => api('mode', { mode: 'in' }).then((r) => render(r.state));
$('modeOut').onclick = () => api('mode', { mode: 'out' }).then((r) => render(r.state));
$('clearBtn').onclick = () => {
  if (state && state.items.length && window.confirm('Remove all tracked items?')) {
    api('clear').then((r) => render(r.state));
  }
};
$('addForm').onsubmit = (ev) => {
  ev.preventDefault();
  const label = $('addLabel').value.trim();
  if (!label) return;
  const days = parseFloat($('addDays').value);
  api('add', days > 0 ? { label, shelf_days: days } : { label }).then((r) => {
    $('addLabel').value = '';
    $('addDays').value = '';
    render(r.state);
  });
};
$('items').onclick = (ev) => {
  const rm = ev.target.closest('.remove');
  if (rm) return api('remove', { item_id: rm.dataset.id }).then((r) => render(r.state));
  const sh = ev.target.closest('.shelf');
  if (sh) {
    const v = window.prompt('Shelf life in days for this item:', sh.dataset.days);
    const days = parseFloat(v);
    if (days > 0) api('shelf', { item_id: sh.dataset.id, days }).then((r) => render(r.state));
  }
};

// ---- Simulator helpers (only shown when running sim/run_sim.py) -----------
let simReady = false;
function setupSim(s) {
  if (simReady || !s.sim) return;
  simReady = true;
  $('simBox').style.display = '';
  $('videoPlaceholder').textContent = 'Simulator: no camera. Use the buttons below.';
  $('simNote').textContent = `Time runs ${s.sim.speed}x faster than real time, so items age quickly.`;
  $('simButtons').innerHTML = s.sim.labels
    .map((l) => `<button class="ghost" data-label="${esc(l)}">${esc(l)}</button>`).join('');
  $('simButtons').onclick = (ev) => {
    const b = ev.target.closest('button');
    if (b) api('sim/show', { label: b.dataset.label });
  };
}

// ---- Camera stream from the object detection brick (port 4912) ------------
function connectVideo() {
  const iframe = $('videoFrame');
  const url = `http://${window.location.hostname}:4912/embed`;
  let timer = setInterval(() => { iframe.src = url; }, 2000);
  iframe.onload = () => {
    clearInterval(timer);
    $('videoPlaceholder').style.display = 'none';
    iframe.style.display = 'block';
  };
}

let videoStarted = false;
const _render = render;
render = (s) => {
  _render(s);
  setupSim(s);
  if (!videoStarted && !s.sim) {
    videoStarted = true;
    connectVideo();
  }
};

refresh();
setInterval(refresh, 1000);
