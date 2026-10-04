'use strict';
// Interface do operador. Sem framework: o aparelho serve isto do próprio Jetson.

const CLASSES = [
  ['intact', 'Intacto'], ['immature', 'Imaturo'], ['broken', 'Quebrado'],
  ['skin-damaged', 'Casca danificada'], ['spotted', 'Manchado'],
];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let loteAtivo = null;      // id do lote em inspeção (do servidor)
let ultimoEstado = null;

async function api(metodo, caminho, corpo) {
  const r = await fetch(caminho, {
    method: metodo,
    headers: corpo ? { 'Content-Type': 'application/json' } : {},
    body: corpo ? JSON.stringify(corpo) : undefined,
  });
  const dados = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(dados.erro || `erro ${r.status}`);
  return dados;
}

function tempo(s) {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  const dd = (n) => String(n).padStart(2, '0');
  return (h ? `${h}:` : '') + `${dd(m)}:${dd(x)}`;
}

// ------------------------------------------------------------- estado ao vivo
function desenharEstado(e) {
  ultimoEstado = e;
  $('n-premium-pct').textContent = e.graos ? `${e.premium_pct.toFixed(1)}%` : '—';
  $('n-graos').textContent = e.graos.toLocaleString('pt-BR');
  $('n-premium').textContent = e.premium.toLocaleString('pt-BR');
  $('n-nao').textContent = e.nao_premium.toLocaleString('pt-BR');
  $('n-kg').textContent = `${e.massa_kg.toLocaleString('pt-BR')} kg`;
  $('n-kgh').textContent = `${e.vazao_kg_h.toLocaleString('pt-BR')} kg/h`;
  $('n-tempo').textContent = tempo(e.duracao_s);

  const total = e.graos || 1;
  $('classes').innerHTML = CLASSES.map(([k, nome]) => {
    const n = e.por_classe[k] || 0;
    return `<div class="classe"><span>${nome}</span>
      <span class="barra"><i style="width:${(100 * n / total).toFixed(1)}%;
        background:var(--classe-${k})"></i></span><b>${n}</b></div>`;
  }).join('');

  const cam = $('st-camera');
  cam.textContent = e.camera_ok ? 'câmera ok' : 'câmera caiu';
  cam.className = 'pilula ' + (e.camera_ok ? 'ok' : 'ruim');
  const rec = $('st-rec');
  rec.hidden = !e.dataset;
  rec.textContent = e.dataset || '';
  rec.className = 'pilula ' + (e.dataset && e.dataset.startsWith('REC ') ? 'rec' : '');

  const t = e.tempos_ms || {};
  const soma = Object.values(t).reduce((a, b) => a + b, 0);
  $('tecnico').innerHTML = `<table>
    <tr><td>Aparelho</td><td>${esc(e.aparelho)} · ${esc(e.aparelho_id)}</td></tr>
    <tr><td>Modelo</td><td>${esc(e.modelo)}</td></tr>
    <tr><td>Varreduras/s</td><td>${e.fps}</td></tr>
    <tr><td>Grãos no quadro</td><td>${e.no_quadro}</td></tr>
    ${Object.entries(t).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.toFixed(1)} ms</td></tr>`).join('')}
    <tr><td><b>total da varredura</b></td><td><b>${soma.toFixed(1)} ms</b></td></tr>
  </table>`;

  if (e.lote_id !== loteAtivo) { loteAtivo = e.lote_id; desenharLote(e); carregarLotes(); }
}

function conectar() {
  const st = $('st-conexao');
  const fonte = new EventSource('/api/eventos');
  fonte.onopen = () => { st.textContent = 'conectado'; st.className = 'pilula ok'; };
  fonte.onmessage = (m) => desenharEstado(JSON.parse(m.data));
  fonte.onerror = () => {
    st.textContent = 'sem conexão'; st.className = 'pilula ruim';
    // o EventSource reconecta sozinho; o vídeo precisa de um empurrão
    setTimeout(() => { $('aovivo').src = '/ao-vivo.mjpg?t=' + Date.now(); }, 3000);
  };
}

// ------------------------------------------------------------- lote
function desenharLote(e) {
  const p = $('painel-lote');
  if (e.lote_id && e.lote_info) {
    const l = e.lote_info;
    p.className = 'cartao lote-ativo';
    p.innerHTML = `<h2>Em inspeção</h2>
      <div class="codigo">${esc(l.codigo)}</div>
      <div class="meta">${[l.produtor, l.variedade, l.amostra_kg ? `${l.amostra_kg} kg declarados` : '']
        .filter(Boolean).map(esc).join(' · ') || 'sem detalhes'}</div>
      <button class="perigo" id="bt-encerrar">Encerrar lote e emitir laudo</button>
      <p class="erro" id="erro-lote"></p>`;
    $('bt-encerrar').onclick = async (ev) => {
      if (!confirm(`Encerrar o lote ${l.codigo} e emitir o laudo?`)) return;
      ev.target.disabled = true;
      try {
        const laudo = await api('POST', `/api/lotes/${e.lote_id}/encerrar`);
        location.href = `/laudo/${laudo.id}`;
      } catch (err) { $('erro-lote').textContent = err.message; ev.target.disabled = false; }
    };
    return;
  }
  p.className = 'cartao';
  p.innerHTML = `<h2>Novo lote</h2>
    <form id="form-lote">
      <label>Código do lote *<input name="codigo" required maxlength="40"
        placeholder="ex.: L2026-031" autocomplete="off"></label>
      <div class="dupla">
        <label>Produtor<input name="produtor" maxlength="80"></label>
        <label>Variedade<input name="variedade" maxlength="60"></label>
      </div>
      <div class="dupla">
        <label>Amostra (kg)<input name="amostra_kg" type="number" step="0.001" min="0" inputmode="decimal"></label>
        <label>Observação<input name="obs" maxlength="200"></label>
      </div>
      <button class="primario" type="submit">Iniciar inspeção</button>
      <p class="erro" id="erro-lote"></p>
    </form>`;
  $('form-lote').onsubmit = async (ev) => {
    ev.preventDefault();
    const bt = ev.target.querySelector('button');
    bt.disabled = true;
    const d = Object.fromEntries(new FormData(ev.target));
    if (!d.amostra_kg) delete d.amostra_kg;
    try {
      const lote = await api('POST', '/api/lotes', d);
      await api('POST', `/api/lotes/${lote.id}/iniciar`);
    } catch (err) { $('erro-lote').textContent = err.message; bt.disabled = false; }
  };
}

const NOME_ESTADO = { criado: 'criado', inspecionando: 'em inspeção',
                      encerrado: 'encerrado', interrompido: 'interrompido' };

async function carregarLotes() {
  try {
    const lotes = await api('GET', '/api/lotes');
    $('lotes').innerHTML = lotes.length ? lotes.slice(0, 15).map((l) => `<li>
      <span><b>${esc(l.codigo)}</b> <small>${esc((l.criado_em || '').slice(0, 16).replace('T', ' '))}</small></span>
      ${l.laudo_id ? `<a href="/laudo/${l.laudo_id}">laudo</a>`
                   : `<span>${esc(NOME_ESTADO[l.estado] || l.estado)}</span>`}</li>`).join('')
      : '<li class="vazio">nenhum lote ainda</li>';
    const reg = await api('GET', '/api/registro');
    $('registro').innerHTML = reg.slice(0, 20).map((r) =>
      `<li><span>${esc(r.tipo)}: ${esc(r.msg)}</span><span>${esc((r.quando || '').slice(11, 16))}</span></li>`).join('');
  } catch (err) { /* a lista volta na próxima mudança de estado */ }
}

desenharLote({});
conectar();
carregarLotes();
setInterval(carregarLotes, 30000);
