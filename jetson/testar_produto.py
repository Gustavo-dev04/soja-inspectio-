#!/usr/bin/env python3
"""Testa o software do aparelho (jetson/produto/) — HTTP de verdade, câmera e
modelo falsos. Roda no PC, sem Jetson, sem TensorRT, sem nada para instalar.

O que se verifica é o que o operador sentiria:

* criar, iniciar e encerrar um lote pela API, e o laudo sair certo;
* o laudo ser auditável — mexer no banco depois e o código de verificação acusar;
* vídeo ao vivo e estado em tempo real chegando no navegador;
* nada derrubar a inspeção: câmera que cai, varredura com erro, pedido ruim;
* desligar no meio de um lote (SIGTERM ou queda de energia) não perder o lote.

    python3 testar_produto.py
"""
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request

import cv2
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.modules.setdefault('tensorrt', types.ModuleType('tensorrt'))
_spec = importlib.util.spec_from_file_location('vigil_jetson', os.path.join(AQUI, 'vigil_jetson.py'))
vj = importlib.util.module_from_spec(_spec)
sys.modules['vigil_jetson'] = vj
_spec.loader.exec_module(vj)

from produto import aparelho as ap      # noqa: E402
from produto import servico as sv       # noqa: E402
from produto.banco import Banco         # noqa: E402
from produto.web import criar_servidor, servir_em_thread   # noqa: E402

LADO, FAIXA = 704, 2 * 704 - int(704 * 0.2)
COR = {'broken': (40, 40, 210), 'immature': (40, 200, 200), 'intact': (60, 200, 60),
       'skin-damaged': (210, 120, 40), 'spotted': (200, 60, 200)}
FALHAS = []


def checa(nome, cond, detalhe=''):
    print(f'{"ok  " if cond else "FALHA"}  {nome}' + (f'   [{detalhe}]' if detalhe else ''))
    if not cond:
        FALHAS.append(nome)


def classe_da_cor(img):
    h, w = img.shape[:2]
    m = img[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(0)
    return min(COR, key=lambda c: float(np.sum((np.array(COR[c]) - m) ** 2)))


class CameraSemFim:
    """Esteira contínua; `quebrar(n)` simula a câmera caindo por n leituras."""

    def __init__(self):
        self.k, self.falhas = 0, 0
        rng = np.random.default_rng(3)
        self.cls = [list(COR)[int(rng.integers(5))] for _ in range(400)]

    def isOpened(self):
        return True

    def get(self, _):
        return 14.0

    def release(self):
        pass

    def read(self):
        time.sleep(0.004)
        if self.falhas > 0:
            self.falhas -= 1
            return False, None
        self.k += 1
        img = np.full((LADO, FAIXA, 3), 14, np.uint8)
        for i in range(60):
            x = (120, 330, 630, 900, 1130)[i % 5]
            y = -60 + (self.k - (i // 5) * 6) % 72 * 12
            if -60 < y < LADO + 60:
                cor = COR[self.cls[(self.k // 72 * 60 + i) % 400]]
                cv2.ellipse(img, (x, y), (44, 34), 0, 0, 360, cor, -1)
                for j in range(25):
                    cv2.circle(img, (x - 30 + (j * 7) % 60, y - 20 + (j * 11) % 40), 1,
                               tuple(int(c * .6) for c in cor), -1)
        return True, img


class ModeloFalso:
    W = H = LADO
    offset = 0

    def __call__(self, tile):
        cinza = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(cinza, 40, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 20 and h >= 15:
                ci = vj.NAMES.index(classe_da_cor(tile[y:y + h, x:x + w]))
                out.append((x, y, x + w, y + h, ci, 0.9))
        return out


CAMERAS = []


def nova_camera(*a, **k):
    c = CameraSemFim()
    CAMERAS.append(c)
    return c


vj.RFDetrTRT = lambda *a, **k: ModeloFalso()
vj.abrir_camera = nova_camera
sv.PARCIAL_S = 0.3

tmp = tempfile.mkdtemp()
engine = os.path.join(tmp, 'falso.engine')
open(engine, 'wb').write(b'engine falsa')
cfg_arq = os.path.join(tmp, 'aparelho.json')
json.dump({'nome': 'rig de teste', 'banco': 'vigil.db', 'engine': 'falso.engine',
           'camera': 'csi:1', 'roi': 704, 'tiles': 2, 'esteira': True,
           'dataset_dir': 'dataset', 'dataset_passo': 2, 'dataset_min_livre': 0},
          open(cfg_arq, 'w'))

# ============================================================ configuração
print('--- configuração do aparelho ---')
cfg = ap.carregar(cfg_arq)
argv = ap.argv_de(cfg)
checa('aparelho.json vira os argumentos da linha de comando',
      argv[:1] == ['--no-window'] and '--esteira' in argv and argv[argv.index('--roi') + 1] == '704')
checa('caminhos relativos ao aparelho.json', cfg['engine'] == engine)
checa('"0" não some (disco mínimo zero é um valor válido)',
      argv[argv.index('--dataset-min-livre') + 1] == '0')
ruim = os.path.join(tmp, 'ruim.json')
json.dump({'engin': 'x'}, open(ruim, 'w'))
try:
    ap.carregar(ruim)
    rejeitou = False
except ValueError:
    rejeitou = True
checa('chave com erro de digitação é rejeitada, não ignorada', rejeitou)
checa('o modelo tem impressão digital no laudo', ap.versao_modelo(engine).startswith('falso.engine@'))
checa('o aparelho tem identidade', bool(ap.id_do_aparelho()))
checa('o aparelho.json do repositório é válido',
      bool(ap.argv_de(ap.carregar(ap.CONFIG_PADRAO))))

# ============================================================ serviço + HTTP
print('\n--- serviço no ar ---')
servico = sv.Servico(cfg, vj=vj, log=lambda *a: None)
srv = criar_servidor(servico, '127.0.0.1', 0)
URL = f'http://127.0.0.1:{srv.server_address[1]}'
servir_em_thread(srv)
servico.iniciar()


def pedir(metodo, caminho, corpo=None, cru=None):
    dados = cru if cru is not None else (json.dumps(corpo).encode() if corpo is not None else None)
    req = urllib.request.Request(URL + caminho, data=dados, method=metodo,
                                 headers={'Content-Type': 'application/json'} if dados else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            corpo = r.read()
            tipo = r.headers.get('Content-Type', '')
            return r.status, (json.loads(corpo) if 'json' in tipo else corpo.decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


def esperar(cond, segundos=10):
    fim = time.time() + segundos
    while time.time() < fim:
        if cond():
            return True
        time.sleep(0.05)
    return False


st, html = pedir('GET', '/')
checa('a página do operador abre', st == 200 and 'Vígil' in html)
checa('o script da interface é servido', pedir('GET', '/ui/app.js')[0] == 200)
checa('não serve arquivo fora da pasta da interface',
      pedir('GET', '/ui/../servico.py')[0] == 404 and pedir('GET', '/ui/%2e%2e/web.py')[0] == 404)
st, e = pedir('GET', '/api/estado')
checa('estado responde, sem lote aberto', st == 200 and e['lote_id'] is None and e['camera_ok'])

st, r = pedir('POST', '/api/lotes', {'produtor': 'sem código'})
checa('lote sem código é recusado (400)', st == 400, r.get('erro'))
st, r = pedir('POST', '/api/lotes', cru=b'{isto nao e json')
checa('JSON quebrado é recusado (400)', st == 400)
st, r = pedir('POST', '/api/lotes', {'codigo': 'L2026-031', 'produtor': 'Sítio Boa Vista',
                                     'variedade': 'BRS 1003', 'amostra_kg': '1.5'})
checa('lote criado (201)', st == 201 and r['estado'] == 'criado', r.get('id', '')[:8])
lote1 = r['id']
st, r = pedir('POST', f'/api/lotes/{lote1}/iniciar')
checa('lote iniciado', st == 200 and r['estado'] == 'inspecionando')
_, outro = pedir('POST', '/api/lotes', {'codigo': 'L2026-032'})
checa('não abre dois lotes ao mesmo tempo (409)',
      pedir('POST', f'/api/lotes/{outro["id"]}/iniciar')[0] == 409)
checa('não reinicia o mesmo lote (409)', pedir('POST', f'/api/lotes/{lote1}/iniciar')[0] == 409)
checa('lote inexistente (404)', pedir('POST', '/api/lotes/nao-existe/iniciar')[0] == 404)

checa('a inspeção conta grãos no lote',
      esperar(lambda: pedir('GET', '/api/estado')[1]['graos'] >= 30, 20),
      f'{pedir("GET", "/api/estado")[1]["graos"]} grãos')

# vídeo ao vivo: o primeiro quadro do MJPEG tem que ser um JPEG que abre
with urllib.request.urlopen(URL + '/ao-vivo.mjpg', timeout=10) as r:
    tipo = r.headers.get('Content-Type', '')
    buf = b''
    while buf.count(b'\xff\xd9') < 1 and len(buf) < 2_000_000:
        buf += r.read(4096)
ini, fim = buf.find(b'\xff\xd8'), buf.find(b'\xff\xd9') + 2
img = cv2.imdecode(np.frombuffer(buf[ini:fim], np.uint8), cv2.IMREAD_COLOR)
checa('vídeo ao vivo chega e abre como imagem', 'multipart' in tipo and img is not None)
checa('o vídeo vem reduzido para o celular', img is not None and img.shape[1] <= 960,
      f'{img.shape[1] if img is not None else "-"} px')

# estado em tempo real (SSE)
with urllib.request.urlopen(URL + '/api/eventos', timeout=10) as r:
    linha = b''
    while not linha.startswith(b'data:'):
        linha = r.readline()
    ev = json.loads(linha[5:])
checa('estado em tempo real chega por SSE', ev['lote_id'] == lote1 and ev['lote_info']['codigo'] == 'L2026-031')
checa('o estado traz o tempo por etapa da varredura',
      set(ev['tempos_ms']) == set(vj.ETAPAS) and ev['tempos_ms']['gpu'] > 0)
checa('fotografia parcial salva no banco durante o lote',
      esperar(lambda: (servico.banco.lote(lote1) or {}).get('parcial') is not None, 5))

# ============================================================ resiliência
print('\n--- nada derruba a inspeção ---')
CAMERAS[-1].falhas = 40
checa('câmera caiu: o aparelho percebe', esperar(lambda: not servico.camera_ok, 10))
checa('câmera reaberta sozinha', esperar(lambda: servico.camera_ok, 15))
checa('a queda fica no registro', any(r['tipo'] == 'camera' for r in pedir('GET', '/api/registro')[1]))

proc_real = servico.insp._processar
estado_erro = {'n': 0}


def processar_quebrado(*a, **k):
    estado_erro['n'] += 1
    if estado_erro['n'] <= 2:
        raise RuntimeError('varredura ruim de propósito')
    return proc_real(*a, **k)


servico.insp._processar = processar_quebrado
antes = servico.varreduras
checa('varredura com erro não para o laço', esperar(lambda: servico.varreduras > antes + 10, 10))
servico.insp._processar = proc_real
checa('o erro fica no registro', any('de propósito' in r['msg'] for r in servico.banco.eventos()))
checa('API segue de pé', pedir('GET', '/api/estado')[0] == 200)

# ============================================================ laudo
print('\n--- laudo ---')
st, laudo = pedir('POST', f'/api/lotes/{lote1}/encerrar')
d = laudo.get('dados', {})
checa('encerrar devolve o laudo', st == 200 and d.get('graos', 0) > 0,
      f'{d.get("graos")} grãos, {d.get("premium_pct")}% premium')
checa('o laudo carrega lote, aparelho e modelo',
      d.get('lote_codigo') == 'L2026-031' and d.get('aparelho_id') and
      d.get('modelo', '').startswith('falso.engine@') and d.get('amostra_declarada_kg') == 1.5)
checa('a contagem por classe fecha com o total',
      sum(d['por_classe'].values()) == d['graos'] == d['premium'] + d['nao_premium'])
checa('o laudo traz o tempo médio por etapa', set(d.get('tempos_ms', {})) == set(vj.ETAPAS))
recortes = [f for _r, _d, fs in os.walk((d.get('dataset') or {}).get('revisar', '/nada'))
            for f in fs if f.endswith('.jpg')]
checa('o lote fechou a sessão de dataset dele, com o código do lote nos grãos',
      (d.get('dataset') or {}).get('graos', 0) > 0 and recortes
      and all(f.startswith('L2026-031_') for f in recortes), f'{len(recortes)} recortes')
st, l2 = pedir('GET', f'/api/laudos/{laudo["id"]}')
checa('laudo íntegro ao ser lido de volta', st == 200 and l2['integro'] is True)
checa('a página do laudo abre', pedir('GET', f'/laudo/{laudo["id"]}')[0] == 200)
checa('o lote aparece como encerrado, com link para o laudo',
      any(l['id'] == lote1 and l['estado'] == 'encerrado' and l['laudo_id'] == laudo['id']
          for l in pedir('GET', '/api/lotes')[1]))
checa('o laudo entra na fila de sincronização',
      [x['id'] for x in pedir('GET', '/api/laudos?pendentes=1')[1]] == [laudo['id']])
checa('laudo inexistente (404)', pedir('GET', '/api/laudos/nao-existe')[0] == 404)
checa('encerrar de novo é recusado (409)', pedir('POST', f'/api/lotes/{lote1}/encerrar')[0] == 409)

# adulterar o laudo direto no banco: o código de verificação tem que acusar
con = sqlite3.connect(cfg['banco'])
dados = json.loads(con.execute('SELECT dados FROM laudos WHERE id=?', (laudo['id'],)).fetchone()[0])
dados['premium_pct'] = 99.9
con.execute('UPDATE laudos SET dados=? WHERE id=?', (json.dumps(dados), laudo['id']))
con.commit(); con.close()
checa('laudo adulterado no banco é denunciado', pedir('GET', f'/api/laudos/{laudo["id"]}')[1]['integro'] is False)

# ============================================================ desligamento
print('\n--- desligar no meio de um lote ---')
st, r = pedir('POST', f'/api/lotes/{outro["id"]}/iniciar')
checa('o próximo lote começa do zero', st == 200 and esperar(
    lambda: pedir('GET', '/api/estado')[1]['lote_id'] == outro['id'], 5))
esperar(lambda: pedir('GET', '/api/estado')[1]['graos'] >= 10, 15)
srv.shutdown()
servico.parar()
srv.server_close()
b = Banco(cfg['banco'])
l_outro = b.lote(outro['id'])
checa('SIGTERM com lote aberto: o lote é encerrado com laudo',
      l_outro['estado'] == 'encerrado'
      and any(x['lote_id'] == outro['id'] for x in b.laudos()))

# queda de energia: o lote fica 'inspecionando' no banco e o serviço volta depois
b.criar_lote('L2026-033')
l3 = b.lotes()[0]
b.marcar_lote(l3['id'], 'inspecionando')
b.salvar_parcial(l3['id'], dict(json.loads(json.dumps(d)), graos=123, obs='parcial'))
b.fechar()
servico2 = sv.Servico(cfg, vj=vj, log=lambda *a: None)
l3b = servico2.banco.lote(l3['id'])
laudo3 = next((x for x in servico2.banco.laudos() if x['lote_id'] == l3['id']), None)
dados3 = servico2.banco.laudo(laudo3['id'])['dados'] if laudo3 else {}
checa('queda de energia: o lote volta como interrompido, com o laudo parcial',
      l3b['estado'] == 'interrompido' and dados3.get('graos') == 123
      and 'interrompido' in dados3.get('obs', ''))
checa('o banco tem versão de esquema', servico2.banco.versao() == 1)
servico2.parar()

shutil.rmtree(tmp, ignore_errors=True)
print()
if FALHAS:
    print(f'{len(FALHAS)} FALHA(S):')
    for f in FALHAS:
        print('  -', f)
    sys.exit(1)
print('=== software do aparelho validado ===')
