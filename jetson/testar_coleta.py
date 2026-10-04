#!/usr/bin/env python3
"""Testa a gravação automática do dataset — o laço REAL do vigil_jetson.py,
com câmera e modelo falsos. Roda no PC, sem Jetson, sem TensorRT.

O que se verifica é o que erraria em silêncio:

* caixa exportada com rótulo de OUTRO grão (janela ou offset trocado);
* grão dentro de uma imagem sem caixa — ensinaria o detector que é fundo;
* o mesmo grão físico em dois splits (a validação viraria memorização);
* imagem na escala errada (o quadro inteiro em vez da janela do modelo);
* a gravação segurando a inspeção, ou a memória crescendo com a jornada.

Cada grão falso tem a COR da sua classe. Isso dá uma verdade que não depende
de nenhum ID: no fim, a cor dentro de cada caixa exportada tem que bater com o
rótulo dela.

    python3 testar_coleta.py
"""
import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict

import cv2
import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)
sys.modules.setdefault('tensorrt', types.ModuleType('tensorrt'))
import gravador_dataset as gd  # noqa: E402

_spec = importlib.util.spec_from_file_location('cd_', os.path.join(_AQUI, 'coletar_dataset.py'))
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)
vj = cd._vigil()

CLASSES = gd.CLASSES
LADO = 704
SOBREPOR = int(LADO * 0.2)
FAIXA = 2 * LADO - SOBREPOR              # o que o nvvidconv entrega no rig
# cor por classe (BGR): a "verdade" que o teste lê de volta das imagens
COR = {'broken': (40, 40, 210), 'immature': (40, 200, 200), 'intact': (60, 200, 60),
       'skin-damaged': (210, 120, 40), 'spotted': (200, 60, 200)}
FALHAS = []


def checa(nome, cond, detalhe=''):
    print(f'{"ok  " if cond else "FALHA"}  {nome}' + (f'   [{detalhe}]' if detalhe else ''))
    if not cond:
        FALHAS.append(nome)


def classe_da_cor(img):
    """Classe pela cor do miolo — o 'humano perfeito' e o juiz do teste."""
    h, w = img.shape[:2]
    m = img[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(0)
    return min(COR, key=lambda c: float(np.sum((np.array(COR[c]) - m) ** 2)))


class Relogio:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


RELOGIO = Relogio()
gd.RELOGIO = RELOGIO


class Esteira:
    """Grãos em 5 colunas descendo a faixa — uma delas em cima da costura
    entre os dois recortes. Velocidade constante, como na esteira."""

    def __init__(self, n_varr=160, vel=30, borrar=False, parado=False,
                 largura=FAIXA, seed=5):
        self.n_varr, self.vel, self.borrar, self.parado = n_varr, vel, borrar, parado
        self.largura = largura
        self.k = 0
        rng = np.random.default_rng(seed)
        colunas = [120, 330, 630, 900, 1130] if largura >= FAIXA else [150, 350, 550]
        self.graos = []
        if parado:   # bandeja: grade fixa, a sessão inteira
            for i, x in enumerate(colunas):
                for j, y in enumerate((150, 360, 560)):
                    self.graos.append((x, None, y, CLASSES[int(rng.integers(5))]))
        else:
            for i in range(200):
                self.graos.append((colunas[i % len(colunas)], (i // len(colunas)) * 6,
                                   None, CLASSES[int(rng.integers(5))]))
        self.rng_tex = np.random.default_rng(9)

    def isOpened(self):
        return True

    def get(self, _):
        return 14.0

    def release(self):
        pass

    def read(self):
        self.k += 1
        RELOGIO.t += 1 / 14            # 14 varreduras/s, como no rig
        if self.k > self.n_varr:
            return False, None
        img = np.full((LADO, self.largura, 3), 14, np.uint8)
        for x, nasce, yfixo, cls in self.graos:
            y = yfixo if self.parado else -60 + (self.k - nasce) * self.vel
            if not -60 < y < LADO + 60:
                continue
            cv2.ellipse(img, (x, y), (44, 34), 0, 0, 360, COR[cls], -1)
            # textura: sem ela o laplaciano é baixo e a nitidez barra tudo
            for _ in range(30):
                px = int(self.rng_tex.normal(x, 14)); py = int(self.rng_tex.normal(y, 10))
                cv2.circle(img, (px, py), 1, tuple(int(c * 0.6) for c in COR[cls]), -1)
        if self.borrar:
            img = cv2.GaussianBlur(img, (31, 31), 0)
        return True, img


class ModeloFalso:
    """Acha os grãos por limiar e propõe a classe pela cor, ERRANDO 20%."""
    W = H = LADO
    offset = 0

    def __init__(self, erro=0.2):
        self.rng = np.random.default_rng(7)
        self.erro = erro

    def __call__(self, tile):
        cinza = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(cinza, 40, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 20 or h < 15:
                continue
            ci = CLASSES.index(classe_da_cor(tile[y:y + h, x:x + w]))
            if self.rng.random() < self.erro:
                ci = (ci + 1) % len(CLASSES)
            out.append((x, y, x + w, y + h, ci, 0.9))
        return out


def rodar(raiz, camera, extra=(), cam_spec='csi'):
    vj.RFDetrTRT = lambda *a, **k: ModeloFalso()
    vj.abrir_camera = lambda *a, **k: camera
    argv = ['--engine', 'falso.engine', '--camera', cam_spec, '--no-window',
            '--dataset-dir', raiz, '--conf', '0.25', '--dataset-passo', '2',
            '--dataset-bloco', '2', '--dataset-guarda', '0.3', '--dataset-min-livre', '0',
            *extra]
    vj.main(argv)


def grãos_nas_imagens(raiz):
    cd.usar_raiz(raiz)
    return {c['grao'] for _, m in cd._sessoes() for q in m['quadros'] for c in q['caixas']}


# ============================================================ 1. esteira
print('--- 1. esteira, 2 recortes: inspecionar = coletar ---')
raiz = tempfile.mkdtemp()
cam = Esteira()
rodar(raiz, cam, ['--roi', str(LADO), '--tiles', '2', '--esteira', '--lote', 'L007',
                  '--dataset-graos', '0'])
cd.usar_raiz(raiz)
sessoes = cd._sessoes()
checa('sessão criada sozinha', len(sessoes) == 1)
dir_q, meta = sessoes[0]
sess = meta['sessao']
pastas = sorted(os.listdir(os.path.join(raiz, 'revisar', sess)))
checa('revisar/<sessao>/ tem as 5 classes + descartar + duvida',
      pastas == sorted(gd.PASTAS), str(pastas))
checa('sessão no padrão do rig marcada com travas', meta['travas'] is True)
checa('a câmera foi lida até o fim (a gravação não parou a inspeção)',
      cam.k > cam.n_varr, f'{cam.k} leituras')

quadros = meta['quadros']
tamanhos = {(q['largura'], q['altura']) for q in quadros}
checa('cada imagem é UMA janela do modelo (704x704), não a faixa inteira',
      tamanhos == {(LADO, LADO)}, str(tamanhos))
img0 = cv2.imread(os.path.join(dir_q, quadros[0]['arquivo']))
checa('a imagem no disco tem o tamanho declarado', img0.shape[:2] == (LADO, LADO))
checa('as duas janelas são gravadas', {q['janela'] for q in quadros} == {0, 1})

revisao = cd.ler_revisao()
nas_imagens = grãos_nas_imagens(raiz)
checa('TODO grão que aparece numa imagem tem recorte para revisar',
      nas_imagens <= set(revisao), f'{len(nas_imagens - set(revisao))} sem recorte')
checa('um recorte por grão, na pasta da classe proposta',
      all(r['prevista'] == r['corrigida'] for r in revisao.values()))
vezes = Counter(c['grao'] for q in quadros for c in q['caixas'])
checa('o mesmo grão aparece em várias imagens (1 correção acerta várias caixas)',
      max(vezes.values()) > 1, f'até {max(vezes.values())}x')
costura = [g for g, n in vezes.items()
           if len({q['janela'] for q in quadros for c in q['caixas'] if c['grao'] == g}) == 2]
checa('grão na costura entra nas DUAS janelas, como na inferência', bool(costura),
      f'{len(costura)} grãos')

# caixa dentro da janela, com o conteúdo certo
erradas = 0
for q in quadros[:40]:
    img = cv2.imread(os.path.join(dir_q, q['arquivo']))
    for c in q['caixas']:
        x1, y1, x2, y2 = c['caixa']
        assert 0 <= x1 < x2 <= LADO and 0 <= y1 < y2 <= LADO, c
        if img[y1:y2, x1:x2].mean() < 30:
            erradas += 1
checa('as caixas caem em cima dos grãos na imagem salva', erradas == 0, f'{erradas} vazias')

# ---- revisão humana: um revisor perfeito move pela cor ----
print('\n--- revisão simulada (revisor perfeito, pela cor) ---')
movidos = 0
for grao, r in cd.ler_revisao().items():
    verdade = classe_da_cor(cv2.imread(r['arquivo']))
    if verdade != r['corrigida']:
        shutil.move(r['arquivo'], os.path.join(os.path.dirname(os.path.dirname(r['arquivo'])),
                                               verdade, os.path.basename(r['arquivo'])))
        movidos += 1
apos = cd.ler_revisao()
checa('o modelo errou e o revisor corrigiu', movidos > 0, f'{movidos} movidos')
checa('a leitura das pastas enxerga as correções',
      sum(1 for r in apos.values() if r['prevista'] != r['corrigida']) == movidos)

# três casos especiais: não-grão, dúvida e recorte perdido
lista = sorted(apos.items())
nao_grao = lista[0][0]
shutil.move(apos[nao_grao]['arquivo'], os.path.join(raiz, 'revisar', sess, gd.DESCARTE,
                                                    os.path.basename(apos[nao_grao]['arquivo'])))
duvidoso = lista[1][0]
shutil.move(apos[duvidoso]['arquivo'], os.path.join(raiz, 'revisar', sess, gd.DUVIDA,
                                                    os.path.basename(apos[duvidoso]['arquivo'])))
perdido = lista[2][0]
os.remove(apos[perdido]['arquivo'])

print('\n--- exportação ---')
args = types.SimpleNamespace(incluir_sem_trava=False)
cd.exportar(args)
pronto = os.path.join(raiz, 'pronto')

# o juiz: a cor dentro de CADA caixa exportada tem que bater com o rótulo
ok = errado = com_nao_grao = 0
graos_split = defaultdict(set)
por_arquivo = {q['arquivo']: q for q in quadros}
for sp in cd.SPLITS:
    for lab in glob.glob(os.path.join(pronto, sp, 'labels', '*.txt')):
        nome = os.path.basename(lab)[:-4] + '.jpg'
        img = cv2.imread(os.path.join(pronto, sp, 'images', nome))
        H, W = img.shape[:2]
        presentes = {c['grao'] for c in por_arquivo[nome]['caixas']}
        assert perdido not in presentes and duvidoso not in presentes, nome
        graos_split[sp] |= presentes - {nao_grao}
        com_nao_grao += nao_grao in presentes
        linhas = open(lab).read().split('\n')
        assert len(linhas) == len(presentes - {nao_grao}), nome
        for linha in linhas:
            ci, cx, cy, bw, bh = linha.split()
            cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
            corte = img[int(cy - bh / 2):int(cy + bh / 2), int(cx - bw / 2):int(cx + bw / 2)]
            if classe_da_cor(corte) == CLASSES[int(ci)]:
                ok += 1
            else:
                errado += 1
checa('TODA caixa exportada tem o rótulo do grão que está nela', errado == 0 and ok > 0,
      f'{ok} certas, {errado} erradas')
checa('imagem com grão em duvida/ ou sem recorte saiu inteira',
      not any(duvidoso in v or perdido in v for v in graos_split.values()))
checa('o não-grão perdeu a caixa, e as imagens dele ficaram', com_nao_grao > 0,
      f'{com_nao_grao} imagens exportadas sem a caixa dele')
vaz = [(a, b) for a in cd.SPLITS for b in cd.SPLITS if a < b and graos_split[a] & graos_split[b]]
checa('nenhum grão físico em dois splits', not vaz, str(vaz))
checa('as imagens exportadas são 704x704', all(
    cv2.imread(p).shape[:2] == (LADO, LADO)
    for p in glob.glob(os.path.join(pronto, '*', 'images', '*.jpg'))[:20]))

coco_n = sum(len(json.load(open(os.path.join(pronto, sp, '_annotations.coco.json')))['annotations'])
             for sp in cd.SPLITS)
yolo_n = sum(len(open(p).read().split('\n'))
             for p in glob.glob(os.path.join(pronto, '*', 'labels', '*.txt')))
checa('COCO e YOLO com as mesmas caixas', coco_n == yolo_n, f'{coco_n} x {yolo_n}')
antes = sorted(os.path.relpath(p, pronto) for p in glob.glob(os.path.join(pronto, '*', '*', '*')))
cd.exportar(args)
depois = sorted(os.path.relpath(p, pronto) for p in glob.glob(os.path.join(pronto, '*', '*', '*')))
checa('exportar de novo dá o mesmo resultado', antes == depois)
checa('RELATORIO.md e dataset.yaml gerados',
      os.path.exists(os.path.join(pronto, 'RELATORIO.md'))
      and os.path.exists(os.path.join(pronto, 'dataset.yaml')))

# copiar em vez de mover: o grão fica em duas pastas e não pode virar rótulo
r = cd.ler_revisao()[lista[3][0]]
outra = next(c for c in CLASSES if c != r['corrigida'])
shutil.copy(r['arquivo'], os.path.join(raiz, 'revisar', sess, outra, os.path.basename(r['arquivo'])))
conf = []
checa('grão copiado para duas pastas vira dúvida, não rótulo',
      cd.ler_revisao(conf)[lista[3][0]]['corrigida'] == gd.DUVIDA and conf == [lista[3][0]])

# ============================================================ 2. bandeja parada
print('\n--- 2. bandeja parada: o mesmo grão a sessão inteira ---')
raiz2 = tempfile.mkdtemp()
rodar(raiz2, Esteira(n_varr=100, parado=True), ['--roi', str(LADO), '--tiles', '2',
                                                 '--parado', '--dataset-graos', '0'])
cd.usar_raiz(raiz2)
_, m2 = cd._sessoes()[0]
blocos = {q['bloco'] for q in m2['quadros']}
checa('a sessão parada atravessa vários blocos de tempo', len(blocos) > 2, f'{len(blocos)} blocos')
try:
    cd.exportar(args)
    exportou = True
except AssertionError as e:
    exportou = False
    print('   ', e)
checa('exportar NÃO quebra quando o grão fica em todos os blocos', exportou)
usados = [sp for sp in cd.SPLITS if glob.glob(os.path.join(raiz2, 'pronto', sp, 'images', '*'))]
checa('a sessão parada inteira vai para UM split só', len(usados) == 1, str(usados))

# ============================================================ 3. filtros e limites
print('\n--- 3. filtros e limites ---')
raiz3 = tempfile.mkdtemp()
rodar(raiz3, Esteira(n_varr=60, borrar=True), ['--roi', str(LADO), '--tiles', '2', '--esteira'])
checa('cena borrada não grava nada e a sessão vazia é removida',
      not os.listdir(os.path.join(raiz3, 'quadros')) and not os.listdir(os.path.join(raiz3, 'revisar')))

raiz4 = tempfile.mkdtemp()
cam4 = Esteira()
rodar(raiz4, cam4, ['--roi', str(LADO), '--tiles', '2', '--esteira', '--dataset-graos', '12'])
cd.usar_raiz(raiz4)
_, m4 = cd._sessoes()[0]
checa('meta de grãos: a gravação para…', m4['resumo']['motivo_parada'] == 'meta'
      and m4['resumo']['graos'] < 40, f'{m4["resumo"]["graos"]} grãos')
checa('…e a inspeção continua até o fim da fonte', cam4.k > cam4.n_varr)
checa('…sem deixar grão de imagem salva sem recorte',
      grãos_nas_imagens(raiz4) <= set(cd.ler_revisao()))

raiz5 = tempfile.mkdtemp()
rodar(raiz5, Esteira(n_varr=40), ['--esteira'], cam_spec='http://10.0.0.9:4747/video')
checa('celular (fora do padrão) NÃO grava por padrão',
      not os.path.exists(os.path.join(raiz5, 'quadros'))
      or not os.listdir(os.path.join(raiz5, 'quadros')))
rodar(raiz5, Esteira(n_varr=60, largura=LADO), ['--esteira', '--dataset'],
      cam_spec='http://10.0.0.9:4747/video')
cd.usar_raiz(raiz5)
s5 = cd._sessoes()
checa('com --dataset grava, marcado sem trava e com sufixo na pasta',
      len(s5) == 1 and s5[0][1]['travas'] is False and s5[0][1]['sessao'].endswith('_sem-trava'))

raiz6 = tempfile.mkdtemp()
rodar(raiz6, Esteira(n_varr=30), ['--roi', str(LADO), '--tiles', '2', '--esteira',
                                   '--sem-dataset'])
checa('--sem-dataset não grava nem no rig', not os.path.exists(os.path.join(raiz6, 'quadros')))

# ============================================================ 4. o gravador sozinho
print('\n--- 4. gravador: fila, memória e épocas ---')
raiz7 = tempfile.mkdtemp()
quadro = np.full((LADO, LADO, 3), 14, np.uint8)
for x in range(80, 680, 120):
    cv2.ellipse(quadro, (x, 352), (44, 34), 0, 0, 360, (60, 200, 60), -1)
    for _ in range(40):
        cv2.circle(quadro, (int(np.random.normal(x, 14)), int(np.random.normal(352, 10))),
                   1, (30, 120, 30), -1)
janela = [(0, 0, LADO, LADO)]
rast = [(i, x - 44, 318, x + 44, 386, 2, 0.9) for i, x in enumerate(range(80, 680, 120), 1)]

imwrite_real = gd.cv2.imwrite


def imwrite_lento(*a, **k):
    time.sleep(0.05)                     # cartão SD ruim: 50 ms por arquivo
    return imwrite_real(*a, **k)


gd.cv2.imwrite = imwrite_lento
try:
    g = gd.GravadorDataset(raiz=raiz7, passo=1, max_graos=0, min_livre_gb=0, fila=6)
    pior = 0.0
    for _ in range(60):
        t = time.perf_counter()
        g.observar(quadro, rast, janela)
        pior = max(pior, time.perf_counter() - t)
    for tid, *_ in rast:
        g.aposentar(tid, 'intact', 0.9, 60)
    r7 = g.fechar()
finally:
    gd.cv2.imwrite = imwrite_real
checa('disco lento NÃO segura o laço', pior < 0.04, f'pior varredura {pior*1000:.1f} ms')
checa('o que não coube na fila é contado, não esperado', r7['perdidos_fila'] > 0,
      f'{r7["perdidos_fila"]} perdidos')
checa('os recortes têm prioridade sobre as imagens', r7['recortes'] == len(rast),
      f'{r7["recortes"]}/{len(rast)}')

g = gd.GravadorDataset(raiz=raiz7, passo=3, max_graos=0, min_livre_gb=0)
pico = 0
for k in range(3000):                    # 3000 varreduras, grãos entrando e saindo
    vivos = [(k // 10 * 10 + i, 80 + 120 * i - 44, 318, 80 + 120 * i + 44, 386, 2, .9)
             for i in range(5)]
    g.observar(quadro, vivos, janela)
    if k % 10 == 9:
        for tid, *_ in vivos:
            g.aposentar(tid, 'intact', .9, 10)
    pico = max(pico, len(g._chave), len(g._melhor), len(g._precisa))
r8 = g.fechar()
checa('memória limitada aos grãos vivos numa jornada longa', pico <= 5,
      f'pico {pico} entradas para {r8["graos"]} grãos gravados')

g = gd.GravadorDataset(raiz=raiz7, passo=1, max_graos=0, min_livre_gb=0)
g.observar(quadro, rast[:1], janela)
c1 = g._chave[rast[0][0]]
g.aposentar(rast[0][0], 'intact')
g.nova_epoca()
g.observar(quadro, rast[:1], janela)    # mesmo tid, rastreador novo (tecla c)
c2 = g._chave[rast[0][0]]
g.aposentar(rast[0][0], 'intact')
g.fechar()
checa('reset da contagem não reaproveita a chave de um grão', c1 != c2, f'{c1[-7:]} x {c2[-7:]}')

g = gd.GravadorDataset(raiz=raiz7, passo=1, max_graos=0, min_livre_gb=10 ** 9)
g.observar(quadro, rast, janela)
checa('disco cheio: para de gravar sozinho', not g.gravando and g.motivo == 'disco')
g.fechar()

g = gd.GravadorDataset(raiz=raiz7, passo=1, max_graos=0, min_livre_gb=0)
try:
    g.observar(quadro, [(1, 'lixo', None, 3, 4, 2, .9)], janela)   # entrada podre
    levantou = False
except Exception:
    levantou = True
g.observar(quadro, rast, janela)                                   # depois disso: nada
checa('falha inesperada no gravador NÃO derruba a inspeção',
      not levantou and g.falha and not g.gravando and g.n_varr == 1, str(g.falha)[:50])
g.fechar()

# o laudo é o produto: nem disco cheio ao fechar o dataset pode impedi-lo
raiz8 = tempfile.mkdtemp()
laudo = os.path.join(raiz8, 'laudo.json')
salvar_real = gd.GravadorDataset._salvar_meta
estado = {'n': 0}


def salvar_quebrado(self, resumo=None):
    estado['n'] += 1
    if resumo is not None:                # só no fechamento
        raise OSError(28, 'No space left on device')
    return salvar_real(self, resumo)


gd.GravadorDataset._salvar_meta = salvar_quebrado
try:
    rodar(raiz8, Esteira(n_varr=40), ['--roi', str(LADO), '--tiles', '2', '--esteira',
                                       '--laudo', laudo])
finally:
    gd.GravadorDataset._salvar_meta = salvar_real
final = json.load(open(laudo)) if os.path.exists(laudo) else {}
checa('disco cheio ao fechar o dataset não impede o laudo final',
      final.get('obs') == 'final' and final.get('graos', 0) > 0,
      f'{final.get("graos")} grãos no laudo')

for d in (raiz, raiz2, raiz3, raiz4, raiz5, raiz6, raiz7, raiz8):
    shutil.rmtree(d, ignore_errors=True)

print()
if FALHAS:
    print(f'{len(FALHAS)} FALHA(S):')
    for f in FALHAS:
        print('  -', f)
    sys.exit(1)
print('=== gravação automática do dataset validada ===')
