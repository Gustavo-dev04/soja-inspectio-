#!/usr/bin/env python3
"""Testa as células de avaliação do `treino_base12k_jetson.ipynb` — sem GPU.

Roda a avaliação contra um modelo FALSO que erra de forma conhecida (perde
grãos, confunde spotted com intact, inventa falso positivo) e confere que os
números batem com a verdade injetada. Roda os dois casos de indexação de
classe: o off-by-one já apareceu 3 vezes neste projeto, e um relatório errado
com cara de certo é pior que um erro na tela.

    python3 testar_base12k_avaliacao.py
"""
import json, os, sys, types, shutil
import numpy as np
from collections import Counter, defaultdict
from PIL import Image

nb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'treino_base12k_jetson.ipynb')))
codigo = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
CEL_PRED, CEL_MATRIZ, CEL_CURVA = codigo[6], codigo[7], codigo[8]

NAMES = ['broken', 'immature', 'intact', 'skin-damaged', 'spotted']
RAIZ, N_IMG, POR_IMG = 'evaltest', 40, 6
rng = np.random.default_rng(7)

# ---- val COCO falso, com verdade conhecida ----
shutil.rmtree(RAIZ, ignore_errors=True)
os.makedirs(f'{RAIZ}/valid')
imgs, anns, verdade = [], [], {}
aid = 0
for i in range(N_IMG):
    Image.new('RGB', (704, 704)).save(f'{RAIZ}/valid/{i:04d}.jpg')
    imgs.append({'id': i, 'file_name': f'{i:04d}.jpg', 'width': 704, 'height': 704})
    caixas = []
    for j in range(POR_IMG):
        c = int(rng.integers(5))
        x, y = 20 + (j % 3) * 220, 20 + (j // 3) * 220
        anns.append({'id': aid, 'image_id': i, 'category_id': c + 1,
                     'bbox': [x, y, 90, 90], 'area': 8100, 'iscrowd': 0})
        caixas.append((c, x, y, x + 90, y + 90)); aid += 1
    verdade[i] = caixas
json.dump({'images': imgs, 'annotations': anns,
           'categories': [{'id': k+1, 'name': n, 'supercategory': 'soja'}
                          for k, n in enumerate(NAMES)]},
          open(f'{RAIZ}/valid/_annotations.coco.json', 'w'))

class ModeloFalso:
    """Erra de forma CONTROLADA: confunde spotted->intact, perde alguns grãos,
    inventa um falso positivo, e devolve os ids com o offset pedido."""
    def __init__(self, offset, perde=0.10, confunde=0.30, inventa=0.15):
        self.offset, self.perde, self.confunde, self.inventa = offset, perde, confunde, inventa
        self.rng = np.random.default_rng(3)
        self.perdidos = Counter(); self.confundidos = 0; self.inventados = 0
    def predict(self, img, threshold=0.35):
        i = self.atual
        cls, caixas, confs = [], [], []
        for c, x1, y1, x2, y2 in verdade[i]:
            if self.rng.random() < self.perde:
                self.perdidos[NAMES[c]] += 1; continue
            pc = c
            if NAMES[c] == 'spotted' and self.rng.random() < self.confunde:
                pc = NAMES.index('intact'); self.confundidos += 1
            cls.append(pc + self.offset); caixas.append([x1, y1, x2, y2])
            confs.append(float(self.rng.uniform(0.5, 0.99)))
        if self.rng.random() < self.inventa:
            cls.append(NAMES.index('broken') + self.offset)
            caixas.append([600, 600, 660, 660]); confs.append(0.6)
            self.inventados += 1
        d = types.SimpleNamespace()
        d.class_id = np.array(cls, int); d.xyxy = np.array(caixas, float).reshape(-1, 4)
        d.confidence = np.array(confs, float)
        return d

def roda(offset):
    fake = ModeloFalso(offset)
    class Wrap:
        def predict(self, img, threshold=0.35): return fake.predict(img, threshold)
    g = {'np': np, 'os': os, 'json': json, 'Image': Image, 'Counter': Counter,
         'defaultdict': defaultdict, 'NAMES': NAMES, 'PREMIUM': 'intact',
         'COCO_DIR': RAIZ, 'BASE_PTH': 'x.pth', 'Modelo': lambda **k: Wrap(),
         'sv': types.ModuleType('sv')}
    sys.modules['supervision'] = types.ModuleType('supervision')
    # o modelo falso precisa saber qual imagem está processando
    orig = Image.open
    def open_marcado(p, *a, **k):
        fake.atual = int(os.path.basename(str(p)).split('.')[0]); return orig(p, *a, **k)
    Image.open = open_marcado
    try:
        exec(CEL_PRED, g); exec(CEL_MATRIZ, g); exec(CEL_CURVA, g)
    finally:
        Image.open = orig
    return g, fake

print('=' * 66); print(' CASO 1: modelo devolve id 0-indexado'); print('=' * 66)
g0, f0 = roda(offset=0)
assert g0['OFFSET'] == 0, g0['OFFSET']
print('\nOK: offset 0 detectado')

print('\n' + '=' * 66); print(' CASO 2: modelo devolve id 1-indexado (o bug histórico)'); print('=' * 66)
g1, f1 = roda(offset=1)
assert g1['OFFSET'] == 1, f'offset 1 NÃO detectado ({g1["OFFSET"]}) — voltaria o off-by-one'
print('\nOK: offset 1 detectado — o mesmo relatório sai dos dois casos')

# as duas matrizes têm que ser iguais: o offset não pode mudar o resultado
assert (g0['M'] == g1['M']).all(), 'offset mudou a matriz de confusão!'
print('OK: matriz idêntica nos dois casos')

# as contas batem com a verdade injetada
M, na, ff = g0['M'], g0['nao_achados'], g0['falso_fundo']
assert sum(na.values()) == sum(f0.perdidos.values()), (dict(na), dict(f0.perdidos))
print(f'OK: perdidos contabilizados ({sum(na.values())} grãos)')
assert ff['broken'] == f0.inventados, (dict(ff), f0.inventados)
print(f'OK: falso positivo de fundo contabilizado ({f0.inventados})')
i_sp, i_in = NAMES.index('spotted'), NAMES.index('intact')
assert M[i_sp, i_in] == f0.confundidos, (M[i_sp, i_in], f0.confundidos)
print(f'OK: confusão spotted->intact bate ({f0.confundidos} casos)')
# recall de spotted tem que refletir perda + confusão, não só a diagonal
sup_sp = M[i_sp].sum() + na['spotted']
rec_sp = M[i_sp, i_sp] / sup_sp
assert rec_sp < 0.85, f'recall de spotted deveria cair, veio {rec_sp:.2f}'
print(f'OK: recall de spotted = {rec_sp:.3f} (caiu, como a simulação forçou)')
# precisão de intact tem que cair, porque recebeu os spotted confundidos
prec_in = M[i_in, i_in] / (M[:, i_in].sum() + ff['intact'])
assert prec_in < 1.0
print(f'OK: precisão de intact = {prec_in:.3f} (contaminada pelos spotted)')

# curva premium: limiar mais alto = menos defeito pego, menos grão bom descartado
c = g0['curva']
r_baixo = c(0.35); r_alto = c(0.90)
rec_b = r_baixo[0] / max(r_baixo[0] + r_baixo[3], 1)
rec_a = r_alto[0] / max(r_alto[0] + r_alto[3], 1)
perda_b = r_baixo[1] / max(r_baixo[1] + r_baixo[2], 1)
perda_a = r_alto[1] / max(r_alto[1] + r_alto[2], 1)
assert rec_a < rec_b and perda_a <= perda_b, (rec_b, rec_a, perda_b, perda_a)
print(f'OK: curva monotônica — recall {rec_b:.2f}->{rec_a:.2f}, '
      f'perda {perda_b:.2f}->{perda_a:.2f} ao subir o limiar')

shutil.rmtree(RAIZ)
print('\n=== avaliação do notebook validada ===')
