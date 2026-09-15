#!/usr/bin/env python3
"""Testa o `coletar_dataset.py` com uma câmera falsa — roda sem Jetson.

O que se verifica aqui é o que erraria em silêncio: grão físico aparecendo em
dois splits (a validação viraria memorização), recorte ruim entrando no
dataset, e captura sem as travas se misturando com a padronizada.

    python3 testar_coleta.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.modules.setdefault('tensorrt', types.ModuleType('tensorrt'))
_spec = importlib.util.spec_from_file_location(
    'cd_', os.path.join(_AQUI, 'coletar_dataset.py'))
cd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cd)


def cena(graos, lado=89, tam=704, borrar=False, encostar=False):
    """Quadro sintético no padrão do rig: fundo preto fosco, grãos claros."""
    img = np.full((tam, tam, 3), 14, np.uint8)
    img += np.random.default_rng(0).integers(0, 6, img.shape, dtype=np.uint8)
    for (cx, cy) in graos:
        cv2.ellipse(img, (cx, cy), (lado // 2, int(lado * 0.38)), 20, 0, 360,
                    (150, 185, 210), -1)
        if encostar:      # segundo grão colado: vira um blob só
            cv2.ellipse(img, (cx + lado - 6, cy), (lado // 2, int(lado * 0.38)),
                        20, 0, 360, (150, 185, 210), -1)
    if borrar:
        img = cv2.GaussianBlur(img, (21, 21), 0)
    return img


# ------------------------------------------------------- segmentação e filtros
print('--- segmentação e filtros de qualidade ---')
img = cena([(150, 150), (400, 300), (250, 500)])
achados = cd.segmentar(img)
bons = [a for a in achados if cd.avaliar(img, a[:4], a[4])[0]]
print(f'3 grãos espalhados -> {len(achados)} contornos, {len(bons)} aprovados')
assert len(bons) == 3, [cd.avaliar(img, a[:4], a[4])[1] for a in achados]

img_b = cena([(150, 150), (400, 300)], borrar=True)
motivos = [cd.avaliar(img_b, a[:4], a[4])[1] for a in cd.segmentar(img_b)]
print(f'cena borrada -> motivos de descarte: {set(motivos)}')
assert all(m == 'borrado' for m in motivos), motivos

img_e = cena([(200, 300)], encostar=True)
motivos = [cd.avaliar(img_e, a[:4], a[4])[1] for a in cd.segmentar(img_e)]
print(f'grãos encostados -> {set(motivos)}')
assert motivos and all(m in ('encostado', 'grande') for m in motivos), motivos

img_borda = cena([(2, 300)])
motivos = [cd.avaliar(img_borda, a[:4], a[4])[1] for a in cd.segmentar(img_borda)]
print(f'grão na borda -> {set(motivos)}')
assert 'na_borda' in motivos, motivos
print('OK: borrado, encostado e cortado na borda são barrados\n')


# ------------------------------------------------------------ captura completa
print('--- captura com câmera falsa ---')
tmp = tempfile.mkdtemp()
cd.RAIZ = tmp
cd.BRUTO = os.path.join(tmp, 'bruto')
cd.PRONTO = os.path.join(tmp, 'pronto')
cd.MANIFESTO = os.path.join(cd.BRUTO, 'manifesto.jsonl')


class CameraFalsa:
    """Grãos andando na esteira: o MESMO grão em vários quadros seguidos."""

    def __init__(self, n_graos=12, passo=40):
        self.n, self.passo, self.k = n_graos, passo, 0

    def read(self):
        self.k += 1
        if self.k > 60:
            return False, None
        graos = []
        for i in range(self.n):
            x = 120 + (i % 4) * 150
            y = 60 + (i // 4) * 170 + (self.k * self.passo) % 200
            if 90 < y < 620:
                graos.append((x, y))
        return True, cena(graos)

    def release(self): pass


import argparse
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa()
args = argparse.Namespace(classe='immature', lote='L001', graos=200, camera='csi',
                          roi=0, sem_trava=False, parado=False, sem_janela=True)
cd.capturar(args)

args.classe, args.lote = 'spotted', 'L002'
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa(n_graos=8)
cd.capturar(args)

# uma captura SEM travas, que não pode se misturar com as outras
args.classe, args.lote, args.sem_trava = 'broken', 'L003', True
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa(n_graos=6)
cd.capturar(args)

linhas = [json.loads(l) for l in open(cd.MANIFESTO)]
print(f'\nmanifesto: {len(linhas)} recortes')
graos = {l['grao'] for l in linhas}
print(f'grãos únicos: {len(graos)}')
assert len(linhas) > len(graos), 'cada grão devia render vários recortes'
por_grao = max(sum(1 for l in linhas if l['grao'] == g) for g in graos)
assert por_grao <= cd.POR_GRAO, f'{por_grao} recortes de um grão só (máx {cd.POR_GRAO})'
print(f'OK: no máximo {cd.POR_GRAO} recortes por grão (deduplicação do tracker)')


# ----------------------------------------------------------------- divisão
print('\n--- divisão ---')
args.dividir, args.incluir_sem_trava = True, False
cd.dividir(args)

import glob
from collections import defaultdict
grao_por_split = defaultdict(set)
for sp in ('train', 'valid', 'test'):
    for p in glob.glob(os.path.join(cd.PRONTO, sp, '*', '*.jpg')):
        nome = os.path.basename(p)
        grao_por_split[sp].add('_'.join(nome.split('_')[:3]))

for a in ('train', 'valid', 'test'):
    for b in ('train', 'valid', 'test'):
        if a < b:
            assert not (grao_por_split[a] & grao_por_split[b]), \
                f'VAZAMENTO entre {a} e {b}'
print('OK: nenhum grão físico em dois splits')

# a captura sem travas tem que ter ficado de fora
sem_trava_dentro = glob.glob(os.path.join(cd.PRONTO, '*', 'broken', '*.jpg'))
assert not sem_trava_dentro, f'captura sem travas entrou: {len(sem_trava_dentro)}'
print('OK: captura sem travas de exposição ficou de fora (outro domínio)')

# determinismo: dividir de novo dá exatamente o mesmo resultado
antes = {os.path.relpath(p, cd.PRONTO)
         for p in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*.jpg'))}
cd.dividir(args)
depois = {os.path.relpath(p, cd.PRONTO)
          for p in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*.jpg'))}
assert antes == depois, 'divisão não é determinística'
print('OK: dividir de novo dá o mesmo resultado (hash, não sorteio)')

# a estrutura tem que ser a que o notebook de treino já lê
assert os.path.isdir(os.path.join(cd.PRONTO, 'train', 'immature'))
assert os.path.exists(os.path.join(cd.PRONTO, 'dataset.yaml'))
assert os.path.exists(os.path.join(cd.PRONTO, 'RELATORIO.md'))
print('OK: train/valid/test por classe + dataset.yaml + RELATORIO.md')

# e o collect_base do notebook precisa reconhecer as pastas
sys.path.insert(0, os.path.join(_AQUI, '..', 'model'))
import unicodedata


def class_of(folder):     # mesma lógica do notebook
    ALIASES = {0: ['broken'], 1: ['immature'], 2: ['intact'],
               3: ['skin', 'casca'], 4: ['spotted']}
    n = unicodedata.normalize('NFKD', folder).encode('ascii', 'ignore').decode().lower()
    for idx in range(5):
        if any(k in n for k in ALIASES[idx]):
            return idx
    return None


for c in cd.CLASSES:
    assert class_of(c) is not None, f'o notebook não reconheceria a pasta {c}'
print('OK: os nomes das pastas são reconhecidos pelo collect_base do notebook')

shutil.rmtree(tmp)
print('\n=== coleta validada ===')
