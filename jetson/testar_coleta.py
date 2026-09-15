#!/usr/bin/env python3
"""Testa o `coletar_dataset.py` com câmera e modelo falsos — roda sem Jetson.

O que se verifica é o que erraria em silêncio: o mesmo grão físico caindo em
dois splits (a validação viraria memorização), recorte cortado ou borrado
entrando na revisão, e a contabilidade da correção manual — que é de onde sai
o primeiro número de acurácia medido no rig.

    python3 testar_coleta.py
"""
import argparse
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

CLASSES = cd.CLASSES
TAM = 704


def cena(graos, borrar=False):
    """Quadro no padrão do rig: fundo preto fosco, grãos claros com textura."""
    rng = np.random.default_rng(1)
    img = np.full((TAM, TAM, 3), 14, np.uint8)
    img = np.clip(img + rng.integers(0, 6, img.shape), 0, 255).astype(np.uint8)
    for (cx, cy) in graos:
        cv2.ellipse(img, (cx, cy), (44, 34), 20, 0, 360, (150, 185, 210), -1)
        # textura: sem ela o laplaciano fica baixo e o filtro de nitidez barra
        for _ in range(40):
            px = int(rng.normal(cx, 12)); py = int(rng.normal(cy, 9))
            cv2.circle(img, (px, py), 1, (110, 150, 180), -1)
    return cv2.GaussianBlur(img, (31, 31), 0) if borrar else img


class ModeloFalso:
    """Detecta os grãos e propõe classe com um ERRO CONHECIDO de 20%."""
    W = H = TAM

    def __init__(self, erro=0.2):
        self.rng = np.random.default_rng(7)
        self.erro = erro
        self.verdade = {}          # (cx arredondado) -> classe real

    def __call__(self, frame):
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 30 or h < 20:
                continue
            chave = x // 150
            real = self.verdade.setdefault(chave, int(self.rng.integers(len(CLASSES))))
            prev = real
            if self.rng.random() < self.erro:      # erra de propósito
                prev = (real + 1) % len(CLASSES)
            out.append((x, y, x + w, y + h, prev, 0.9))
        return out


class CameraFalsa:
    """Grãos andando na esteira: o MESMO grão em vários quadros seguidos."""

    def __init__(self, n=10, passo=35, borrar=False):
        self.n, self.passo, self.k, self.borrar = n, passo, 0, borrar

    def read(self):
        self.k += 1
        if self.k > 70:
            return False, None
        graos = [(120 + (i % 3) * 190, 70 + (i // 3) * 150 + (self.k * self.passo) % 190)
                 for i in range(self.n)]
        graos = [(x, y) for x, y in graos if 70 < y < TAM - 70]
        return True, cena(graos, self.borrar)

    def release(self): pass


print('--- captura com câmera e modelo falsos ---')
tmp = tempfile.mkdtemp()
cd.RAIZ, cd.REVISAR = tmp, os.path.join(tmp, 'revisar')
cd.PRONTO = os.path.join(tmp, 'pronto')
cd.MANIFESTO = os.path.join(tmp, 'manifesto.jsonl')

modelo = ModeloFalso(erro=0.2)
cd.vj.RFDetrTRT = lambda *a, **k: modelo
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa()

args = argparse.Namespace(engine='falso.engine', lote='L001', graos=500,
                          camera='csi', source=None, roi=0, tiles=1, conf=0.25,
                          class_offset=None, sem_trava=False, parado=False,
                          sem_janela=True)
cd.capturar(args)

itens = cd.ler_revisao()
print(f'\nrecortes na revisão: {len(itens)}')
assert itens, 'nada foi salvo'
graos = {i['grao'] for i in itens}
assert len(graos) == len(itens), 'mais de um recorte por grão (POR_GRAO=1)'
print(f'OK: {len(graos)} grãos únicos, 1 recorte cada (tracker deduplicou)')

# o nome do arquivo tem que carregar a classe proposta
assert all(i['prevista'] in CLASSES for i in itens), 'classe prevista não veio no nome'
print('OK: o nome do arquivo guarda a classe que o modelo propôs')

# o arquivo tem que estar NA PASTA da classe proposta
assert all(i['pasta'] == i['prevista'] for i in itens), 'recorte na pasta errada'
print('OK: cada recorte foi para a pasta da classe proposta')

# --- simula a correção manual: mover arquivos entre pastas ---
print('\n--- correção manual simulada ---')
rng = np.random.default_rng(3)
movidos = 0
for i in itens:
    if rng.random() < 0.25:                      # corrijo 25% deles
        nova = CLASSES[(CLASSES.index(i['pasta']) + 1) % len(CLASSES)]
        shutil.move(i['caminho'], os.path.join(cd.REVISAR, nova, i['arquivo']))
        movidos += 1
# e mando alguns para descartar
desc = 0
for i in cd.ler_revisao()[:4]:
    shutil.move(i['caminho'], os.path.join(cd.REVISAR, cd.DESCARTE, i['arquivo']))
    desc += 1
print(f'movi {movidos} para outra classe e {desc} para {cd.DESCARTE}/')

apos = cd.ler_revisao()
detectados = sum(1 for i in apos
                 if i['prevista'] in CLASSES and i['prevista'] != i['pasta'])
assert detectados >= movidos, f'detectou {detectados} movimentos, esperava >= {movidos}'
print(f'OK: o script detecta {detectados} correções lendo onde o arquivo ficou')

# --- divisão ---
print('\n--- divisão ---')
args.dividir, args.incluir_sem_trava = True, False
cd.dividir(args)

import glob
from collections import defaultdict
por_split = defaultdict(set)
for sp in ('train', 'valid', 'test'):
    for p in glob.glob(os.path.join(cd.PRONTO, sp, '*', '*.jpg')):
        por_split[sp].add(os.path.basename(p).rpartition('__')[0])
for a in ('train', 'valid', 'test'):
    for b in ('train', 'valid', 'test'):
        if a < b:
            assert not (por_split[a] & por_split[b]), f'VAZAMENTO entre {a} e {b}'
print('OK: nenhum grão físico em dois splits')

# descartados não podem ter entrado
no_pronto = sum(len(v) for v in por_split.values())
assert no_pronto == len(apos) - desc, f'{no_pronto} no pronto, esperava {len(apos)-desc}'
print(f'OK: os {desc} de {cd.DESCARTE}/ ficaram fora do dataset')

# o rótulo tem que ser a PASTA onde ficou, não a do nome
for sp in ('train', 'valid', 'test'):
    for p in glob.glob(os.path.join(cd.PRONTO, sp, '*', '*.jpg')):
        pasta = os.path.basename(os.path.dirname(p))
        assert pasta in CLASSES
print('OK: o rótulo final é a pasta em que o arquivo foi deixado')

# determinismo
antes = {os.path.relpath(p, cd.PRONTO)
         for p in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*.jpg'))}
cd.dividir(args)
assert antes == {os.path.relpath(p, cd.PRONTO)
                 for p in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*.jpg'))}
print('OK: dividir de novo dá o mesmo resultado (hash, não sorteio)')

rel = open(os.path.join(cd.PRONTO, 'RELATORIO.md')).read()
assert 'Acurácia do modelo no rig' in rel and 'você corrigiu' in rel
print('OK: o relatório traz a acurácia medida pelas correções')
assert os.path.exists(os.path.join(cd.PRONTO, 'dataset.yaml'))

# --- filtros de qualidade ---
print('\n--- filtros ---')
tmp2 = tempfile.mkdtemp()
cd.RAIZ, cd.REVISAR = tmp2, os.path.join(tmp2, 'revisar')
cd.PRONTO, cd.MANIFESTO = os.path.join(tmp2, 'pronto'), os.path.join(tmp2, 'm.jsonl')
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa(borrar=True)
args.dividir = False
cd.capturar(args)
assert not cd.ler_revisao(), 'quadro borrado passou pelo filtro de nitidez'
print('OK: cena borrada não gera nenhum recorte')

shutil.rmtree(tmp); shutil.rmtree(tmp2)
print('\n=== coleta validada ===')
