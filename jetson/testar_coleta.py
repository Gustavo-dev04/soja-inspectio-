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
from collections import Counter

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
cd.PRONTO, cd.QUADROS = os.path.join(tmp, 'pronto'), os.path.join(tmp, 'quadros')

modelo = ModeloFalso(erro=0.2)
cd.vj.RFDetrTRT = lambda *a, **k: modelo
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa()

args = argparse.Namespace(engine='falso.engine', lote='L001', graos=500,
                          camera='csi', source=None, roi=0, tiles=1, conf=0.25,
                          class_offset=None, sem_trava=False, parado=False,
                          sem_janela=True, passo=2, bloco=2.0, guarda=0.3)
cd.capturar(args)

revisao = cd.ler_revisao()
print(f'\ngrãos na revisão: {len(revisao)}')
assert revisao, 'nada foi salvo para revisar'
assert all(r['prevista'] in CLASSES for r in revisao.values()), \
    'a classe proposta não veio no nome do arquivo'
assert all(r['prevista'] == r['corrigida'] for r in revisao.values()), \
    'recorte foi parar em pasta diferente da proposta'
print('OK: um recorte por grão, na pasta da classe proposta, com a proposta no nome')

# os QUADROS inteiros também têm que estar salvos, com as caixas
sessoes = cd._sessoes()
assert sessoes, 'nenhuma sessão de quadros salva'
_, meta = sessoes[0]
nq = len(meta['quadros'])
nc = sum(len(q['caixas']) for q in meta['quadros'])
print(f'OK: {nq} quadros inteiros salvos com {nc} caixas (dataset de detecção)')
assert nq > 0 and nc > nq, 'esperava várias caixas por quadro (cena multi-grão)'
assert meta['quadros'][0]['largura'] == TAM
# o mesmo grão tem que aparecer em VÁRIOS quadros — é o que faz a correção render
vezes = Counter(c['grao'] for q in meta['quadros'] for c in q['caixas'])
assert max(vezes.values()) > 1, 'nenhum grão apareceu em mais de um quadro'
print(f'OK: um grão aparece em até {max(vezes.values())} quadros — '
      f'corrigir 1 recorte acerta {max(vezes.values())} caixas')

# --- simula a correção manual: mover arquivos entre pastas ---
print('\n--- correção manual simulada ---')
rng = np.random.default_rng(3)
movidos = 0
for grao, r in list(revisao.items()):
    origem = os.path.join(cd.REVISAR, r['corrigida'], f"{grao}__{r['prevista']}.jpg")
    if rng.random() < 0.25:
        nova = CLASSES[(CLASSES.index(r['corrigida']) + 1) % len(CLASSES)]
        shutil.move(origem, os.path.join(cd.REVISAR, nova, os.path.basename(origem)))
        movidos += 1
desc = 0
for grao, r in list(cd.ler_revisao().items())[:3]:
    origem = os.path.join(cd.REVISAR, r['corrigida'], f"{grao}__{r['prevista']}.jpg")
    if os.path.exists(origem):
        shutil.move(origem, os.path.join(cd.REVISAR, cd.DESCARTE,
                                         os.path.basename(origem)))
        desc += 1
print(f'movi {movidos} para outra classe e {desc} para {cd.DESCARTE}/')

apos = cd.ler_revisao()
detectados = sum(1 for r in apos.values()
                 if r['prevista'] in CLASSES and r['prevista'] != r['corrigida'])
assert detectados >= movidos, f'detectou {detectados}, esperava >= {movidos}'
print(f'OK: o script detecta {detectados} correções lendo onde o arquivo ficou')

# --- exportação ---
print('\n--- exportação (YOLO + COCO) ---')
args.exportar, args.incluir_sem_trava = True, False
cd.exportar(args)

import glob
from collections import defaultdict
graos_split = defaultdict(set)
for sp in ('train', 'valid', 'test'):
    for p_ in glob.glob(os.path.join(cd.PRONTO, sp, 'labels', '*.txt')):
        nome = os.path.basename(p_)[:-4] + '.jpg'
        for _, m in cd._sessoes():
            for q in m['quadros']:
                if q['arquivo'] == nome:
                    graos_split[sp] |= {c['grao'] for c in q['caixas']}
for a in ('train', 'valid', 'test'):
    for b in ('train', 'valid', 'test'):
        if a < b:
            assert not (graos_split[a] & graos_split[b]), f'VAZAMENTO entre {a} e {b}'
print('OK: nenhum grão físico em dois splits (divisão por bloco + banda de guarda)')

# YOLO: rótulo normalizado, classe dentro da faixa, e é a classe CORRIGIDA
n_caixas = 0
for sp in ('train', 'valid', 'test'):
    for p_ in glob.glob(os.path.join(cd.PRONTO, sp, 'labels', '*.txt')):
        for linha in open(p_).read().splitlines():
            ci, cx, cy, w_, h_ = linha.split()
            assert 0 <= int(ci) < len(CLASSES)
            assert all(0 <= float(v) <= 1 for v in (cx, cy, w_, h_)), linha
            n_caixas += 1
        assert os.path.exists(os.path.join(cd.PRONTO, sp, 'images',
                                           os.path.basename(p_)[:-4] + '.jpg'))
print(f'OK: {n_caixas} rótulos YOLO normalizados, cada um com sua imagem')

# COCO: mesmo conteúdo, em pixels
tot_coco = 0
for sp in ('train', 'valid', 'test'):
    c = json.load(open(os.path.join(cd.PRONTO, sp, '_annotations.coco.json')))
    assert [k['name'] for k in c['categories']] == CLASSES
    for a_ in c['annotations']:
        assert 1 <= a_['category_id'] <= len(CLASSES)
        assert a_['bbox'][2] > 1 and a_['bbox'][3] > 1
    tot_coco += len(c['annotations'])
assert tot_coco == n_caixas, f'COCO {tot_coco} x YOLO {n_caixas}'
print(f'OK: COCO com as mesmas {tot_coco} caixas, em pixels')

# o grão descartado não pode ter deixado caixa em lugar nenhum
descartados = {g for g, r in apos.items() if r['corrigida'] == cd.DESCARTE}
for sp in ('train', 'valid', 'test'):
    for p_ in glob.glob(os.path.join(cd.PRONTO, sp, 'labels', '*.txt')):
        nome = os.path.basename(p_)[:-4] + '.jpg'
        for _, m in cd._sessoes():
            for q in m['quadros']:
                if q['arquivo'] == nome:
                    presentes = {c['grao'] for c in q['caixas']}
                    n_esperado = len(presentes - descartados
                                     - {g for g in presentes if g not in apos})
                    n_real = len(open(p_).read().splitlines())
                    assert n_real == n_esperado, (nome, n_real, n_esperado)
print(f'OK: as caixas dos {len(descartados)} grãos descartados sumiram das anotações')

# determinismo
antes = {os.path.relpath(p_, cd.PRONTO)
         for p_ in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*'))}
cd.exportar(args)
assert antes == {os.path.relpath(p_, cd.PRONTO)
                 for p_ in glob.glob(os.path.join(cd.PRONTO, '*', '*', '*'))}
print('OK: exportar de novo dá o mesmo resultado (hash, não sorteio)')

rel = open(os.path.join(cd.PRONTO, 'RELATORIO.md')).read()
assert 'Acurácia do modelo atual no rig' in rel
assert os.path.exists(os.path.join(cd.PRONTO, 'dataset.yaml'))
print('OK: RELATORIO.md com a acurácia medida + dataset.yaml')

# --- filtros de qualidade ---
print('\n--- filtros ---')
tmp2 = tempfile.mkdtemp()
cd.RAIZ, cd.REVISAR = tmp2, os.path.join(tmp2, 'revisar')
cd.PRONTO, cd.QUADROS = os.path.join(tmp2, 'pronto'), os.path.join(tmp2, 'quadros')
cd.vj.abrir_camera = lambda *a, **k: CameraFalsa(borrar=True)
args.exportar = False
cd.capturar(args)
assert not cd.ler_revisao(), 'quadro borrado passou pelo filtro de nitidez'
print('OK: cena borrada não gera nenhum recorte')

shutil.rmtree(tmp); shutil.rmtree(tmp2)
print('\n=== coleta validada ===')
