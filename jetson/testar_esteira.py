#!/usr/bin/env python3
"""Vígil.ia — testa o modo esteira do vigil_jetson SEM hardware.

Roda no PC, sem Jetson, sem câmera e sem engine: o que ele exercita é a parte
que não depende de GPU nenhuma — rastreamento com movimento, junção dos
recortes e contabilidade do laudo. São exatamente os três pontos que quebram em
silêncio quando o grão passa a se mover, cada um com o número que
`calcular_vazao.py` prevê.

    python3 testar_esteira.py
"""
import argparse
import importlib.util
import os
import sys
import time
import types
from collections import Counter, defaultdict

import numpy as np

# vigil_jetson importa tensorrt no topo (vem do JetPack); no PC ele não existe,
# e nada do que se testa aqui usa TensorRT.
sys.modules.setdefault('tensorrt', types.ModuleType('tensorrt'))
_spec = importlib.util.spec_from_file_location(
    'vj', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vigil_jetson.py'))
vj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vj)

GRAO = 89                      # px: o grão no rig (12,7 px/mm x 7 mm)
PASSO = int(GRAO * 1.4)        # espaçamento em monocamada
LADO, N_TILES = 704, 2
SOBREPOR = int(LADO * 0.2)


def fila(desloc, n_varr, compensar, n_graos=5):
    """n grãos em fila na esteira; devolve o pior nº de IDs recebidos por um grão."""
    tr = vj.IoUTracker(compensar=compensar)
    hist = {}
    for k in range(n_varr):
        dets = [(60 + k * desloc + i * PASSO, 100,
                 60 + k * desloc + i * PASSO + GRAO, 100 + GRAO, 2, 0.9)
                for i in range(n_graos)]
        for i, (tid, *_) in enumerate(tr.update(dets)):
            hist.setdefault(i, set()).add(tid)
    return max(len(v) for v in hist.values())


def teste_rastreamento():
    print('--- rastreamento na esteira ---')
    print(f'grão {GRAO} px, espaçamento {PASSO} px, 12 varreduras, 5 grãos em fila')
    print(f'{"desloc":>8} {"= grãos":>8} | {"IDs/grão IoU cru":>17} {"compensado":>11}')
    falhou = 0
    for d in (10, 20, 30, 40, 48, 55, 62):
        s, c = fila(d, 12, False), fila(d, 12, True)
        assert c == 1, f'compensação falhou em {d} px ({d/GRAO:.2f} grão): {c} IDs'
        if s > 1:
            falhou += 1
        print(f'{d:6d}px {d/GRAO:7.2f} | {s:17d} {c:11d}'
              + ('   <- IoU cru PERDE o grão' if s > 1 else ''))
    assert falhou >= 3, 'o teste deveria reproduzir a falha do IoU cru'
    print(f'OK: compensação mantém 1 ID/grão até 0,70 grão/varredura '
          f'(IoU cru falhou em {falhou} casos)')

    # acima de meio espaçamento é ambíguo DE VERDADE — o teste registra o limite
    # em vez de escondê-lo: o vizinho da frente fica mais perto do que o grão
    # andou, e nenhum algoritmo desempata isso sem outra fonte de informação.
    assert fila(int(GRAO * 0.9), 12, True) > 1, 'esperava ambiguidade acima do limite'
    print('OK: a 0,90 grão/varredura vira ambíguo — limite físico, não bug')

    assert fila(0, 12, True) == 1 and fila(0, 12, False) == 1
    print('OK: grão parado (bandeja) segue com 1 ID nos dois modos')

    tr = vj.IoUTracker(compensar=True)
    for _ in range(4):
        tr.update([(50, 50, 50 + GRAO, 50 + GRAO, 2, .9)])
    saiu = []
    for _ in range(12):
        tr.update([])
        saiu += tr.aposentados
    assert saiu == [1], f'esperava aposentar o id 1 uma vez, veio {saiu}'
    assert not tr.tracks, 'o dicionário de tracks deveria esvaziar'
    print('OK: track que sai de quadro é aposentado uma vez e some da memória')


class _ModeloFalso:
    """Devolve detecções em coords do RECORTE, uma lista por chamada."""
    W = LADO

    def __init__(self, por_tile):
        self.por_tile, self.chamadas = por_tile, 0

    def __call__(self, tile):
        self.chamadas += 1
        return list(self.por_tile[self.chamadas - 1])


def teste_recortes():
    print('\n--- recortes lado a lado ---')
    frame = np.zeros((800, 1500, 3), np.uint8)
    tiles = vj.crop_tiles(frame, LADO, N_TILES, SOBREPOR)
    xs = [dx for _, dx, _ in tiles]
    cobertura = xs[-1] + LADO - xs[0]
    assert len(tiles) == N_TILES
    assert all(t.shape[:2] == (LADO, LADO) for t, _, _ in tiles)
    assert cobertura == N_TILES * LADO - (N_TILES - 1) * SOBREPOR
    assert xs[0] >= 0 and xs[-1] + LADO <= frame.shape[1]
    print(f'OK: {N_TILES} recortes de {LADO}px, sobreposição {SOBREPOR}px '
          f'-> {cobertura}px de cobertura, centrados')

    assert len(vj.crop_tiles(np.zeros((800, 800, 3), np.uint8), LADO, 3, SOBREPOR)) == 1
    print('OK: frame estreito reduz o nº de recortes em vez de estourar')

    # grão em cima da costura: aparece nos DOIS recortes, na mesma posição global
    gx = xs[1] + 20
    modelo = _ModeloFalso([
        [(gx - xs[0], 300, gx - xs[0] + GRAO, 300 + GRAO, 2, 0.90),
         (100, 500, 100 + GRAO, 500 + GRAO, 0, 0.80)],
        [(gx - xs[1], 300, gx - xs[1] + GRAO, 300 + GRAO, 2, 0.88)],
    ])
    dets = vj.detectar(modelo, frame, argparse.Namespace(tiles=N_TILES, roi=LADO),
                       SOBREPOR)
    assert modelo.chamadas == N_TILES
    assert len(dets) == 2, f'a costura duplicou o grão: {len(dets)} detecções'
    assert sorted(d[0] for d in dets) == [xs[0] + 100, gx]
    print('OK: coords globais corretas e o grão da costura conta UMA vez')


def teste_contagem_e_memoria(n_varr=4000):
    """Jornada comprimida: a contagem tem que fechar e a memória, estabilizar."""
    print(f'\n--- contabilidade e memória ({n_varr} varreduras) ---')
    DESL, JANELA = int(GRAO * 0.5), LADO
    votos = defaultdict(Counter)
    visto, primeiro, travado, suave = Counter(), {}, {}, {}
    contagem = Counter()
    tr = vj.IoUTracker(compensar=True)
    rng = np.random.default_rng(0)
    vivos, prox, esperado, pico = [], 0.0, 0, None

    for k in range(n_varr):
        prox += DESL
        while prox >= PASSO:                     # entra grão novo
            prox -= PASSO
            vivos.append([-GRAO, 'broken' if rng.random() < 0.2 else 'intact'])
            esperado += 1
        for g in vivos:
            g[0] += DESL
        vivos = [g for g in vivos if g[0] < JANELA]
        dets = [(int(g[0]), 100, int(g[0]) + GRAO, 100 + GRAO,
                 vj.NAMES.index(g[1]), 0.95) for g in vivos if g[0] >= 0]

        for tid, x1, y1, x2, y2, ci, cf in tr.update(dets):
            votos[tid][vj.NAMES[ci]] += cf
            visto[tid] += 1
            primeiro.setdefault(tid, k * 0.07)
            if tid not in travado and visto[tid] >= vj.LOCK_MIN_FRAMES:
                travado[tid] = vj.veredito(votos[tid])
            suave[tid] = (x1, y1, x2, y2)
        for tid in tr.aposentados:               # mesma lógica do laço do app
            cls = travado.pop(tid, None)
            if cls is None and visto.get(tid, 0) >= vj.MIN_DRAW_FRAMES and votos[tid]:
                cls = vj.veredito(votos[tid])
            if cls:
                contagem[cls] += 1
            votos.pop(tid, None); visto.pop(tid, None)
            primeiro.pop(tid, None); suave.pop(tid, None)
        if k == 500:
            pico = (len(votos), len(visto), len(travado), len(suave), len(tr.tracks))

    fim = (len(votos), len(visto), len(travado), len(suave), len(tr.tracks))
    total = sum(contagem.values()) + len(travado)
    print(f'passaram {esperado} grãos, contados {total} '
          f'({sum(contagem.values())} acumulados + {len(travado)} em quadro)')
    print(f'dicionários: varredura 500 {pico} -> fim {fim}')
    assert abs(total - esperado) <= 8, f'contagem {total} vs {esperado} reais'
    assert all(f <= p + 2 for f, p in zip(fim, pico)), f'memória cresceu: {pico}->{fim}'
    assert max(fim) < 40, f'dicionários grandes demais: {fim}'
    print('OK: contagem fecha (diferença = grãos ainda em trânsito)')
    print('OK: memória estável — não cresce com a produção do dia')

    frac = contagem['broken'] / max(1, sum(contagem.values()))
    assert 0.12 < frac < 0.28, frac
    print(f'OK: {100*frac:.0f}% classificados como broken (a simulação injetou 20%)')


def teste_laudo():
    print('\n--- laudo ---')
    args = argparse.Namespace(engine='e.engine', conf=.35, roi=LADO,
                              tiles=N_TILES, esteira=True)
    arq = 'laudo_teste.json'
    l = vj.escrever_laudo(arq, Counter({'intact': 900, 'broken': 60, 'spotted': 40}),
                          time.time() - 60, 27.5, args, 'teste')
    try:
        assert (l['graos'], l['premium'], l['premium_pct']) == (1000, 900, 90.0)
        assert abs(l['massa_estimada_kg'] - 0.16) < 1e-6
        assert abs(l['vazao_kg_h'] - 9.6) < 0.1        # 0,16 kg em 60 s
        assert l['por_classe']['immature'] == 0
        print(f'OK: {l["graos"]} grãos, {l["premium_pct"]}% premium, '
              f'{l["massa_estimada_kg"]} kg, {l["vazao_kg_h"]} kg/h')
    finally:
        os.path.exists(arq) and os.remove(arq)


if __name__ == '__main__':
    teste_rastreamento()
    teste_recortes()
    teste_contagem_e_memoria()
    teste_laudo()
    print('\n=== tudo passou ===')
