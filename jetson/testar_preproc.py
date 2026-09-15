#!/usr/bin/env python3
"""Confere que o pré-processamento otimizado produz o MESMO tensor do ingênuo.

Otimização que muda resultado não é otimização, é bug — e este seria calado: as
caixas continuariam saindo, só que de uma imagem levemente diferente da que o
modelo viu no treino. Aqui o caminho novo é comparado ao antigo pixel a pixel,
nos dois regimes (quadro já do tamanho da entrada, e quadro maior com borda).

    python3 testar_preproc.py
"""
import importlib.util
import os
import sys
import time
import types

import cv2
import numpy as np

sys.modules.setdefault('tensorrt', types.ModuleType('tensorrt'))
_spec = importlib.util.spec_from_file_location(
    'vj', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vigil_jetson.py'))
vj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vj)

H = W = 704


def referencia(frame, host):
    """O caminho antigo, palavra por palavra — é o gabarito."""
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((H, W, 3), np.uint8)
    dx, dy = (W - nw) // 2, (H - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    np.copyto(host, np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))
    return s, dx, dy


class Fake(vj.RFDetrTRT):
    """Só a parte de pré-processamento, sem TensorRT nem CUDA."""

    def __init__(self):
        self.H, self.W = H, W
        self.in_name = 'input'
        self.host = {'input': np.zeros((1, 3, H, W), np.float32)}
        self._canvas = np.zeros((H, W, 3), np.uint8)
        self._rgb8 = np.empty((H, W, 3), np.uint8)
        self._hwc32 = np.empty((H, W, 3), np.float32)


def caso(nome, frame):
    m = Fake()
    ref = np.zeros((1, 3, H, W), np.float32)
    s_r, dx_r, dy_r = referencia(frame, ref)
    s_n, dx_n, dy_n = m.preparar(frame)

    assert (s_n, dx_n, dy_n) == (s_r, dx_r, dy_r), \
        f'{nome}: transformação diferente {(s_n, dx_n, dy_n)} vs {(s_r, dx_r, dy_r)}'
    difmax = float(np.abs(m.host['input'] - ref).max())
    assert difmax == 0.0, f'{nome}: tensor diferente (máx {difmax})'

    def t(fn, n=100):
        fn(); a = time.perf_counter()
        for _ in range(n): fn()
        return (time.perf_counter() - a) / n * 1000

    ms_r = t(lambda: referencia(frame, ref))
    ms_n = t(lambda: m.preparar(frame))
    print(f'  {nome}')
    print(f'    tensor idêntico (diferença máxima {difmax}), '
          f'escala={s_n:.3f} borda=({dx_n},{dy_n})')
    print(f'    antigo {ms_r:5.2f} ms  ->  novo {ms_n:5.2f} ms   '
          f'({ms_r/ms_n:.1f}x)')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print('pré-processamento: novo x antigo')
    # o caso do rig: recorte 1:1, já do tamanho da entrada
    caso('quadro 704x704 (modo --roi/--tiles)',
         rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    # o caso binado: precisa reduzir e não precisa de borda (4:3 -> quadrado dá borda)
    caso('quadro 1640x1232 (modo binado, com borda)',
         rng.integers(0, 256, (1232, 1640, 3), dtype=np.uint8))
    # quadro alto: borda lateral, que é o caso de fatia NÃO contígua no canvas
    caso('quadro 400x900 (borda lateral)',
         rng.integers(0, 256, (900, 400, 3), dtype=np.uint8))
    # o canvas é reaproveitado entre chamadas: lixo do quadro anterior não pode sobrar
    m = Fake()
    m.preparar(np.full((900, 400, 3), 255, np.uint8))
    m.preparar(np.full((300, 900, 3), 255, np.uint8))
    ref = np.zeros((1, 3, H, W), np.float32)
    referencia(np.full((300, 900, 3), 255, np.uint8), ref)
    assert np.array_equal(m.host['input'], ref), 'sobrou lixo do quadro anterior no canvas'
    print('  buffer reaproveitado: sem resíduo do quadro anterior')
    print('\n=== pré-processamento validado ===')
