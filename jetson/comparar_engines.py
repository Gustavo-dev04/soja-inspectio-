#!/usr/bin/env python3
"""Vígil.ia — compara engines TensorRT no MESMO vídeo.

Rodar cada modelo ao vivo em momentos diferentes não compara nada: cada um vê
frames diferentes e a conclusão vira memória. Aqui os dois recebem exatamente os
mesmos quadros.

    # compara num vídeo que você já tem
    python3 comparar_engines.py --engines ft1.engine ft4.engine --source teste.mp4

    # grava 15s da câmera e compara nesse clipe
    python3 comparar_engines.py --engines ft1.engine ft4.engine --gravar 15

Gera um .mp4 anotado por engine e uma tabela lado a lado.

⚠️ Sobre domain shift: comparação RELATIVA (qual dos dois vai melhor no mesmo
input) continua válida mesmo fora do domínio de treino, porque o viés atinge os
dois igualmente. O que NÃO vale ainda é ler os números como acurácia absoluta —
ver a seção de padronização de captura no README.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vigil_jetson import (  # noqa: E402
    COLORS, LOCK_MIN_FRAMES, MIN_DRAW_FRAMES, NAMES, PT_LABEL, SMOOTH,
    IoUTracker, RFDetrTRT, abrir_camera, rotate, veredito,
)


def grava_clipe(camera, segundos, destino, rot=0):
    """Grava um clipe cru da câmera, pra os dois modelos verem o mesmo."""
    cap = abrir_camera(camera)
    ok, frame = cap.read()
    if not ok or frame is None:
        sys.exit(f'não consegui ler da câmera {camera}')
    if rot:
        frame = rotate(frame, rot)
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr = cv2.VideoWriter(destino, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    print(f'gravando {segundos}s de {camera} ({w}x{h}) -> {destino}')
    print('  posicione os grãos agora…')
    t0, n = time.time(), 0
    while time.time() - t0 < segundos:
        ok, frame = cap.read()
        if not ok:
            break
        if rot:
            frame = rotate(frame, rot)
        wr.write(frame)
        n += 1
        if n % 30 == 0:
            print(f'  {time.time() - t0:.0f}s…', flush=True)
    cap.release(); wr.release()
    print(f'gravado: {n} frames')
    return destino


def roda(engine_path, video, conf, hold, offset, salvar):
    """Passa o vídeo inteiro por uma engine. Devolve as métricas."""
    modelo = RFDetrTRT(engine_path, conf=conf, offset=offset)
    tracker = IoUTracker()
    votos, visto, primeiro, travado, suave = (defaultdict(Counter), Counter(),
                                              {}, {}, {})
    cap = cv2.VideoCapture(video)
    wr, k, n_det, t_inf = None, 0, 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        dets = modelo(frame)
        t_inf += time.time() - t0
        n_det += len(dets)
        # o tempo do vídeo, não o relógio: o veredito tem que travar no mesmo
        # ponto pros dois modelos, independente de quem processa mais rápido
        agora = k / 30.0
        for tid, x1, y1, x2, y2, ci, cf in tracker.update(dets):
            nome = NAMES[ci] if 0 <= ci < len(NAMES) else 'intact'
            votos[tid][nome] += cf
            visto[tid] += 1
            primeiro.setdefault(tid, agora)
            if (tid not in travado and visto[tid] >= LOCK_MIN_FRAMES
                    and agora - primeiro[tid] >= hold):
                travado[tid] = veredito(votos[tid])
            if visto[tid] < MIN_DRAW_FRAMES:
                continue
            if tid in suave:
                px1, py1, px2, py2 = suave[tid]
                x1 = int(SMOOTH * x1 + (1 - SMOOTH) * px1)
                y1 = int(SMOOTH * y1 + (1 - SMOOTH) * py1)
                x2 = int(SMOOTH * x2 + (1 - SMOOTH) * px2)
                y2 = int(SMOOTH * y2 + (1 - SMOOTH) * py2)
            suave[tid] = (x1, y1, x2, y2)
            cls = travado.get(tid)
            cor = COLORS[cls] if cls else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(frame, PT_LABEL[cls] if cls else '...',
                        (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
        if salvar:
            if wr is None:
                h, w = frame.shape[:2]
                wr = cv2.VideoWriter(salvar, cv2.VideoWriter_fourcc(*'mp4v'),
                                     cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
            cv2.putText(frame, os.path.basename(engine_path), (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            wr.write(frame)
        k += 1
    cap.release()
    if wr:
        wr.release()
    dist = Counter(travado.values())
    return {
        'dist': dist,
        'graos': sum(dist.values()),
        'premium': dist.get('intact', 0),
        'expulso': sum(v for c, v in dist.items() if c != 'intact'),
        'det_frame': n_det / max(k, 1),
        'ms': (t_inf / max(k, 1)) * 1000,
        'frames': k,
        'hist': modelo.hist_cru.copy(),
    }


def main():
    ap = argparse.ArgumentParser(description='Compara engines no mesmo vídeo')
    ap.add_argument('--engines', nargs='+', required=True)
    ap.add_argument('--source', default=None, help='vídeo pra comparar')
    ap.add_argument('--gravar', type=float, default=0,
                    help='grava N segundos da câmera e compara nesse clipe')
    ap.add_argument('--camera', default=None, help='câmera pra --gravar')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--hold', type=float, default=3.0)
    ap.add_argument('--class-offset', type=int, default=None)
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    args = ap.parse_args()

    if args.gravar:
        from vigil_jetson import CAMERA_PADRAO
        video = grava_clipe(args.camera or CAMERA_PADRAO, args.gravar,
                            'clipe_comparacao.mp4', args.rotate)
    elif args.source:
        video = args.source
    else:
        sys.exit('use --source video.mp4 ou --gravar 15')
    if not os.path.exists(video):
        sys.exit(f'vídeo não encontrado: {video}')

    res = {}
    for eng in args.engines:
        if not os.path.exists(eng):
            print(f'{eng}: não existe, pulando')
            continue
        tag = os.path.basename(eng).replace('.engine', '')
        print(f'\n>>> {tag}')
        saida = f'comparacao_{tag}.mp4'
        res[tag] = roda(eng, video, args.conf, args.hold, args.class_offset, saida)
        r = res[tag]
        print(f'    {r["frames"]} frames | {r["graos"]} grãos | '
              f'{r["det_frame"]:.1f} det/frame | {r["ms"]:.0f} ms -> {saida}')

    if not res:
        sys.exit('nenhuma engine rodou')

    print('\n' + '=' * 78)
    print(f'{"engine":32s} {"Premium":>8s} {"Expulso":>8s} {"grãos":>7s} '
          f'{"det/frm":>8s} {"ms":>6s}')
    print('-' * 78)
    for tag, r in res.items():
        print(f'{tag[:32]:32s} {r["premium"]:8d} {r["expulso"]:8d} {r["graos"]:7d} '
              f'{r["det_frame"]:8.1f} {r["ms"]:6.0f}')
    print('\npor classe:')
    print(f'  {"engine":32s} ' + ' '.join(f'{PT_LABEL[c][:9]:>10s}' for c in NAMES))
    for tag, r in res.items():
        print(f'  {tag[:32]:32s} ' + ' '.join(f'{r["dist"].get(c, 0):10d}' for c in NAMES))

    print('\nComo ler:')
    print('  • os dois viram os MESMOS frames, então a diferença é do modelo')
    print('  • assista os comparacao_*.mp4 lado a lado: o número diz QUANTOS,')
    print('    só o vídeo diz QUAL está certo')
    print('  • ms/frame quase igual é esperado (mesma arquitetura); diferença')
    print('    grande aí indica problema na engine, não no treino')
    print('  • acurácia ABSOLUTA ainda não vale — falta padronizar a captura.')
    print('    A comparação relativa entre os dois, sim, porque o viés é o mesmo.')


if __name__ == '__main__':
    main()
