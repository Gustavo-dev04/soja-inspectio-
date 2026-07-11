#!/usr/bin/env python3
"""Vígil.ia — inspeção de soja AO VIVO, 100% local no Steam Deck.

Sem internet, sem site, sem servidor. Carrega o campeão .pt, abre a câmera,
detecta e classifica cada grão em tempo real com voto temporal exigente por
classe (mata a alucinação de defeito nos grãos bons).

Câmera:
  - Por padrão usa o dispositivo 0.
  - CELULAR via USB (modo webcam UVC do Android 14+): aparece como outro
    /dev/videoN — rode:
        python vigil_deck.py --list-cameras     (descobre o índice)
        python vigil_deck.py --camera 2          (usa o celular)
  - CELULAR sem modo webcam USB (ex. XOS/Infinix): use o DroidCam por Wi-Fi —
    abra o app no celular (mesma rede Wi-Fi do Deck) e passe a URL dele:
        python vigil_deck.py --camera http://192.168.0.15:4747/video

Teclas na janela: q = sair | espaço = zera a contagem | p = pausa.

Captura p/ treino futuro (--save-dir PASTA): salva o recorte mais nítido de cada
grão com veredito fechado + um revisao.csv (mesmo formato do
model/aprendizado_ativo.ipynb) pra revisar e reaproveitar no re-treino.
"""
import argparse
import csv
import os
import time
from collections import defaultdict, Counter

import cv2
from ultralytics import YOLO

# ordem das classes = índices do modelo (validada no treino)
NAMES = ['broken', 'immature', 'intact', 'skin-damaged', 'spotted']
PT_LABEL = {'broken': 'Quebrado', 'immature': 'Imaturo', 'intact': 'Intacto',
            'skin-damaged': 'Casca danif.', 'spotted': 'Manchado'}
# cores BGR (OpenCV)
COLORS = {'intact': (90, 200, 90), 'immature': (60, 200, 200), 'broken': (170, 100, 210),
          'skin-damaged': (255, 160, 60), 'spotted': (70, 70, 235)}

# --- voto exigente POR CLASSE (calibrado no vídeo): defeito só é aceito se a
#     classe tiver >= ratio dos votos ponderados; senão o grão vira 'intact'
#     (benefício da dúvida). Ajuste aqui se precisar. ---
RATIOS = {'broken': 0.85, 'skin-damaged': 0.80, 'spotted': 0.75, 'immature': 0.75}
LOCK_MIN_FRAMES = 8   # frames mínimos rastreando antes de travar a classe do grão
MIN_DRAW_FRAMES = 3   # só desenha o grão depois disso (mata detecção piscante de 1-2 frames)
SMOOTH = 0.4          # suavização da caixa (EMA): menor = mais estável, menos treme


def veredito(cnt: Counter) -> str:
    """Classe final de um grão a partir dos votos acumulados."""
    top, w = cnt.most_common(1)[0]
    if top == 'intact':
        return 'intact'
    return top if w >= RATIOS[top] * sum(cnt.values()) else 'intact'


def list_cameras(max_idx=10):
    print(f'procurando câmeras (0..{max_idx})…')
    achou = []
    for i in range(max_idx + 1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            achou.append(i)
            print(f'  câmera {i}: OK')
    if not achou:
        print('  nenhuma. No celular: ative o modo webcam USB (Android 14+) ou DroidCam.')
    else:
        print('use --camera <índice>. O de maior número costuma ser o celular USB.')
    return achou


def rotate(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw_hud(frame, locked, fps, paused, capturando):
    dist = Counter(locked.values())
    n = sum(dist.values())
    intact = dist.get('intact', 0)
    linhas = [f'graos: {n}   intactos: {intact}   defeitos: {n - intact}',
              f'{fps:.0f} fps' + ('   [PAUSA]' if paused else '')
              + ('   ● capturando p/ treino' if capturando else '')]
    for cls in ('broken', 'immature', 'skin-damaged', 'spotted'):
        if dist.get(cls):
            linhas.append(f'  {PT_LABEL[cls]}: {dist[cls]}')
    y = 26
    for txt in linhas:
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 26


def main():
    ap = argparse.ArgumentParser(description='Vígil.ia ao vivo no Steam Deck (local).')
    ap.add_argument('--model', default='soja_yolo11n_base12k_v2.pt',
                    help='caminho do .pt (padrão: ao lado do script)')
    ap.add_argument('--camera', default='0',
                    help='índice da câmera (0 = padrão) OU URL de stream, ex. '
                         'http://192.168.0.15:4747/video (DroidCam por Wi-Fi)')
    ap.add_argument('--imgsz', type=int, default=640, help='resolução (480 = mais fps)')
    ap.add_argument('--conf', type=float, default=0.35, help='confiança mínima da detecção')
    ap.add_argument('--hold', type=float, default=3.0,
                    help='segundos observando cada grão antes de travar a classe '
                         '(padrão 3; suba p/ 5 se travar cedo demais). Segure a câmera parada.')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270],
                    help='girar a imagem (útil p/ celular em pé)')
    ap.add_argument('--list-cameras', action='store_true', help='lista as câmeras e sai')
    ap.add_argument('--device', default=None,
                    help='dispositivo de inferência: cpu, 0 (GPU CUDA) ou, com modelo '
                         'exportado p/ OpenVINO, "intel:gpu" / "intel:cpu" / "intel:npu"')
    ap.add_argument('--save-dir', default=None,
                    help='pasta onde salvar recortes+revisao.csv p/ treino futuro '
                         '(cria uma subpasta por sessão; formato igual ao '
                         'model/aprendizado_ativo.ipynb)')
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    print(f'carregando {args.model} …')
    model = YOLO(args.model)

    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(
            f'não abri a câmera {args.camera}. Rode --list-cameras pra ver as disponíveis '
            '(no celular: modo webcam USB, ou DroidCam por Wi-Fi com '
            '--camera http://IP_DO_CELULAR:4747/video).')
    print(f'câmera {args.camera} aberta | q para sair')

    session_dir = None
    if args.save_dir:
        session_dir = os.path.join(args.save_dir, time.strftime('sessao_%Y%m%d_%H%M%S'))
        os.makedirs(os.path.join(session_dir, 'graos'), exist_ok=True)
        print(f'capturando p/ treino futuro em: {session_dir}')

    votes = defaultdict(Counter)   # tid -> votos ponderados por confiança
    seen = Counter()               # tid -> nº de frames
    first_seen = {}                # tid -> timestamp do 1º frame (trava por tempo)
    locked = {}                    # tid -> classe travada (anti-flicker)
    smooth = {}                    # tid -> caixa suavizada (x1,y1,x2,y2 float) p/ desenho
    best = {}                      # tid -> (nitidez, crop_bgr, conf) — só usado com --save-dir
    t_prev, fps, paused = time.time(), 0.0, False

    win = 'Vigil.ia — Steam Deck (q sai)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('câmera parou de enviar frames.'); break
            if args.rotate:
                frame = rotate(frame, args.rotate)
            now = time.time()

            if not paused:
                # detecta+rastreia no frame JÁ rotacionado -> caixas sempre batem
                r = model.track(frame, imgsz=args.imgsz, conf=args.conf, iou=0.5,
                                agnostic_nms=True, tracker='bytetrack.yaml',
                                persist=True, verbose=False, device=args.device)[0]
                if r.boxes.id is not None:
                    for xyxy, tid, c, cf in zip(r.boxes.xyxy.cpu().numpy().astype(int),
                                                r.boxes.id.int().tolist(),
                                                r.boxes.cls.int().tolist(),
                                                r.boxes.conf.tolist()):
                        if tid not in locked:
                            first_seen.setdefault(tid, now)
                            votes[tid][NAMES[c]] += cf
                            seen[tid] += 1
                            # trava só depois de frames E tempo suficientes observando
                            if seen[tid] >= LOCK_MIN_FRAMES and now - first_seen[tid] >= args.hold:
                                locked[tid] = veredito(votes[tid])
                        if session_dir:
                            x1c, y1c, x2c, y2c = xyxy
                            crop = frame[max(0, y1c):y2c, max(0, x1c):x2c]
                            if crop.size:
                                nitidez = cv2.Laplacian(
                                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                                if tid not in best or nitidez > best[tid][0]:
                                    best[tid] = (nitidez, crop.copy(), cf)
                        cls = locked.get(tid)
                        # ignora detecção piscante (grão visto por poucos frames) até estabilizar
                        if cls is None and seen[tid] < MIN_DRAW_FRAMES:
                            continue
                        # suaviza a caixa (EMA) pra não tremer frame a frame
                        x1, y1, x2, y2 = xyxy
                        if tid in smooth:
                            px1, py1, px2, py2 = smooth[tid]
                            x1 = int(SMOOTH * x1 + (1 - SMOOTH) * px1)
                            y1 = int(SMOOTH * y1 + (1 - SMOOTH) * py1)
                            x2 = int(SMOOTH * x2 + (1 - SMOOTH) * px2)
                            y2 = int(SMOOTH * y2 + (1 - SMOOTH) * py2)
                        smooth[tid] = (x1, y1, x2, y2)
                        color = COLORS[cls] if cls else (160, 160, 160)
                        label = PT_LABEL[cls] if cls else 'analisando...'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'#{tid} {label}', (x1, max(18, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            draw_hud(frame, locked, fps, paused, bool(session_dir))
            cv2.imshow(win, frame)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord(' '):
                votes.clear(); seen.clear(); first_seen.clear()
                locked.clear(); smooth.clear(); best.clear()
            if k == ord('p'):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if session_dir:
            prontos = sorted(tid for tid in locked if tid in best)
            for tid in prontos:
                cv2.imwrite(f'{session_dir}/graos/{tid:04d}_{locked[tid]}.jpg', best[tid][1])
            with open(f'{session_dir}/revisao.csv', 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['id', 'classe_prevista', 'confianca', 'n_frames', 'classe_corrigida'])
                for tid in prontos:
                    w.writerow([tid, locked[tid], round(best[tid][2], 3), seen[tid], ''])
            print(f'{len(prontos)} grãos salvos em {session_dir}/graos/ | revise em '
                  f'{session_dir}/revisao.csv (mesmo fluxo do model/aprendizado_ativo.ipynb)')


if __name__ == '__main__':
    main()
