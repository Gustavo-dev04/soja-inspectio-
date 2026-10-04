#!/usr/bin/env python3
"""Vígil.ia — testar as CAIXAS de um modelo (cru, sem voto nem suavização).

Serve pra julgar a qualidade da detecção: desenha TODA caixa que o modelo produz,
com a confiança, quadro a quadro. Nada é filtrado por voto temporal — é o que o
modelo realmente vê. Use quando as bounding boxes não parecem certas.

Fonte (--source):
  - arquivo de vídeo  -> gera um .mp4 anotado ao lado (ou em --out)
  - número (0,1,2…)   -> câmera ao vivo (janela)
  - URL http://…      -> stream (DroidCam por Wi-Fi), janela ao vivo

Exemplos:
  python testar_video.py --model soja_yolo11n_multi_v3.pt --source teste_soja.mp4
  python testar_video.py --model soja_yolo11s_final_v3.pt --source teste_soja.mp4 --conf 0.2
  python testar_video.py --model soja_yolo11n_multi_v3.pt --source 0        # webcam
  python testar_video.py --model soja_..._openvino_model --device intel:gpu --source video.mp4

Na janela ao vivo: q = sair.
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO

NAMES = ['broken', 'immature', 'intact', 'skin-damaged', 'spotted']
PT_LABEL = {'broken': 'Quebrado', 'immature': 'Imaturo', 'intact': 'Intacto',
            'skin-damaged': 'Casca danif.', 'spotted': 'Manchado'}
COLORS = {'intact': (90, 200, 90), 'immature': (60, 200, 200), 'broken': (170, 100, 210),
          'skin-damaged': (255, 160, 60), 'spotted': (70, 70, 235)}


def rotate(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def draw(frame, r, show_conf=True):
    """Desenha TODA caixa detectada no frame (crua, sem voto)."""
    n = 0
    if r.boxes is not None and len(r.boxes):
        for xyxy, c, cf in zip(r.boxes.xyxy.cpu().numpy().astype(int),
                               r.boxes.cls.int().tolist(),
                               r.boxes.conf.tolist()):
            cls = NAMES[c]
            color = COLORS[cls]
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            txt = f'{PT_LABEL[cls]} {cf:.2f}' if show_conf else PT_LABEL[cls]
            cv2.putText(frame, txt, (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description='Testa as caixas cruas de um modelo Vígil.ia.')
    ap.add_argument('--model', required=True, help='caminho do .pt ou pasta _openvino_model')
    ap.add_argument('--source', required=True,
                    help='vídeo (arquivo), número da câmera (0), ou URL http://…')
    ap.add_argument('--conf', type=float, default=0.25, help='confiança mínima (baixe p/ ver mais)')
    ap.add_argument('--imgsz', type=int, default=640, help='resolução (use 1280 p/ modelo HD)')
    ap.add_argument('--iou', type=float, default=0.6, help='IoU do NMS')
    ap.add_argument('--device', default=None, help='cpu | 0 | intel:gpu (OpenVINO)')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--out', default=None, help='saída .mp4 (só p/ arquivo; padrão: ao lado)')
    ap.add_argument('--no-conf', action='store_true', help='não escrever a confiança nas caixas')
    args = ap.parse_args()

    print(f'carregando {args.model} …')
    model = YOLO(args.model)

    is_file = os.path.isfile(args.source)
    show_conf = not args.no_conf

    def infer(frame):
        return model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                             agnostic_nms=True, device=args.device, verbose=False)[0]

    if is_file:
        # ---- arquivo de vídeo: gera .mp4 anotado ----
        cap = cv2.VideoCapture(args.source)
        assert cap.isOpened(), f'não abri o vídeo: {args.source}'
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out_path = args.out or (os.path.splitext(args.source)[0] + '_testado.mp4')
        writer, k, soma = None, 0, 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.rotate:
                frame = rotate(frame, args.rotate)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            soma += draw(frame, infer(frame), show_conf)
            writer.write(frame)
            k += 1
            if k % 30 == 0:
                pct = f'{100*k//total}%' if total else f'{k}'
                print(f'  {pct}  ({k} frames, {soma} caixas)…', flush=True)
        cap.release()
        writer.release()
        dt = time.time() - t0
        print(f'\npronto: {out_path}')
        print(f'{k} frames | {soma} caixas no total | média {soma/max(k,1):.1f} caixas/frame '
              f'| {k/max(dt,1e-6):.1f} fps de processamento')
        print('abra o .mp4 e veja se as caixas colam nos grãos.')
    else:
        # ---- câmera / URL: janela ao vivo ----
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        assert cap.isOpened(), (f'não abri a câmera/URL: {args.source} '
                                '(webcam USB, ou DroidCam http://IP:4747/video)')
        win = 'Vigil.ia — teste de caixas (q sai)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        fps, t_prev = 0.0, time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                print('câmera parou de enviar frames.'); break
            if args.rotate:
                frame = rotate(frame, args.rotate)
            n = draw(frame, infer(frame), show_conf)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            cv2.putText(frame, f'{n} caixas   {fps:.0f} fps', (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, f'{n} caixas   {fps:.0f} fps', (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
