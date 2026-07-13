#!/usr/bin/env python3
"""Vígil.ia — câmera do celular LOCAL + inferência na GPU do Colab (remoto).

O Colab não enxerga o IP do seu Wi-Fi (é outra máquina, na nuvem) — por isso
não dá pra apontar o notebook direto pro DroidCam. O caminho é o inverso:
1. Rode `demo_servidor_colab.ipynb` até a célula do túnel Cloudflare.
   Ele imprime um link tipo https://xxxx.trycloudflare.com
2. Rode este script AQUI (no seu PC/Windows), passando esse link em --api.
   Ele lê o celular (local) e manda cada frame pro Colab; desenha o que voltar.

Exemplo:
  python testar_gpu_colab.py --api https://xxxx.trycloudflare.com --camera http://192.168.15.8:4747/video
  python testar_gpu_colab.py --api https://xxxx.trycloudflare.com --camera 0   # webcam/USB

Tecla: q = sair.
"""
import argparse
import base64
import time

import cv2
import requests

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


def inspect(api, frame, jpeg_quality, timeout):
    """Manda 1 frame pro servidor do Colab; devolve a lista de detecções (ou None se falhar)."""
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode()
    try:
        r = requests.post(f'{api}/inspect', json={'image': f'data:image/jpeg;base64,{b64}'},
                          timeout=timeout)
        r.raise_for_status()
        return r.json().get('detections', [])
    except requests.RequestException as exc:
        print(f'  [aviso] falha no /inspect: {exc}')
        return None


def draw(frame, detections):
    for d in detections:
        cls = d['class']
        color = COLORS.get(cls, (160, 160, 160))
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{PT_LABEL.get(cls, cls)} {d['confidence']:.2f}",
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    ap = argparse.ArgumentParser(description='Câmera local + inferência na GPU do Colab.')
    ap.add_argument('--api', required=True,
                    help='URL do túnel do demo_servidor_colab.ipynb, ex. https://xxxx.trycloudflare.com')
    ap.add_argument('--camera', default='0',
                    help='índice da câmera (0) ou URL DroidCam, ex. http://192.168.15.8:4747/video')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--quality', type=int, default=80, help='qualidade JPEG enviada (menor = mais rápido)')
    ap.add_argument('--timeout', type=float, default=8.0, help='timeout de cada chamada ao Colab (s)')
    args = ap.parse_args()

    api = args.api.rstrip('/')
    print(f'checando servidor em {api} …')
    try:
        r = requests.get(f'{api}/health', timeout=10)
        r.raise_for_status()
        print('servidor OK:', r.json())
    except requests.RequestException as exc:
        raise SystemExit(f'não consegui falar com {api}/health — confira o link e se a célula '
                         f'do túnel ainda está rodando no Colab. Erro: {exc}')

    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f'não abri a câmera {args.camera}.')
    print(f'câmera {args.camera} aberta | q para sair')

    win = 'Vigil.ia — GPU do Colab (q sai)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    detections, fps, t_prev = [], 0.0, time.time()
    n_req, t_lat = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('câmera parou de enviar frames.'); break
            if args.rotate:
                frame = rotate(frame, args.rotate)

            t0 = time.time()
            result = inspect(api, frame, args.quality, args.timeout)
            if result is not None:
                detections = result
                n_req += 1
                t_lat = 0.9 * t_lat + 0.1 * (time.time() - t0) if n_req > 1 else (time.time() - t0)

            draw(frame, detections)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            hud = f'{len(detections)} caixas   {fps:.1f} fps local   {t_lat*1000:.0f} ms/chamada'
            cv2.putText(frame, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
            cv2.putText(frame, hud, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(win, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
