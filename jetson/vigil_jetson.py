#!/usr/bin/env python3
"""Vígil.ia no Jetson — inspeção de soja ao vivo com RF-DETR + TensorRT.

Equivalente do `deck/vigil_deck.py`, mas consumindo a engine TensorRT do
RF-DETR em vez do .pt via ultralytics. Mesma regra de voto exigente por classe
e veredito travado por grão, pra o comportamento ser idêntico ao do Deck.

    python3 vigil_jetson.py                            # celular (DroidCam, padrão)
    python3 vigil_jetson.py --camera 0                 # webcam/USB
    python3 vigil_jetson.py --camera csi               # câmera CSI (conector da placa)
    python3 vigil_jetson.py --source video.mp4 --out saida.mp4   # arquivo

O padrão é o celular via DroidCam — veja CAMERA_PADRAO logo abaixo se o IP mudar.

Teclas: q sai · c zera a contagem · p pausa.

Dependências (além do JetPack): numpy, opencv, e pycuda OU cuda-python.
    sudo apt install python3-opencv
    pip3 install pycuda        # (ou: pip3 install cuda-python)

Nota de projeto: o rastreamento é um IoU tracker próprio, de ~40 linhas, em vez
de ByteTrack. Grão praticamente não se move entre quadros, então IoU basta — e
evita uma dependência que já quebrou em silêncio antes (o sv.ByteTrack
depreciado devolvia tracker_id vazio e zerava os votos sem erro nenhum).
"""
import argparse
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    sys.exit('tensorrt não encontrado — ele vem com o JetPack. Se estiver num '
             'venv, crie-o com --system-site-packages.')

# ---------------------------------------------------------------- constantes
# (idênticas ao deck/vigil_deck.py — o comportamento tem que bater)
NAMES = ['broken', 'immature', 'intact', 'skin-damaged', 'spotted']
PT_LABEL = {'broken': 'Quebrado', 'immature': 'Imaturo', 'intact': 'Intacto',
            'skin-damaged': 'Casca danif.', 'spotted': 'Manchado'}
COLORS = {'intact': (90, 200, 90), 'immature': (60, 200, 200),
          'broken': (170, 100, 210), 'skin-damaged': (255, 160, 60),
          'spotted': (70, 70, 235)}
RATIOS = {'broken': 0.85, 'skin-damaged': 0.80, 'spotted': 0.75, 'immature': 0.75}

# Câmera padrão: o celular via DroidCam (o app mostra o IP na tela ao abrir).
# Se o IP mudar — e ele muda quando o roteador renova o DHCP — troque aqui ou
# passe --camera na linha de comando.
CAMERA_PADRAO = 'http://192.168.15.5:4747/video'
LOCK_MIN_FRAMES = 8    # frames rastreando antes de travar a classe
MIN_DRAW_FRAMES = 3    # abaixo disso é ruído piscante: não desenha
SMOOTH = 0.4           # EMA da caixa (menor = mais estável)


def veredito(cnt):
    """Classe final de um grão a partir dos votos acumulados."""
    top, w = cnt.most_common(1)[0]
    if top == 'intact':
        return 'intact'
    return top if w >= RATIOS[top] * sum(cnt.values()) else 'intact'


# ---------------------------------------------------------------- CUDA
class Cuda:
    """Memória CUDA por pycuda ou cuda-python — o que estiver instalado."""

    def __init__(self):
        try:
            import pycuda.autoinit  # noqa: F401  (inicializa o contexto)
            import pycuda.driver as drv
            self.drv, self.api = drv, 'pycuda'
        except ImportError:
            try:
                from cuda import cudart
                self.rt, self.api = cudart, 'cuda-python'
            except ImportError:
                sys.exit(
                    'preciso de cuda-python OU pycuda pra alocar memória na GPU.\n'
                    '\n'
                    '  1) o pip existe?\n'
                    '       sudo apt install -y python3-pip python3-dev\n'
                    '\n'
                    '  2) tente o cuda-python (wheel pronto, não compila):\n'
                    '       pip3 install cuda-python\n'
                    '\n'
                    '  3) se der "externally-managed-environment" (Ubuntu novo):\n'
                    '       pip3 install cuda-python --break-system-packages\n'
                    '\n'
                    '  4) plano B — pycuda COMPILA, precisa do nvcc no PATH:\n'
                    '       export PATH=/usr/local/cuda/bin:$PATH\n'
                    '       export CUDA_ROOT=/usr/local/cuda\n'
                    '       pip3 install pycuda --break-system-packages\n'
                    '\n'
                    '  (num venv, crie com --system-site-packages: o tensorrt vem\n'
                    '   do sistema, via JetPack, e não do pip)')

    def alloc(self, nbytes):
        if self.api == 'pycuda':
            return self.drv.mem_alloc(nbytes)
        err, ptr = self.rt.cudaMalloc(nbytes)
        assert err == 0, f'cudaMalloc falhou: {err}'
        return ptr

    def h2d(self, dst, src):
        if self.api == 'pycuda':
            self.drv.memcpy_htod(dst, src)
        else:
            self.rt.cudaMemcpy(dst, src.ctypes.data, src.nbytes,
                               self.rt.cudaMemcpyKind.cudaMemcpyHostToDevice)

    def d2h(self, dst, src):
        if self.api == 'pycuda':
            self.drv.memcpy_dtoh(dst, src)
        else:
            self.rt.cudaMemcpy(dst.ctypes.data, src, dst.nbytes,
                               self.rt.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    def sync(self):
        if self.api == 'pycuda':
            self.drv.Context.synchronize()
        else:
            self.rt.cudaDeviceSynchronize()

    def ptr(self, buf):
        return int(buf)


# ---------------------------------------------------------------- modelo
class RFDetrTRT:
    """Carrega a engine e faz inferência num frame BGR."""

    def __init__(self, engine_path, conf=0.35):
        self.conf = conf
        self.cuda = Cuda()
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            sys.exit(f'não carreguei a engine {engine_path} — ela é atada a este '
                     'aparelho e à versão do TensorRT; reconstrua se o JetPack mudou.')
        self.ctx = self.engine.create_execution_context()

        self.nomes, self.entradas, self.saidas = [], [], []
        self.host, self.dev, self.shapes = {}, {}, {}
        for i in range(self.engine.num_io_tensors):
            nome = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(nome))
            dtype = trt.nptype(self.engine.get_tensor_dtype(nome))
            arr = np.empty(shape, dtype=dtype)
            self.host[nome] = arr
            self.dev[nome] = self.cuda.alloc(arr.nbytes)
            self.shapes[nome] = shape
            self.nomes.append(nome)
            eh_in = self.engine.get_tensor_mode(nome) == trt.TensorIOMode.INPUT
            (self.entradas if eh_in else self.saidas).append(nome)
            self.ctx.set_tensor_address(nome, self.cuda.ptr(self.dev[nome]))

        self.in_name = self.entradas[0]
        _, _, self.H, self.W = self.shapes[self.in_name]
        # 'dets' = caixas (…,4) | 'labels' = logits (…,C)
        self.box_name = next(n for n in self.saidas if self.shapes[n][-1] == 4)
        self.cls_name = next(n for n in self.saidas if n != self.box_name)
        self.n_cls = self.shapes[self.cls_name][-1]
        # COCO é 1-indexado: com 5 classes o modelo sai com 6 colunas e a 0 é
        # fundo. Detecta em vez de assumir (foi assim que o off-by-one passou
        # despercebido antes).
        self.offset = self.n_cls - len(NAMES)
        print(f'engine: entrada {self.W}x{self.H} | {self.shapes[self.box_name][1]} queries '
              f'| {self.n_cls} colunas de classe -> offset {self.offset}')
        if self.offset not in (0, 1):
            print(f'  [aviso] esperava 5 ou 6 colunas, achei {self.n_cls}. '
                  'Confira o mapeamento de classe.')

    def letterbox(self, frame):
        """Redimensiona mantendo proporção e preenche com preto (igual ao treino)."""
        h, w = frame.shape[:2]
        s = min(self.W / w, self.H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        dx, dy = (self.W - nw) // 2, (self.H - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, s, dx, dy

    def __call__(self, frame):
        """Devolve [(x1,y1,x2,y2,classe_idx,conf), …] em coordenadas do frame."""
        canvas, s, dx, dy = self.letterbox(frame)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

        np.copyto(self.host[self.in_name], blob)
        self.cuda.h2d(self.dev[self.in_name], self.host[self.in_name])
        self.ctx.execute_async_v3(0)
        self.cuda.sync()
        for nome in self.saidas:
            self.cuda.d2h(self.host[nome], self.dev[nome])

        boxes = self.host[self.box_name][0].astype(np.float32)     # (N,4)
        logits = self.host[self.cls_name][0].astype(np.float32)    # (N,C)
        probs = 1.0 / (1.0 + np.exp(-logits))                      # sigmoid (focal loss)

        # ignora a coluna de fundo quando ela existe
        cols = probs[:, self.offset:self.offset + len(NAMES)]
        cls_idx = cols.argmax(1)
        conf = cols.max(1)
        keep = conf >= self.conf
        if not keep.any():
            return []
        boxes, cls_idx, conf = boxes[keep], cls_idx[keep], conf[keep]

        # cxcywh normalizado (padrão DETR) ou já em pixels? decide pelo valor
        if boxes.max() <= 1.5:
            cx, cy, bw, bh = boxes.T
            x1 = (cx - bw / 2) * self.W
            y1 = (cy - bh / 2) * self.H
            x2 = (cx + bw / 2) * self.W
            y2 = (cy + bh / 2) * self.H
        else:
            x1, y1, x2, y2 = boxes.T

        # desfaz o letterbox -> coordenadas do frame original
        x1 = (x1 - dx) / s
        y1 = (y1 - dy) / s
        x2 = (x2 - dx) / s
        y2 = (y2 - dy) / s
        h, w = frame.shape[:2]
        x1 = np.clip(x1, 0, w - 1); x2 = np.clip(x2, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1); y2 = np.clip(y2, 0, h - 1)

        # NMS agnóstico de classe (paridade com o pipeline do Colab)
        rects = np.stack([x1, y1, x2 - x1, y2 - y1], 1).tolist()
        idx = cv2.dnn.NMSBoxes(rects, conf.tolist(), self.conf, 0.6)
        if len(idx) == 0:
            return []
        idx = np.array(idx).ravel()
        return [(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]),
                 int(cls_idx[i]), float(conf[i])) for i in idx]


# ---------------------------------------------------------------- tracker
class IoUTracker:
    """Rastreio por IoU. Grão quase não se move entre quadros, então isso basta —
    e não depende de biblioteca externa (o sv.ByteTrack depreciado já zerou
    votos em silêncio uma vez)."""

    def __init__(self, iou_min=0.3, sumido_max=8):
        self.iou_min, self.sumido_max = iou_min, sumido_max
        self.prox_id, self.tracks = 1, {}   # id -> [caixa, sumido_ha]

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, dets):
        saida, usados = [], set()
        for det in dets:
            caixa = det[:4]
            melhor, melhor_iou = None, self.iou_min
            for tid, (prev, _) in self.tracks.items():
                if tid in usados:
                    continue
                v = self._iou(caixa, prev)
                if v > melhor_iou:
                    melhor, melhor_iou = tid, v
            if melhor is None:
                melhor = self.prox_id
                self.prox_id += 1
            usados.add(melhor)
            self.tracks[melhor] = [caixa, 0]
            saida.append((melhor, *det))
        for tid in list(self.tracks):
            if tid not in usados:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.sumido_max:
                    del self.tracks[tid]
        return saida


# ---------------------------------------------------------------- câmera
def abrir_camera(spec, largura=1280, altura=720):
    if spec == 'csi':
        pipe = (f'nvarguscamerasrc ! video/x-raw(memory:NVMM),width={largura},'
                f'height={altura},framerate=30/1 ! nvvidconv ! video/x-raw,'
                f'format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1')
        return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec)
    if spec.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap


def rotate(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Vígil.ia no Jetson (RF-DETR + TensorRT)')
    ap.add_argument('--engine', default='soja_rfdetr_small_CAMPEAO_fp16.engine')
    ap.add_argument('--camera', default=CAMERA_PADRAO,
                    help=f"índice (0), URL do DroidCam, ou 'csi'. "
                         f"padrão: {CAMERA_PADRAO}")
    ap.add_argument('--source', default=None, help='arquivo de vídeo (em vez da câmera)')
    ap.add_argument('--out', default=None, help='grava a saída anotada em .mp4')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--hold', type=float, default=3.0,
                    help='segundos observando o grão antes de travar a classe')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--no-window', action='store_true', help='sem janela (só terminal/arquivo)')
    args = ap.parse_args()

    print(f'carregando {args.engine} …')
    modelo = RFDetrTRT(args.engine, conf=args.conf)
    tracker = IoUTracker()

    fonte = args.source or args.camera
    cap = (cv2.VideoCapture(args.source) if args.source
           else abrir_camera(args.camera))
    if not cap.isOpened():
        msg = [f'não abri a fonte: {fonte}']
        if str(fonte).startswith('http'):
            msg += ['  • o DroidCam está aberto no celular?',
                    '  • o IP bate com o que o app mostra? (muda quando o roteador'
                    ' renova o DHCP)',
                    '  • Jetson e celular na MESMA rede Wi-Fi?',
                    f'  • teste: curl -sI {fonte} | head -1']
        sys.exit('\n'.join(msg))
    print(f'fonte: {fonte} | q sai · c zera · p pausa')

    votos = defaultdict(Counter)
    visto, primeiro, travado, suave = Counter(), {}, {}, {}
    writer = None
    fps, t_prev, pausado = 0.0, time.time(), False
    win = 'Vigil.ia Jetson (q sai)'
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not pausado:
                ok, frame = cap.read()
                if not ok:
                    print('fim da fonte.')
                    break
                if args.rotate:
                    frame = rotate(frame, args.rotate)

                t0 = time.time()
                dets = modelo(frame)
                ms = (time.time() - t0) * 1000
                agora = time.time()

                for tid, x1, y1, x2, y2, cls_i, cf in tracker.update(dets):
                    nome = NAMES[cls_i] if 0 <= cls_i < len(NAMES) else 'intact'
                    votos[tid][nome] += cf
                    visto[tid] += 1
                    primeiro.setdefault(tid, agora)
                    # trava a classe só depois de observar o suficiente
                    if (tid not in travado and visto[tid] >= LOCK_MIN_FRAMES
                            and agora - primeiro[tid] >= args.hold):
                        travado[tid] = veredito(votos[tid])
                    if visto[tid] < MIN_DRAW_FRAMES:
                        continue
                    if tid in suave:            # EMA: caixa não treme
                        px1, py1, px2, py2 = suave[tid]
                        x1 = int(SMOOTH * x1 + (1 - SMOOTH) * px1)
                        y1 = int(SMOOTH * y1 + (1 - SMOOTH) * py1)
                        x2 = int(SMOOTH * x2 + (1 - SMOOTH) * px2)
                        y2 = int(SMOOTH * y2 + (1 - SMOOTH) * py2)
                    suave[tid] = (x1, y1, x2, y2)
                    cls = travado.get(tid)
                    cor = COLORS[cls] if cls else (160, 160, 160)
                    rot = PT_LABEL[cls] if cls else 'analisando...'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                    cv2.putText(frame, rot, (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

                dist = Counter(travado.values())
                bons = dist.get('intact', 0)
                ruins = sum(v for k, v in dist.items() if k != 'intact')
                hud = (f'{len(dets)} graos  |  Premium {bons}  Expulso {ruins}  |  '
                       f'{fps:.0f} fps  {ms:.0f} ms')
                cv2.putText(frame, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(frame, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if args.out:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                                 cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
                    writer.write(frame)

            if not args.no_window:
                cv2.imshow(win, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('p'):
                    pausado = not pausado
                if k == ord('c'):
                    votos.clear(); visto.clear(); primeiro.clear()
                    travado.clear(); suave.clear()
                    tracker = IoUTracker()
                    print('contagem zerada')
    finally:
        cap.release()
        if writer:
            writer.release()
            print('salvo:', args.out)
        if not args.no_window:
            cv2.destroyAllWindows()

    dist = Counter(travado.values())
    print('\n=== veredito final ===')
    for c in NAMES:
        print(f'  {PT_LABEL[c]:14s} {dist.get(c, 0)}')
    total = sum(dist.values())
    if total:
        print(f'  Premium: {dist.get("intact", 0)}/{total} '
              f'({100 * dist.get("intact", 0) / total:.0f}%)')


if __name__ == '__main__':
    main()
