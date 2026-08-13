#!/usr/bin/env python3
"""Vígil.ia no Jetson — inspeção de soja ao vivo com RF-DETR + TensorRT.

Equivalente do `deck/vigil_deck.py`, mas consumindo a engine TensorRT do
RF-DETR em vez do .pt via ultralytics. Mesma regra de voto exigente por classe
e veredito travado por grão, pra o comportamento ser idêntico ao do Deck.

    python3 vigil_jetson.py                            # celular (DroidCam, padrão)
    python3 vigil_jetson.py --camera 0                 # webcam/USB
    python3 vigil_jetson.py --camera csi               # câmera CSI (conector da placa)
    python3 vigil_jetson.py --source video.mp4 --out saida.mp4   # arquivo

MODO RIG (câmara de inspeção com esteira — ver jetson/PADRAO_CAPTURA.md):

    python3 vigil_jetson.py --camera csi --roi 704 --tiles 2 --esteira \\
                            --laudo laudo.json

`--tiles 2` divide a faixa de 100 mm em dois recortes 1:1 lado a lado, cobrindo
a largura toda sem reescalar. `--esteira` liga a compensação de movimento no
rastreamento e trava o veredito por contagem de varreduras, não por segundos.
Dimensione antes com `calcular_vazao.py`, que diz a distância da câmera e a
velocidade máxima da esteira para a configuração escolhida.

O padrão é o celular via DroidCam — veja CAMERA_PADRAO logo abaixo se o IP mudar.

Teclas: q sai · c zera a contagem · p pausa.

Dependências (além do JetPack): numpy, opencv, e pycuda OU cuda-python.
    sudo apt install python3-opencv
    pip3 install pycuda        # (ou: pip3 install cuda-python)

Nota de projeto: o rastreamento é um IoU tracker próprio, de ~100 linhas, em vez
de ByteTrack — evita uma dependência que já quebrou em silêncio antes (o
sv.ByteTrack depreciado devolvia tracker_id vazio e zerava os votos sem erro
nenhum). Com o grão parado, IoU cru basta; na esteira não basta, e o porquê está
no docstring do IoUTracker.
"""
import argparse
import json
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
MASSA_GRAO_G = 0.16    # peso de mil grãos 120-200 g -> massa estimada no laudo


def veredito(cnt):
    """Classe final de um grão a partir dos votos acumulados."""
    top, w = cnt.most_common(1)[0]
    if top == 'intact':
        return 'intact'
    return top if w >= RATIOS[top] * sum(cnt.values()) else 'intact'


# ---------------------------------------------------------------- CUDA
def _cuda_runtime():
    """Importa o runtime do cuda-python.

    O layout mudou entre as versões: no 12.x era `from cuda import cudart`;
    no 13.x virou `cuda.bindings.runtime`. Tenta os dois em vez de assumir.
    """
    import importlib
    for mod in ('cuda.bindings.runtime', 'cuda.cudart', 'cuda.runtime'):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    return None


class Cuda:
    """Memória CUDA por pycuda ou cuda-python — o que estiver instalado."""

    def __init__(self):
        try:
            import pycuda.autoinit  # noqa: F401  (inicializa o contexto)
            import pycuda.driver as drv
            self.drv, self.api = drv, 'pycuda'
        except ImportError:
            rt = _cuda_runtime()
            if rt is not None:
                self.rt, self.api = rt, 'cuda-python'
                print(f'CUDA via cuda-python ({rt.__name__})')
            else:
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

    @staticmethod
    def _ok(ret, oque):
        """cuda-python devolve (err, …) — erro silencioso aqui vira detecção
        lixo depois, então falha alto."""
        err = ret[0] if isinstance(ret, (tuple, list)) else ret
        if int(err) != 0:
            raise RuntimeError(f'{oque} falhou (código CUDA {int(err)})')
        return ret

    def alloc(self, nbytes):
        if self.api == 'pycuda':
            return self.drv.mem_alloc(nbytes)
        return self._ok(self.rt.cudaMalloc(nbytes), 'cudaMalloc')[1]

    def h2d(self, dst, src):
        if self.api == 'pycuda':
            self.drv.memcpy_htod(dst, src)
        else:
            self._ok(self.rt.cudaMemcpy(
                dst, src.ctypes.data, src.nbytes,
                self.rt.cudaMemcpyKind.cudaMemcpyHostToDevice), 'cudaMemcpy H2D')

    def d2h(self, dst, src):
        if self.api == 'pycuda':
            self.drv.memcpy_dtoh(dst, src)
        else:
            self._ok(self.rt.cudaMemcpy(
                dst.ctypes.data, src, dst.nbytes,
                self.rt.cudaMemcpyKind.cudaMemcpyDeviceToHost), 'cudaMemcpy D2H')

    def sync(self):
        if self.api == 'pycuda':
            self.drv.Context.synchronize()
        else:
            self._ok(self.rt.cudaDeviceSynchronize(), 'cudaDeviceSynchronize')

    def ptr(self, buf):
        return int(buf)


# ---------------------------------------------------------------- modelo
class RFDetrTRT:
    """Carrega a engine e faz inferência num frame BGR."""

    def __init__(self, engine_path, conf=0.35, offset=None):
        self.conf = conf
        self.offset_forcado = offset
        self.hist_cru = None   # histograma das colunas cruas (diagnóstico)
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
        # Com 5 classes o modelo pode sair com 6 colunas — mas ONDE fica a coluna
        # extra depende da implementação: COCO 1-indexado põe o fundo na FRENTE,
        # o DETR clássico põe o "no-object" no FIM. Errar isso desloca todos os
        # rótulos (foi o que fez `intact` sair como `immature`).
        # Padrão = 0 (extra no fim). Use --class-offset pra forçar, e o
        # histograma abaixo pra decidir com dado em vez de chute.
        self.offset = self.offset_forcado if self.offset_forcado is not None else 0
        self.hist_cru = np.zeros(self.n_cls, np.int64)
        print(f'engine: entrada {self.W}x{self.H} | {self.shapes[self.box_name][1]} queries '
              f'| {self.n_cls} colunas de classe | offset={self.offset}'
              f'{" (forçado)" if self.offset_forcado is not None else ""}')
        if self.n_cls > len(NAMES):
            print(f'  há {self.n_cls - len(NAMES)} coluna(s) a mais que classes — '
                  'rode --diag pra ver qual coluna realmente dispara')

    def diagnostico_classes(self):
        """Qual coluna crua está ganhando? Diz onde estão as classes de verdade."""
        tot = self.hist_cru.sum()
        if not tot:
            return 'nenhuma detecção ainda'
        linhas = ['  coluna crua | detecções | com offset atual seria']
        for c in range(self.n_cls):
            i = c - self.offset
            nome = NAMES[i] if 0 <= i < len(NAMES) else '(fora)'
            barra = '#' * int(40 * self.hist_cru[c] / tot)
            linhas.append(f'  {c:^11d} | {self.hist_cru[c]:9d} | {nome:14s} {barra}')
        return '\n'.join(linhas)

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

        # histograma das colunas CRUAS: é o que diz onde as classes realmente
        # estão, sem depender do offset estar certo
        cru = probs.argmax(1)
        forte = probs.max(1) >= self.conf
        if forte.any():
            np.add.at(self.hist_cru, cru[forte], 1)

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
    """Rastreio por IoU, com predição de movimento opcional.

    Não depende de biblioteca externa (o sv.ByteTrack depreciado já zerou votos
    em silêncio uma vez).

    Com o grão PARADO, IoU cru basta. Com ele numa esteira, não: duas caixas de
    7 mm deslocadas de d têm IoU 0,33 em d=3,5 mm e 0,27 em d=4 mm — ou seja, o
    limite prático é ~0,54 grão por varredura. Passando disso o grão troca de ID
    no meio da travessia, os votos zeram e a contagem infla, tudo em silêncio.
    Ver `calcular_vazao.py`, que reporta esse limite em mm/s.

    Por isso cada track guarda a própria velocidade (média móvel do
    deslocamento do centro) e a caixa é PREVISTA antes de casar. É um preditor
    de velocidade constante em 1ª ordem — o suficiente para esteira, que é
    movimento uniforme, e barato o bastante para não pesar no laço.

    Duas peças fazem isso funcionar de verdade, e sem elas a compensação não sai
    do lugar:

    * **Fluxo global.** Track recém-criado tem velocidade zero, então a primeira
      previsão dele é a caixa parada — e o casamento falha logo no primeiro
      passo, antes de aprender qualquer velocidade. Como numa esteira TODO grão
      anda igual, o track novo nasce com a velocidade mediana da cena.
    * **Porta por distância.** Quando a sobreposição zera (deslocamento perto de
      um grão inteiro), IoU não distingue nada. Aí vale a distância entre o
      centro previsto e o detectado: o grão anda ~0,5 grão por varredura,
      enquanto o vizinho mais próximo está a ~1,4 grão. A porta separa os dois
      com folga, e ainda cobre o arranque frio da primeira varredura.
    """

    def __init__(self, iou_min=0.3, sumido_max=8, compensar=True, inercia=0.6,
                 porta=0.7):
        self.iou_min, self.sumido_max = iou_min, sumido_max
        self.compensar, self.inercia, self.porta = compensar, inercia, porta
        self.prox_id = 1
        self.tracks = {}          # id -> [caixa, sumido_ha, (vx, vy)]
        self.aposentados = []     # ids que sumiram de vez desde o último update
        self.fluxo = (0.0, 0.0)   # velocidade mediana da cena (esteira)

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _previsto(self, tid):
        """Onde a caixa deve estar AGORA, dado onde estava e como se movia."""
        caixa, sumido, (vx, vy) = self.tracks[tid]
        if not self.compensar:
            return caixa
        # `sumido+1` porque a previsão é para a varredura atual: se o track
        # ficou 2 varreduras sem aparecer, ele andou 3 passos desde a última vez.
        k = sumido + 1
        return (caixa[0] + vx * k, caixa[1] + vy * k,
                caixa[2] + vx * k, caixa[3] + vy * k)

    @staticmethod
    def _centro(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def _casa(self, caixa, usados):
        """Melhor track para esta caixa: IoU primeiro, distância como reserva.

        O score de distância é mantido abaixo de `iou_min` de propósito, para
        que qualquer casamento por IoU sempre ganhe de um casamento por porta.
        """
        lado = ((caixa[2] - caixa[0]) + (caixa[3] - caixa[1])) / 2
        limite = self.porta * lado
        cx, cy = self._centro(caixa)
        melhor, melhor_score = None, self.iou_min
        reserva, reserva_d = None, limite
        for tid in self.tracks:
            if tid in usados:
                continue
            prev = self._previsto(tid)
            v = self._iou(caixa, prev)
            if v > melhor_score:
                melhor, melhor_score = tid, v
            elif self.compensar and melhor is None:
                px, py = self._centro(prev)
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < reserva_d:
                    reserva, reserva_d = tid, d
        return melhor if melhor is not None else reserva

    def update(self, dets):
        saida, usados, deslocs = [], set(), []
        self.aposentados = []
        for det in dets:
            caixa = det[:4]
            melhor = self._casa(caixa, usados)
            if melhor is None:
                melhor = self.prox_id
                self.prox_id += 1
                # nasce com a velocidade da cena: sem isso a primeira previsão
                # é a caixa parada e o casamento falha antes de aprender nada
                self.tracks[melhor] = [caixa, 0,
                                       self.fluxo if self.compensar else (0.0, 0.0)]
            else:
                ant, sumido, (vx, vy) = self.tracks[melhor]
                k = sumido + 1
                ax, ay = self._centro(ant)
                cx, cy = self._centro(caixa)
                dx, dy = (cx - ax) / k, (cy - ay) / k
                deslocs.append((dx, dy))
                a = self.inercia
                self.tracks[melhor] = [caixa, 0,
                                       (a * vx + (1 - a) * dx, a * vy + (1 - a) * dy)]
            usados.add(melhor)
            saida.append((melhor, *det))
        if deslocs:
            self.fluxo = (float(np.median([d[0] for d in deslocs])),
                          float(np.median([d[1] for d in deslocs])))
        for tid in list(self.tracks):
            if tid not in usados:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.sumido_max:
                    del self.tracks[tid]
                    self.aposentados.append(tid)
        return saida


# ---------------------------------------------------------------- câmera
# ---- câmera CSI (IMX219) do rig padronizado -------------------------------
# 1640x1232 é o modo binado do IMX219: FOV COMPLETO a 30 fps. Os modos 1080p
# recortam o sensor, ou seja, mudam o enquadramento — o que quebraria a
# padronização entre sessões.
CSI_LARGURA, CSI_ALTURA, CSI_FPS = 1640, 1232, 30

# Modo ROI: sensor CHEIO (3280x2464 @ 21 fps) para recortar janelas em escala
# 1:1, sem reduzir. É o que faz a lente de 120° render neste projeto.
#
# O problema nunca foi o sensor, foi o DOWNSCALE: reduzir o quadro inteiro de
# 1640 px até a entrada do modelo deixa o grão com ~16 px, contra os 60-150 px
# em que o modelo é treinado. Recortando 1:1 do sensor cheio o grão chega no
# tamanho certo — e usando justamente a região onde a 120° distorce menos.
#
# Configuração do rig (câmara de 100 mm de faixa, câmera a ~8,4 cm):
#   campo 233 mm em 3280 px   ->  14,1 px/mm
#   2 recortes de 704 px      ->  2 x 50 mm = os 100 mm da faixa  (--tiles 2)
#   grão de 7 mm              ->  ~99 px  (dentro do alvo 60-150)
#
# Confira o px/mm real com a régua (calibrar_rig.py) e dimensione a vazão com
# calcular_vazao.py antes de congelar a geometria.
CSI_ROI_LARGURA, CSI_ROI_ALTURA, CSI_ROI_FPS = 3280, 2464, 21

# Exposição e balanço de branco TRAVADOS. Em automático a câmera compensa
# sozinha entre sessões e recria o domain shift que o rig existe pra eliminar —
# é o jeito mais silencioso de invalidar um dataset inteiro.
# Ajuste EXPOSICAO_NS olhando o histograma no rig (ver jetson/PADRAO_CAPTURA.md).
CSI_TRAVAS = {
    'wbmode': 0,            # 0 = balanço de branco manual (desliga o automático)
    'awblock': 'true',      # trava o AWB
    'aelock': 'true',       # trava a exposição automática
    'exposuretimerange': '"13000000 13000000"',   # ns — fixo, não faixa
    'gainrange': '"1 1"',                          # ganho analógico fixo
    'ispdigitalgainrange': '"1 1"',                # ganho digital fixo
}


def pipeline_csi(sensor=0, largura=CSI_LARGURA, altura=CSI_ALTURA, fps=CSI_FPS,
                 travar=True):
    travas = ' '.join(f'{k}={v}' for k, v in CSI_TRAVAS.items()) if travar else ''
    return (f'nvarguscamerasrc sensor-id={sensor} {travas} ! '
            f'video/x-raw(memory:NVMM),width={largura},height={altura},'
            f'framerate={fps}/1 ! nvvidconv ! video/x-raw,format=BGRx ! '
            f'videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2')


def abrir_camera(spec, largura=1280, altura=720, travar_csi=True, roi=0):
    if spec == 'csi':
        if roi:   # sensor cheio: a janela vem do recorte, não do downscale
            pipe = pipeline_csi(largura=CSI_ROI_LARGURA, altura=CSI_ROI_ALTURA,
                                fps=CSI_ROI_FPS, travar=travar_csi)
            print(f'CSI (IMX219) {CSI_ROI_LARGURA}x{CSI_ROI_ALTURA}@{CSI_ROI_FPS} '
                  f'-> ROI central {roi}x{roi} (1:1, sem reescalar)')
        else:
            pipe = pipeline_csi(travar=travar_csi)
            print(f'CSI (IMX219) {CSI_LARGURA}x{CSI_ALTURA}@{CSI_FPS} (binado)')
        print(f'  AE/AWB {"TRAVADOS" if travar_csi else "AUTOMÁTICOS (não padronizado!)"}')
        return cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec)
    if spec.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
    return cap


def crop_roi(frame, lado):
    """Recorta uma janela quadrada de `lado` px no centro, SEM reescalar.

    Diferente do crop_quadrado (que pega o maior quadrado possível e depois é
    reduzido), aqui o recorte já sai no tamanho da entrada do modelo. É o que
    preserva a densidade de pixel por grão com lente grande angular.
    """
    h, w = frame.shape[:2]
    lado = min(lado, h, w)
    x, y = (w - lado) // 2, (h - lado) // 2
    return frame[y:y + lado, x:x + lado]


def crop_tiles(frame, lado, n, sobrepor=0):
    """Divide a faixa central em `n` recortes quadrados lado a lado.

    Devolve [(recorte, dx, dy), …], onde (dx, dy) é o canto do recorte no frame
    — é o que permite mapear as caixas de volta pra coordenadas globais e deixar
    o tracker trabalhar num quadro só.

    `sobrepor` (px) faz os recortes se cruzarem, pra grão em cima da costura não
    ser cortado ao meio nos dois lados. Quem chama junta com NMS depois.
    """
    h, w = frame.shape[:2]
    lado = min(lado, h)
    passo = lado - sobrepor
    total = passo * (n - 1) + lado
    if total > w:                       # não cabe: reduz o nº de recortes
        n = max(1, (w - lado) // passo + 1) if passo > 0 else 1
        total = passo * (n - 1) + lado
    x0, y0 = (w - total) // 2, (h - lado) // 2
    return [(frame[y0:y0 + lado, x:x + lado], x, y0)
            for x in (x0 + i * passo for i in range(n))]


def nms_global(dets, iou_max=0.6):
    """NMS agnóstico de classe sobre caixas já em coordenadas globais.

    Só faz diferença na faixa de sobreposição entre recortes; sem isso o grão da
    costura entra duas vezes e o laudo conta a mais.
    """
    if len(dets) < 2:
        return dets
    rects = [[d[0], d[1], d[2] - d[0], d[3] - d[1]] for d in dets]
    scores = [float(d[5]) for d in dets]
    idx = cv2.dnn.NMSBoxes(rects, scores, 0.0, iou_max)
    if len(idx) == 0:
        return dets
    return [dets[i] for i in np.array(idx).ravel()]


def crop_quadrado(frame):
    """Recorta o quadrado central.

    Serve a dois propósitos de uma vez: descarta as bordas da grande angular de
    120°, onde a distorção de barril é pior, e evita o letterbox — o modelo come
    512x512 quadrado, então uma imagem 4:3 gastaria ~25% da entrada em barra
    preta.
    """
    h, w = frame.shape[:2]
    lado = min(h, w)
    x, y = (w - lado) // 2, (h - lado) // 2
    return frame[y:y + lado, x:x + lado]


def rotate(frame, deg):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def enquadrar(frame, args):
    """Recorte + rotação, iguais em todos os caminhos (diag, 1º frame, laço)."""
    if args.roi and args.tiles <= 1:
        frame = crop_roi(frame, args.roi)
    elif args.quadrado:
        frame = crop_quadrado(frame)
    if args.rotate:
        frame = rotate(frame, args.rotate)
    return frame


def detectar(modelo, frame, args, sobrepor):
    """Detecta no frame, em N recortes se pedido, sempre em coords globais."""
    if args.tiles <= 1:
        return modelo(frame)
    lado = args.roi or modelo.W
    dets = []
    for tile, dx, dy in crop_tiles(frame, lado, args.tiles, sobrepor):
        for x1, y1, x2, y2, ci, cf in modelo(tile):
            dets.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, ci, cf))
    return nms_global(dets)


def escrever_laudo(caminho, contagem, t0, fps, args, extra=''):
    """Laudo do lote: é o produto que o MVP entrega.

    Gravado periodicamente para que uma jornada de 14-16 h não perca tudo se o
    app cair no fim.
    """
    total = sum(contagem.values())
    premium = contagem.get('intact', 0)
    dur = time.time() - t0
    laudo = {
        'gerado_em': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duracao_s': round(dur, 1),
        'graos': total,
        'premium': premium,
        'nao_premium': total - premium,
        'premium_pct': round(100 * premium / total, 2) if total else 0.0,
        'por_classe': {k: contagem.get(k, 0) for k in NAMES},
        'massa_estimada_kg': round(total * MASSA_GRAO_G / 1000, 4),
        'vazao_kg_h': round(total * MASSA_GRAO_G / 1000 * 3600 / dur, 2) if dur > 0 else 0.0,
        'graos_por_s': round(total / dur, 2) if dur > 0 else 0.0,
        'fps': round(fps, 1),
        'config': {'engine': args.engine, 'conf': args.conf, 'roi': args.roi,
                   'tiles': args.tiles, 'esteira': bool(args.esteira),
                   'massa_grao_g': MASSA_GRAO_G},
        'obs': extra,
    }
    with open(caminho, 'w') as f:
        json.dump(laudo, f, ensure_ascii=False, indent=2)
    return laudo


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
                    help='segundos observando o grão antes de travar a classe '
                         '(ignorado em --esteira)')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument('--roi', type=int, default=0, metavar='PX',
                    help='captura no sensor CHEIO e recorta PX x PX do centro, '
                         'sem reescalar (rig CSI: use 704 — ver PADRAO_CAPTURA.md §1c)')
    ap.add_argument('--tiles', type=int, default=1, metavar='N',
                    help='divide a faixa em N recortes de --roi px lado a lado, '
                         'cobrindo mais largura sem reescalar (rig: 2)')
    ap.add_argument('--esteira', action='store_true',
                    help='grão em movimento: trava por contagem de varreduras '
                         '(hold=0) e liga a compensação de movimento no tracker')
    ap.add_argument('--parado', action='store_true',
                    help='desliga a compensação de movimento (bandeja estática)')
    ap.add_argument('--laudo', default=None, metavar='ARQ.json',
                    help='grava o laudo (contagem, %% premium, massa) e o '
                         'atualiza a cada --laudo-seg')
    ap.add_argument('--laudo-seg', type=float, default=300,
                    help='de quantos em quantos segundos regravar o laudo')
    ap.add_argument('--quadrado', action='store_true',
                    help='recorta o quadrado central (rig CSI: tira a distorção '
                         'das bordas da 120° e evita o letterbox)')
    ap.add_argument('--csi-sem-trava', action='store_true',
                    help='CSI com AE/AWB automáticos — só para ajustar o rig, '
                         'NUNCA para capturar dataset')
    ap.add_argument('--no-window', action='store_true', help='sem janela (só terminal/arquivo)')
    ap.add_argument('--class-offset', type=int, default=None,
                    help='desloca a leitura das colunas de classe (0 = extra no fim, '
                         '1 = extra na frente). Use --diag pra descobrir o certo.')
    ap.add_argument('--diag', type=int, default=0, metavar='N',
                    help='processa N frames, imprime qual coluna de classe dispara, e sai')
    args = ap.parse_args()

    if args.esteira:
        args.hold = 0.0
    print(f'carregando {args.engine} …')
    modelo = RFDetrTRT(args.engine, conf=args.conf, offset=args.class_offset)
    tracker = IoUTracker(compensar=not args.parado)
    # sobreposição entre recortes: precisa ser MAIOR que um grão, senão o grão da
    # costura sai cortado nos dois lados. 20% de 704 px = 141 px; a ~12,7 px/mm
    # do rig isso são 11 mm, ~1,6 grãos. Ver calcular_vazao.py --sobrepor.
    sobrepor = int((args.roi or modelo.W) * 0.2) if args.tiles > 1 else 0
    if args.tiles > 1:
        print(f'recortes: {args.tiles} x {args.roi or modelo.W}px, '
              f'sobreposição {sobrepor}px')

    fonte = args.source or args.camera
    cap = (cv2.VideoCapture(args.source) if args.source
           else abrir_camera(args.camera, travar_csi=not args.csi_sem_trava,
                            roi=args.roi))
    # isOpened() NÃO basta em stream de rede: o GStreamer abre um pipeline vazio
    # e devolve True mesmo sem conexão. Só ler um frame de verdade comprova.
    quadro0 = None
    if cap.isOpened():
        for _ in range(15):
            ok, quadro0 = cap.read()
            if ok and quadro0 is not None:
                quadro0 = enquadrar(quadro0, args)
                break
            time.sleep(0.2)
        else:
            quadro0 = None
    if quadro0 is None:
        msg = [f'a fonte abriu mas não veio nenhum frame: {fonte}', '']
        if str(fonte).startswith('http'):
            ip = str(fonte).split('//')[-1].split(':')[0]
            msg += ['  Parece rede. Confira, nesta ordem:',
                    f'    1. o Jetson enxerga o celular?   ping -c2 {ip}',
                    '    2. qual o IP do Jetson?           ip -4 addr | grep inet',
                    '       (Jetson e celular precisam estar na MESMA faixa,',
                    '        ex. ambos 192.168.15.x — Wi-Fi vs cabo costuma separar)',
                    '    3. o DroidCam está aberto e mostrando esse IP na tela?',
                    f'    4. o stream responde?             curl -sI {fonte} | head -1',
                    '',
                    '  Pra testar o resto do app sem depender da rede:',
                    '    python3 vigil_jetson.py --source teste_soja.mp4 --out saida.mp4']
        else:
            msg += ['  Câmera local: veja os índices disponíveis com',
                    '    ls /dev/video*']
        sys.exit('\n'.join(msg))
    print(f'fonte OK: {quadro0.shape[1]}x{quadro0.shape[0]}')

    # ---- modo diagnóstico: descobre onde estão as classes de verdade ----
    if args.diag:
        print(f'\ndiagnóstico: {args.diag} frames…')
        frame = quadro0
        for i in range(args.diag):
            if i:
                ok, frame = cap.read()
                if not ok:
                    break
            detectar(modelo, enquadrar(frame, args), args, sobrepor)
        cap.release()
        print('\n=== onde as classes realmente estão ===')
        print(modelo.diagnostico_classes())
        print('\nComo ler: a coluna com MAIS detecções deve ser a classe mais comum')
        print('no seu vídeo (normalmente `intact`). Se a coluna campeã não estiver')
        print('caindo em `intact` na tabela acima, ajuste --class-offset:')
        print('  --class-offset 0  -> coluna extra fica no FIM   (col 0 = broken)')
        print('  --class-offset 1  -> coluna extra fica na FRENTE (col 1 = broken)')
        return
    print(f'fonte: {fonte} | q sai · c zera · p pausa')

    votos = defaultdict(Counter)
    visto, primeiro, travado, suave = Counter(), {}, {}, {}
    # Contagem ACUMULADA: numa jornada de 14-16 h passam ~3 milhões de grãos, e
    # os dicionários acima são por track. Quando o track sai de quadro o verdito
    # dele é dobrado aqui e as entradas são liberadas — senão a memória cresce
    # linearmente com a produção do dia.
    contagem = Counter()
    writer = None
    t_inicio, prox_laudo = time.time(), time.time() + args.laudo_seg
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
                frame = enquadrar(frame, args)

                t0 = time.time()
                dets = detectar(modelo, frame, args, sobrepor)
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

                # aposenta os tracks que saíram: dobra o verdito no acumulado e
                # libera as entradas por track (ver `contagem`, acima)
                for tid in tracker.aposentados:
                    cls = travado.pop(tid, None)
                    if cls is None and visto.get(tid, 0) >= MIN_DRAW_FRAMES and votos[tid]:
                        cls = veredito(votos[tid])   # saiu antes de travar: usa o voto
                    if cls:
                        contagem[cls] += 1
                    votos.pop(tid, None); visto.pop(tid, None)
                    primeiro.pop(tid, None); suave.pop(tid, None)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

                if args.laudo and now >= prox_laudo:
                    escrever_laudo(args.laudo, contagem + Counter(travado.values()),
                                   t_inicio, fps, args, 'parcial')
                    prox_laudo = now + args.laudo_seg

                dist = contagem + Counter(travado.values())
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
                    travado.clear(); suave.clear(); contagem.clear()
                    tracker = IoUTracker(compensar=not args.parado)
                    t_inicio = time.time()
                    print('contagem zerada')
    finally:
        cap.release()
        if writer:
            writer.release()
            print('salvo:', args.out)
        if not args.no_window:
            cv2.destroyAllWindows()

    dist = contagem + Counter(travado.values())
    print('\n=== veredito final ===')
    for c in NAMES:
        print(f'  {PT_LABEL[c]:14s} {dist.get(c, 0)}')
    total = sum(dist.values())
    if total:
        print(f'  Premium: {dist.get("intact", 0)}/{total} '
              f'({100 * dist.get("intact", 0) / total:.0f}%)')
    if args.laudo:
        laudo = escrever_laudo(args.laudo, dist, t_inicio, fps, args, 'final')
        print(f'\nlaudo: {args.laudo}')
        print(f'  {laudo["graos"]} grãos, {laudo["premium_pct"]:.1f}% premium, '
              f'~{laudo["massa_estimada_kg"]:.3f} kg, {laudo["vazao_kg_h"]:.1f} kg/h')


if __name__ == '__main__':
    main()
