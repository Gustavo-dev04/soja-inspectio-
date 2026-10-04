#!/usr/bin/env python3
"""Vígil.ia — grava o dataset do rig ENQUANTO inspeciona.

Usado pelo `vigil_jetson.py`: inspecionar no padrão do rig já é coletar. Não
existe mais um passo separado de captura — o app cria as pastas sozinho e
separa cada grão pela classe que deu a ele.

Layout, criado na primeira execução:

    dataset/
      quadros/<sessao>/        o que o MODELO viu, do jeito que ele viu
        *.jpg                  uma imagem por janela de entrada (704x704 no rig)
        quadros.jsonl          uma linha por imagem: caixas + grão de cada caixa
        graos.jsonl            uma linha por grão: classe proposta e confiança
        sessao.json            configuração e domínio (travas, câmera, engine)
      revisar/<sessao>/        UM recorte por grão, já na pasta da classe:
        intact/ immature/ broken/ skin-damaged/ spotted/
        descartar/             não é grão (sujeira, reflexo, caixa dupla)
        duvida/                é grão, mas a classe não dá pra saber
      pronto/                  sai de `coletar_dataset.py --exportar`

Decisões que importam (e quebrariam em silêncio se fossem outras):

* **Imagem = janela de entrada do modelo, não o quadro inteiro.** No rig com
  `--tiles 2` o quadro tem ~1267x704, e o modelo vê dois recortes 704x704. Um
  detector treinado com o quadro inteiro reduzido para 704 veria o grão com
  ~50 px em vez dos ~89 px da inferência — escala errada, em silêncio.
* **Todo grão que aparece numa imagem salva ganha recorte para revisão.** Grão
  sem rótulo dentro de uma imagem ensina o detector que aquilo é FUNDO. Por isso
  o recorte não é opcional, e o exportador descarta a imagem inteira se algum
  grão dela ficar sem rótulo.
* **Nunca segura a inspeção.** JPEG e disco ficam numa thread; se a fila encher,
  a gravação perde a imagem e conta, mas o laço não espera. A inspeção é o
  produto; o dataset é subproduto.
* **Memória limitada aos grãos vivos.** Numa jornada de 14-16 h passam milhões de
  grãos; cada um é liberado quando o rastreamento o aposenta.
* **Para sozinho**, sem parar a inspeção, ao bater a meta de grãos da sessão ou
  quando o disco livre cai abaixo do mínimo.

Sem dependência de TensorRT: roda e testa no PC.
"""
import json
import os
import queue
import re
import shutil
import threading
import time

import cv2
import numpy as np

# Mesma ordem do vigil_jetson.NAMES — o vigil confere na criação.
CLASSES = ['broken', 'immature', 'intact', 'skin-damaged', 'spotted']
DESCARTE = 'descartar'     # não é grão: a caixa sai, a imagem fica
DUVIDA = 'duvida'          # é grão sem classe: a imagem inteira sai
PASTAS = CLASSES + [DESCARTE, DUVIDA]

_AQUI = os.path.dirname(os.path.abspath(__file__))
# VIGIL_DATASET aponta para outro disco. No Orin Nano vale usar o SSD NVMe do
# slot M.2 se houver: gravação contínua desgasta cartão SD, e ele enche rápido.
RAIZ_PADRAO = os.environ.get('VIGIL_DATASET') or os.path.join(_AQUI, 'dataset')

NITIDEZ_MIN = 50.0     # variância do laplaciano (canal verde) — abaixo é borrão
MARGEM_BORDA = 3       # px: caixa a menos disso da borda da janela = grão cortado
CONTEXTO = 0.15        # borda extra no recorte de revisão
JPEG_QUADRO, JPEG_RECORTE = 92, 95
FORMATO = 2            # versão do layout (1 = coletar_dataset antigo)
RELOGIO = time.monotonic   # trocável nos testes, que rodam mais rápido que o real


def caminhos(raiz):
    return {'raiz': raiz,
            'quadros': os.path.join(raiz, 'quadros'),
            'revisar': os.path.join(raiz, 'revisar'),
            'pronto': os.path.join(raiz, 'pronto')}


def recortar(frame, caixa, contexto=CONTEXTO):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    pad = int(contexto * max(x2 - x1, y2 - y1))
    return frame[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)].copy()


def nitidez_de(img):
    """Variância do laplaciano no canal verde.

    O verde é o canal de maior contraste no grão de soja e evita o cvtColor —
    medido: 46 µs contra 106 µs por grão no caminho em cinza/float64.
    """
    if img.size == 0:
        return 0.0
    canal = img[:, :, 1] if img.ndim == 3 else img
    return float(cv2.Laplacian(canal, cv2.CV_32F).var())


def _limpo(texto):
    """Lote vai no nome de arquivo: só letra, número e hífen."""
    return re.sub(r'[^A-Za-z0-9-]+', '-', str(texto)).strip('-') or 'L000'


def _dentro(caixa, janela, margem=MARGEM_BORDA):
    x1, y1, x2, y2 = caixa
    jx, jy, jw, jh = janela
    return (x1 >= jx + margem and y1 >= jy + margem
            and x2 <= jx + jw - margem and y2 <= jy + jh - margem)


class GravadorDataset:
    """Recebe cada varredura do laço de inspeção e decide o que vira dado.

    Contrato com quem chama (o vigil_jetson.py):

    * `observar(frame, rastreados, janelas)` a cada varredura, com o quadro
      **limpo** — antes de desenhar caixa e texto por cima;
    * `aposentar(tid, classe)` quando o rastreamento aposenta um grão, com o
      veredito final dele;
    * `nova_epoca()` se o rastreador for recriado (os IDs recomeçam do 1);
    * `fechar()` no fim, depois de aposentar os grãos ainda vivos.
    """

    def __init__(self, raiz=RAIZ_PADRAO, lote='L001', passo=6, max_graos=2000,
                 min_livre_gb=2.0, bloco_s=20.0, guarda_s=3.0, travas=True,
                 meta=None, classes=CLASSES, fila=24, fila_recortes=512,
                 relogio=None):
        self.classes = list(classes)
        self.passo, self.max_graos = max(1, int(passo)), int(max_graos)
        self.min_livre = float(min_livre_gb) * 1e9
        self.bloco_s, self.guarda_s = float(bloco_s), float(guarda_s)
        self.travas = bool(travas)
        self.relogio = relogio or RELOGIO
        self.lote = _limpo(lote)

        cam = caminhos(raiz)
        sufixo = '' if self.travas else '_sem-trava'
        base = time.strftime('%Y%m%d-%H%M%S') + sufixo
        self.sessao, n = base, 1
        while (os.path.exists(os.path.join(cam['quadros'], self.sessao))
               or os.path.exists(os.path.join(cam['revisar'], self.sessao))):
            n += 1
            self.sessao = f'{base}-{n}'
        self.raiz = raiz
        self.dir_q = os.path.join(cam['quadros'], self.sessao)
        self.dir_r = os.path.join(cam['revisar'], self.sessao)
        os.makedirs(self.dir_q, exist_ok=True)
        for p in PASTAS:
            os.makedirs(os.path.join(self.dir_r, p), exist_ok=True)
        self.prefixo = f'{self.lote}_{self.sessao}'

        # estado por grão VIVO — tudo é liberado em aposentar()
        self._chave = {}         # tid -> chave do grão
        self._melhor = {}        # chave -> (distância ao centro, recorte)
        self._precisa = set()    # chaves que estão em pelo menos 1 imagem salva
        self._seq = 0

        self.n_varr = 0
        self.t0 = self.relogio()
        self.gravando = True
        self.motivo = None       # por que parou: 'meta' | 'disco' | 'erro'
        self._prox_disco = 0.0
        self.graos = 0           # grãos comprometidos (entraram em imagem salva)
        self.quadros = self.caixas = self.recortes = 0
        self.perdidos_quadros = self.perdidos_recortes = 0
        self.borrados = self.erros = 0
        self.bytes = 0
        self._erros_seguidos = 0
        self.falha = None        # exceção inesperada: a gravação desliga, o app segue

        self.meta = {'formato': FORMATO, 'sessao': self.sessao, 'lote': self.lote,
                     'quando': time.strftime('%Y-%m-%d %H:%M:%S'),
                     'travas': self.travas, 'passo': self.passo,
                     'bloco_s': self.bloco_s, 'guarda_s': self.guarda_s,
                     'max_graos': self.max_graos, 'classes': self.classes}
        self.meta.update(meta or {})
        self._salvar_meta()       # já na partida: se o app cair, a sessão vale

        self._jq = open(os.path.join(self.dir_q, 'quadros.jsonl'), 'a', buffering=1)
        self._jg = open(os.path.join(self.dir_q, 'graos.jsonl'), 'a', buffering=1)
        # Uma fila só, com PRIORIDADE para recorte e limite por tipo. Perder
        # uma imagem custa uma imagem; perder um recorte deixa grão sem rótulo
        # e invalida todas as imagens dele. E recorte chega em rajada: grãos da
        # mesma fileira saem do quadro na mesma varredura.
        self._fila = queue.PriorityQueue()
        self._lim = {'quadro': max(2, int(fila)), 'recorte': max(8, int(fila_recortes))}
        self._pend = {'quadro': 0, 'recorte': 0}
        self._trava = threading.Lock()
        self._ordem = 0
        self._thread = threading.Thread(target=self._escritor, daemon=True,
                                        name='gravador-dataset')
        self._thread.start()

    # ------------------------------------------------------------ escrita
    def _escritor(self):
        while True:
            _, _, item = self._fila.get()
            if item is None:
                return
            tipo, caminho, img, reg = item
            try:
                q = JPEG_QUADRO if tipo == 'quadro' else JPEG_RECORTE
                if not cv2.imwrite(caminho, img, [cv2.IMWRITE_JPEG_QUALITY, q]):
                    raise OSError(f'imwrite falhou: {caminho}')
                self.bytes += os.path.getsize(caminho)
                # a linha só entra depois que a imagem existe: o .jsonl nunca
                # aponta para arquivo que não foi escrito
                (self._jq if tipo == 'quadro' else self._jg).write(
                    json.dumps(reg, ensure_ascii=False) + '\n')
                if tipo == 'quadro':
                    self.quadros += 1
                    self.caixas += len(reg['caixas'])
                else:
                    self.recortes += 1
                self._erros_seguidos = 0
            except Exception as e:                     # disco cheio, cartão fora…
                self.erros += 1
                self._erros_seguidos += 1
                self.ultimo_erro = str(e)
                if self._erros_seguidos >= 5:
                    self._parar('erro')
            finally:
                with self._trava:
                    self._pend[tipo] -= 1

    def _enfileirar(self, item):
        """Nunca bloqueia: sem vaga, o item é perdido e contado."""
        tipo = item[0]
        with self._trava:
            if self._pend[tipo] >= self._lim[tipo]:
                if tipo == 'quadro':
                    self.perdidos_quadros += 1
                else:
                    self.perdidos_recortes += 1
                return False
            self._pend[tipo] += 1
            self._ordem += 1
            ordem = self._ordem
        # (prioridade, ordem, item): a ordem desempata e impede que o
        # PriorityQueue tente comparar arrays numpy
        self._fila.put((0 if tipo == 'recorte' else 1, ordem, item))
        return True

    def pendentes(self):
        with self._trava:
            return self._pend['quadro'] + self._pend['recorte']

    def _parar(self, motivo):
        if self.gravando:
            self.gravando, self.motivo = False, motivo

    def _checar_disco(self, agora):
        if agora < self._prox_disco:
            return
        self._prox_disco = agora + 5.0
        try:
            if shutil.disk_usage(self.raiz).free < self.min_livre:
                self._parar('disco')
        except OSError:
            self._parar('disco')

    # ------------------------------------------------------------ grãos
    def _chave_de(self, tid):
        c = self._chave.get(tid)
        if c is None:
            self._seq += 1
            c = self._chave[tid] = f'{self.prefixo}_g{self._seq:06d}'
        return c

    def _falhar(self, e):
        """Bug ou caso não previsto: desliga a gravação, nunca a inspeção.

        Numa jornada de 14-16 h, uma exceção aqui derrubaria o app e levaria
        junto o laudo — que é o produto. O dataset é subproduto.
        """
        if self.falha is None:
            self.falha = repr(e)
            print(f'\ndataset: falha inesperada ({self.falha}) — gravação '
                  'DESLIGADA, a inspeção continua')
        self._parar('erro')
        self._chave.clear()
        self._melhor.clear()
        self._precisa.clear()

    def observar(self, frame, rastreados, janelas):
        """Uma varredura: atualiza a melhor vista de cada grão e, a cada
        `passo`, grava as janelas com as caixas.

        `rastreados` = [(tid, x1, y1, x2, y2, classe_idx, conf), …] em
        coordenadas do quadro; `janelas` = [(x, y, largura, altura), …] — os
        recortes que o modelo come.
        """
        if self.falha is not None:
            return
        try:
            self._observar(frame, rastreados, janelas)
        except Exception as e:
            self._falhar(e)

    def _observar(self, frame, rastreados, janelas):
        self.n_varr += 1
        if not self.gravando and not self._precisa:
            return                   # meta batida e nada pendente: custo zero

        # 1) melhor vista por grão: a mais CENTRADA numa janela. No rig a
        #    esteira tem velocidade constante, então o borrão é igual em todo
        #    ponto; o centro tem menos distorção da 120° e a luz mais uniforme
        #    do ring light. É um critério geométrico — não custa processamento.
        dentro = []                  # (chave, caixa, ci, cf, [janelas])
        for tid, x1, y1, x2, y2, ci, cf in rastreados:
            if not self.gravando and self._chave.get(tid) not in self._precisa:
                continue
            caixa = (int(x1), int(y1), int(x2), int(y2))
            js = [k for k, j in enumerate(janelas) if _dentro(caixa, j)]
            if not js:
                continue             # cortado na borda em todas as janelas
            chave = self._chave_de(tid)
            cx, cy = (caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2
            d = min(((cx - (janelas[k][0] + janelas[k][2] / 2)) ** 2
                     + (cy - (janelas[k][1] + janelas[k][3] / 2)) ** 2) ** 0.5
                    / janelas[k][2] for k in js)
            ant = self._melhor.get(chave)
            if ant is None or d < ant[0] - 1e-3:
                self._melhor[chave] = (d, recortar(frame, caixa))
            dentro.append((chave, caixa, ci, cf, js))

        # 2) gravar esta varredura?
        if not self.gravando or self.n_varr % self.passo:
            return
        agora = self.relogio() - self.t0
        self._checar_disco(agora)
        if self.max_graos and self.graos >= self.max_graos:
            self._parar('meta')
        if not self.gravando or not dentro:
            return

        # um quadro borrado ensinaria borrão: mediana de até 8 grãos
        amostra = dentro[::max(1, len(dentro) // 8)][:8]
        nits = [nitidez_de(frame[c[1]:c[3], c[0]:c[2]]) for _, c, *_ in amostra]
        if float(np.median(nits)) < NITIDEZ_MIN:
            self.borrados += 1
            return

        bloco = int(agora // self.bloco_s)
        for k, (jx, jy, jw, jh) in enumerate(janelas):
            caixas = [{'grao': chave,
                       'caixa': [x1 - jx, y1 - jy, x2 - jx, y2 - jy],
                       'prev': (self.classes[ci] if 0 <= ci < len(self.classes)
                                else None),
                       'conf': round(float(cf), 3)}
                      for chave, (x1, y1, x2, y2), ci, cf, js in dentro if k in js]
            if not caixas:
                continue
            nome = f'{self.prefixo}_b{bloco:04d}_{self.n_varr:07d}_j{k}.jpg'
            reg = {'arquivo': nome, 'bloco': bloco, 't': round(agora, 2),
                   'varredura': self.n_varr, 'janela': k,
                   'largura': int(jw), 'altura': int(jh), 'caixas': caixas}
            # cópia: logo depois o laço desenha caixa e HUD em cima do quadro
            img = frame[jy:jy + jh, jx:jx + jw].copy()
            if self._enfileirar(('quadro', os.path.join(self.dir_q, nome), img, reg)):
                novos = {c['grao'] for c in caixas} - self._precisa
                self._precisa |= novos
                self.graos += len(novos)

    def aposentar(self, tid, classe, confianca=None, n_obs=None):
        """O grão saiu: grava o recorte dele, se ele estiver em imagem salva."""
        if self.falha is not None:
            return
        try:
            self._aposentar(tid, classe, confianca, n_obs)
        except Exception as e:
            self._falhar(e)

    def _aposentar(self, tid, classe, confianca, n_obs):
        chave = self._chave.pop(tid, None)
        if chave is None:
            return
        melhor = self._melhor.pop(chave, None)
        if chave not in self._precisa:
            return
        self._precisa.discard(chave)
        pasta = classe if classe in self.classes else DUVIDA
        reg = {'grao': chave, 'classe_prevista': pasta,
               'confianca': None if confianca is None else round(float(confianca), 3),
               'n_varreduras': n_obs}
        if melhor is None:                 # não deveria: grão salvo tem vista
            self.erros += 1
            return
        self._enfileirar(('recorte', os.path.join(self.dir_r, pasta,
                                                  f'{chave}__{pasta}.jpg'),
                          melhor[1], reg))

    def nova_epoca(self):
        """Rastreador recriado: IDs recomeçam, então o mapa tid->grão zera.

        Chame DEPOIS de aposentar os vivos. As chaves já usadas continuam
        únicas porque a sequência não volta.
        """
        self._chave.clear()

    # ------------------------------------------------------------ status
    def hud(self):
        if self.gravando:
            alvo = f'/{self.max_graos}' if self.max_graos else ''
            return f'REC {self.graos}{alvo}'
        return f'REC parado ({self.motivo})'

    def cabecalho(self):
        try:
            livre = shutil.disk_usage(self.raiz).free / 1e9
        except OSError:
            livre = float('nan')
        meta_txt = f'até {self.max_graos} grãos' if self.max_graos else 'sem limite de grãos'
        linhas = [f'dataset : GRAVANDO -> {self.dir_r}',
                  f'          sessão {self.sessao} | lote {self.lote} | {meta_txt} | '
                  f'1 varredura a cada {self.passo}',
                  f'          {livre:.1f} GB livres (para sozinho abaixo de '
                  f'{self.min_livre / 1e9:.1f} GB)']
        if not self.travas:
            linhas.append('          AVISO: sem travas de exposição — esta sessão fica FORA')
            linhas.append('          do --exportar por padrão (outro domínio).')
        return '\n'.join(linhas)

    def _salvar_meta(self, resumo=None):
        if resumo is not None:
            self.meta['resumo'] = resumo
        tmp = os.path.join(self.dir_q, 'sessao.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.dir_q, 'sessao.json'))   # atômico

    def fechar(self, timeout=120.0):
        """Esvazia a fila, fecha os arquivos e devolve o resumo da sessão."""
        self._parar('fim')
        self._fila.put((9, float('inf'), None))     # depois de tudo que já entrou
        self._thread.join(timeout=timeout)
        for arq in (self._jq, self._jg):
            try:
                arq.close()
            except OSError:
                pass
        resumo = {'quadros': self.quadros, 'caixas': self.caixas,
                  'recortes': self.recortes, 'graos': self.graos,
                  'varreduras': self.n_varr,
                  'perdidos_fila': self.perdidos_quadros + self.perdidos_recortes,
                  'perdidos_quadros': self.perdidos_quadros,
                  'perdidos_recortes': self.perdidos_recortes,
                  'quadros_borrados': self.borrados,
                  'erros': self.erros, 'mb': round(self.bytes / 1e6, 1),
                  'duracao_s': round(self.relogio() - self.t0, 1),
                  'motivo_parada': self.motivo, 'falha': self.falha}
        if not self.quadros and not self.recortes:
            # sessão vazia (câmera ligada sem grão): não deixa lixo para revisar
            shutil.rmtree(self.dir_q, ignore_errors=True)
            shutil.rmtree(self.dir_r, ignore_errors=True)
            resumo['removida'] = True
        else:
            try:
                self._salvar_meta(resumo)
            except OSError as e:          # disco cheio: o .jsonl já está salvo
                resumo['erro_meta'] = repr(e)
        return resumo
