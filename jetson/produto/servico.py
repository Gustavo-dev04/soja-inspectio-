"""O serviço do aparelho: a inspeção rodando sem parar, e os lotes por cima.

Uma thread roda `Inspecao.passo()` em laço; a API (outra thread) só pede
coisas — criar, iniciar e encerrar lote — e lê o estado. Princípio que manda
em tudo aqui: **nada derruba a inspeção**. Câmera que cai é reaberta; erro numa
varredura é registrado e o laço segue; erro da API morre na API.
"""
import os
import sys
import threading
import time

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))          # jetson/

from produto import aparelho as ap                 # noqa: E402
from produto.banco import Banco                    # noqa: E402

PARCIAL_S = 30.0        # de quanto em quanto tempo o lote em andamento é salvo


class LoteInvalido(Exception):
    """Pedido que não faz sentido agora (409 na API)."""


class Servico:
    def __init__(self, cfg, vj=None, log=print):
        self.cfg, self.log = cfg, log
        if vj is None:
            import vigil_jetson as vj
        self.vj = vj
        self.banco = Banco(cfg.get('banco') or os.path.join(AQUI, 'vigil.db'))
        self.aparelho_id = ap.id_do_aparelho()
        self.modelo = ap.versao_modelo(cfg.get('engine', ''))
        self.nome = cfg.get('nome', 'Vígil.ia')
        self._recuperar_interrompidos()

        args = vj.criar_parser().parse_args(ap.argv_de(cfg))
        self.insp = vj.Inspecao(args, log=log)
        self.lote_id = None
        self.camera_ok = True
        self.varreduras = 0
        self._parar = threading.Event()
        self._trava_lote = threading.Lock()        # dois toques em "iniciar" = um lote
        self._prox_parcial = 0.0
        self._jpeg = (None, None)                  # (quadro, bytes) — cache
        self._trava_jpeg = threading.Lock()
        self._thread = threading.Thread(target=self._laco, daemon=True, name='inspecao')
        self.banco.evento('inicio', f'serviço iniciado — {self.modelo} em {self.aparelho_id}')

    # ------------------------------------------------------------ recuperação
    def _recuperar_interrompidos(self):
        """Aparelho desligou no meio de um lote: ele volta como 'interrompido',
        com o laudo montado a partir da última fotografia parcial."""
        ativo = self.banco.lote_ativo()
        while ativo:
            parcial = self.banco.lote(ativo['id']).get('parcial')
            if parcial:
                parcial = dict(parcial, obs='interrompido — números da última '
                                            'fotografia parcial')
                self.banco.salvar_laudo(ativo['id'], self.aparelho_id, self.modelo, parcial)
            self.banco.marcar_lote(ativo['id'], 'interrompido')
            self.banco.evento('recuperacao', f'lote {ativo["codigo"]} estava aberto '
                                             'quando o serviço parou — interrompido')
            ativo = self.banco.lote_ativo()

    # ------------------------------------------------------------ laço
    def iniciar(self):
        self._thread.start()

    def _laco(self):
        while not self._parar.is_set():
            try:
                ok = self.insp.passo()
            except Exception as e:                  # varredura ruim não para nada
                self.banco.evento('erro', f'varredura: {e!r}')
                time.sleep(0.2)
                continue
            if not ok:
                if self.camera_ok:
                    self.camera_ok = False
                    self.banco.evento('camera', 'sem quadro — tentando reabrir')
                    self.log('câmera sem quadro — tentando reabrir')
                if not self._parar.wait(1.0) and self.insp.reabrir():
                    self.camera_ok = True
                    self.banco.evento('camera', 'câmera de volta')
                    self.log('câmera de volta')
                continue
            self.varreduras += 1
            if self.lote_id and time.time() >= self._prox_parcial:
                self._prox_parcial = time.time() + PARCIAL_S
                try:
                    self.banco.salvar_parcial(self.lote_id, self.insp.laudo('parcial'))
                except Exception as e:
                    self.banco.evento('erro', f'parcial: {e!r}')

    # ------------------------------------------------------------ lotes
    def criar_lote(self, **campos):
        return self.banco.criar_lote(**campos)

    def iniciar_lote(self, lid):
        with self._trava_lote:
            return self._iniciar_lote(lid)

    def _iniciar_lote(self, lid):
        lote = self.banco.lote(lid)
        if lote is None:
            raise KeyError(lid)
        if self.lote_id:
            raise LoteInvalido('já existe um lote em inspeção — encerre antes')
        if lote['estado'] != 'criado':
            raise LoteInvalido(f'lote {lote["codigo"]} já foi {lote["estado"]}')
        self.insp.iniciar_lote(lote['codigo'])
        self.lote_id = lid
        self._prox_parcial = time.time() + PARCIAL_S
        self.banco.marcar_lote(lid, 'inspecionando')
        self.banco.evento('lote', f'{lote["codigo"]} iniciado')
        return self.banco.lote(lid)

    def encerrar_lote(self, lid=None):
        with self._trava_lote:
            return self._encerrar_lote(lid)

    def _encerrar_lote(self, lid=None):
        lid = lid or self.lote_id
        if not lid or lid != self.lote_id:
            raise LoteInvalido('esse lote não está em inspeção')
        dados = self.insp.encerrar_lote()
        lote = self.banco.lote(lid)
        dados['lote_codigo'] = lote['codigo']
        if lote.get('amostra_kg'):
            dados['amostra_declarada_kg'] = lote['amostra_kg']
        self.lote_id = None
        laudo = self.banco.salvar_laudo(lid, self.aparelho_id, self.modelo, dados)
        self.banco.marcar_lote(lid, 'encerrado')
        self.banco.evento('lote', f'{lote["codigo"]} encerrado — {dados["graos"]} grãos, '
                                  f'{dados["premium_pct"]}% premium')
        return laudo

    # ------------------------------------------------------------ leitura
    def estado(self):
        e = self.insp.estado()
        e.update(aparelho=self.nome, aparelho_id=self.aparelho_id, modelo=self.modelo,
                 camera_ok=self.camera_ok, lote_id=self.lote_id,
                 lote_info=self.banco.lote(self.lote_id) if self.lote_id else None)
        return e

    def jpeg(self):
        """Último quadro em JPEG, codificado uma vez por quadro mesmo com
        vários celulares assistindo."""
        q = self.insp.ultimo_quadro()
        with self._trava_jpeg:
            if self._jpeg[0] is q and self._jpeg[1] is not None:
                return self._jpeg[1]
            b = self.insp.ultimo_quadro_jpeg()
            self._jpeg = (q, b)
            return b

    # ------------------------------------------------------------ fim
    def parar(self, timeout=10.0):
        """SIGTERM/desligamento: encerra o lote aberto com laudo, fecha tudo."""
        self._parar.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self.lote_id:
            try:
                l = self.encerrar_lote()
                self.log(f'lote encerrado no desligamento — laudo {l["id"]}')
            except Exception as e:
                self.banco.evento('erro', f'encerrar no desligamento: {e!r}')
        self.insp.fechar()
        self.banco.evento('fim', 'serviço parado')
        self.banco.fechar()
