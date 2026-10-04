"""API e interface locais do aparelho — só biblioteca padrão.

O Jetson serve a página; o operador abre no celular ou tablet pela rede
local. Sem FastAPI, sem Node: `http.server` com uma thread por cliente dá
conta de vídeo ao vivo e estado em tempo real para os poucos aparelhos de uma
rede de armazém, e não acrescenta nada para instalar (no Jetson o pip já
brigou com o "externally-managed-environment").

    GET  /                       interface
    GET  /laudo/<id>             laudo imprimível (salvar como PDF no navegador)
    GET  /ao-vivo.mjpg           vídeo anotado (MJPEG — abre em qualquer navegador)
    GET  /api/estado             estado atual (JSON)
    GET  /api/eventos            estado em tempo real (Server-Sent Events)
    GET  /api/lotes              lotes recentes
    POST /api/lotes              cria lote {codigo, produtor, variedade, amostra_kg, obs}
    POST /api/lotes/<id>/iniciar
    POST /api/lotes/<id>/encerrar   -> laudo
    GET  /api/laudos[?pendentes=1]
    GET  /api/laudos/<id>
    GET  /api/registro           eventos do aparelho (câmera, erros, lotes)
"""
import json
import mimetypes
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui')
FPS_VIDEO = 10           # o celular não precisa dos 14 quadros/s; poupa Wi-Fi
ESTADO_HZ = 2


class _Handler(BaseHTTPRequestHandler):
    servico = None                       # injetado por criar_servidor()
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *a):      # sem uma linha por requisição no journal
        pass

    # ------------------------------------------------------------ respostas
    def _json(self, dados, status=200):
        corpo = json.dumps(dados, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(corpo)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(corpo)

    def _erro(self, status, msg):
        self._json({'erro': msg}, status)

    def _arquivo(self, nome):
        caminho = os.path.realpath(os.path.join(UI, nome))
        if not caminho.startswith(os.path.realpath(UI) + os.sep) or not os.path.isfile(caminho):
            return self._erro(404, 'não encontrado')
        with open(caminho, 'rb') as f:
            corpo = f.read()
        tipo = mimetypes.guess_type(caminho)[0] or 'application/octet-stream'
        if tipo.startswith('text/') or tipo.endswith('javascript'):
            tipo += '; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', tipo)
        self.send_header('Content-Length', str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _corpo(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > 64 * 1024:
            raise ValueError('corpo grande demais')
        if not n:
            return {}
        dados = json.loads(self.rfile.read(n) or b'{}')
        if not isinstance(dados, dict):
            raise ValueError('esperava um objeto JSON')
        return dados

    # ------------------------------------------------------------ fluxos
    def _mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=quadro')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            while not self.servico._parar.is_set():
                jpg = self.servico.jpeg()
                if jpg:
                    self.wfile.write(b'--quadro\r\nContent-Type: image/jpeg\r\n'
                                     b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n'
                                     + jpg + b'\r\n')
                    self.wfile.flush()
                time.sleep(1 / FPS_VIDEO)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                          # celular fechou a aba

    def _sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        try:
            while not self.servico._parar.is_set():
                dados = json.dumps(self.servico.estado(), ensure_ascii=False, default=str)
                self.wfile.write(f'data: {dados}\n\n'.encode())
                self.wfile.flush()
                time.sleep(1 / ESTADO_HZ)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ------------------------------------------------------------ rotas
    def do_GET(self):
        try:
            caminho, _, query = self.path.partition('?')
            if caminho in ('/', '/index.html'):
                return self._arquivo('index.html')
            if re.fullmatch(r'/laudo/[\w-]+', caminho):
                return self._arquivo('laudo.html')
            if caminho.startswith('/ui/'):
                return self._arquivo(caminho[4:])
            if caminho == '/ao-vivo.mjpg':
                return self._mjpeg()
            if caminho == '/api/estado':
                return self._json(self.servico.estado())
            if caminho == '/api/eventos':
                return self._sse()
            if caminho == '/api/lotes':
                return self._json(self.servico.banco.lotes())
            if caminho == '/api/laudos':
                return self._json(self.servico.banco.laudos(pendentes='pendentes=1' in query))
            m = re.fullmatch(r'/api/laudos/([\w-]+)', caminho)
            if m:
                l = self.servico.banco.laudo(m.group(1))
                return self._json(l) if l else self._erro(404, 'laudo não encontrado')
            if caminho == '/api/registro':
                return self._json(self.servico.banco.eventos())
            return self._erro(404, 'não encontrado')
        except Exception as e:            # erro da API morre na API
            self._erro_interno(e)

    def do_POST(self):
        try:
            caminho = self.path.partition('?')[0]
            if caminho == '/api/lotes':
                d = self._corpo()
                campos = {k: d.get(k) for k in ('codigo', 'produtor', 'variedade',
                                                 'amostra_kg', 'obs')}
                return self._json(self.servico.criar_lote(**campos), 201)
            m = re.fullmatch(r'/api/lotes/([\w-]+)/(iniciar|encerrar)', caminho)
            if m:
                lid, acao = m.groups()
                if acao == 'iniciar':
                    return self._json(self.servico.iniciar_lote(lid))
                return self._json(self.servico.encerrar_lote(lid))
            return self._erro(404, 'não encontrado')
        except Exception as e:
            self._erro_interno(e)

    def _erro_interno(self, e):
        from produto.servico import LoteInvalido
        if isinstance(e, KeyError):
            return self._erro(404, 'lote não encontrado')
        if isinstance(e, LoteInvalido):
            return self._erro(409, str(e))
        if isinstance(e, (ValueError, json.JSONDecodeError)):
            return self._erro(400, str(e))
        try:
            self.servico.banco.evento('erro', f'api {self.command} {self.path}: {e!r}')
            self._erro(500, 'erro interno — veja /api/registro')
        except Exception:
            pass


def criar_servidor(servico, host='0.0.0.0', porta=8080):
    handler = type('Handler', (_Handler,), {'servico': servico})
    srv = ThreadingHTTPServer((host, porta), handler)
    srv.daemon_threads = True             # cliente pendurado não segura o desligamento
    return srv


def servir_em_thread(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True, name='web')
    t.start()
    return t
