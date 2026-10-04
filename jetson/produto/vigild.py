#!/usr/bin/env python3
"""Vígil.ia — serviço do aparelho. Sobe a inspeção e a página do operador.

    python3 produto/vigild.py                          # usa produto/aparelho.json
    python3 produto/vigild.py --config outro.json --porta 8081

Depois, no celular ou tablet na mesma rede:  http://<ip-do-jetson>:8080

Em produção roda como serviço do sistema (sobe no boot, reinicia se cair):
    sudo ./produto/instalar_servico.sh
"""
import argparse
import os
import signal
import socket
import sys
import threading

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))

from produto import aparelho as ap      # noqa: E402
from produto.servico import Servico     # noqa: E402
from produto.web import criar_servidor  # noqa: E402


def ips_locais():
    """Endereços para digitar no celular (o que o operador precisa ver)."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))     # não envia nada: só escolhe a interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips) or ['127.0.0.1']


def main(argv=None):
    p = argparse.ArgumentParser(description='Serviço do aparelho Vígil.ia')
    p.add_argument('--config', default=ap.CONFIG_PADRAO)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--porta', type=int, default=None)
    a = p.parse_args(argv)

    cfg = ap.carregar(a.config)
    porta = a.porta or cfg.get('porta', 8080)
    servico = Servico(cfg)
    srv = criar_servidor(servico, a.host, porta)
    servico.iniciar()

    def desligar(sig, _frame):
        # systemd manda SIGTERM: o lote aberto é encerrado com laudo antes de sair
        print(f'\nsinal {sig}: encerrando o lote aberto e fechando…', flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, desligar)
    signal.signal(signal.SIGINT, desligar)
    print('=' * 60)
    print(f' {servico.nome} — no ar')
    for ip in ips_locais():
        print(f'   abra no celular:  http://{ip}:{porta}')
    print(f'   aparelho {servico.aparelho_id} | modelo {servico.modelo}')
    print('=' * 60, flush=True)
    try:
        srv.serve_forever()
    finally:
        servico.parar()
        srv.server_close()
        print('parado.', flush=True)


if __name__ == '__main__':
    main()
