"""Configuração e identidade do aparelho.

O operador não vê flag nenhuma: o técnico deixa a configuração do rig em
`aparelho.json` uma vez, na instalação, e o serviço a traduz para os mesmos
argumentos da linha de comando do `vigil_jetson.py` — uma configuração só,
testada pelos dois caminhos.
"""
import hashlib
import json
import os
import socket

AQUI = os.path.dirname(os.path.abspath(__file__))
CONFIG_PADRAO = os.path.join(AQUI, 'aparelho.json')

# chave do aparelho.json -> flag do vigil_jetson.py
_FLAGS = {
    'engine': '--engine', 'camera': '--camera', 'source': '--source',
    'conf': '--conf', 'roi': '--roi', 'tiles': '--tiles', 'rotate': '--rotate',
    'class_offset': '--class-offset', 'dataset_dir': '--dataset-dir',
    'dataset_passo': '--dataset-passo', 'dataset_graos': '--dataset-graos',
    'dataset_min_livre': '--dataset-min-livre',
}
_BOOLS = {'esteira': '--esteira', 'parado': '--parado', 'quadrado': '--quadrado',
          'csi_sem_trava': '--csi-sem-trava', 'sem_dataset': '--sem-dataset',
          'dataset': '--dataset', 'tempos': '--tempos'}


def carregar(caminho=CONFIG_PADRAO):
    with open(caminho) as f:
        cfg = json.load(f)
    desconhecidas = set(cfg) - set(_FLAGS) - set(_BOOLS) - {'nome', 'porta', 'banco', '_comentario'}
    if desconhecidas:
        raise ValueError(f'chaves desconhecidas em {caminho}: {sorted(desconhecidas)}')
    # caminhos relativos valem a partir da pasta do aparelho.json
    base = os.path.dirname(os.path.abspath(caminho))
    for k in ('engine', 'source', 'dataset_dir', 'banco'):
        if cfg.get(k) and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.normpath(os.path.join(base, cfg[k]))
    return cfg


def argv_de(cfg):
    """aparelho.json -> argv do vigil_jetson (sempre sem janela)."""
    argv = ['--no-window']
    for k, flag in _FLAGS.items():
        if cfg.get(k) not in (None, ''):
            argv += [flag, str(cfg[k])]
    for k, flag in _BOOLS.items():
        if cfg.get(k):
            argv.append(flag)
    return argv


def id_do_aparelho():
    """Identidade estável do aparelho: o número de série gravado no Jetson.

    É o que o laudo carrega e, na fase de proteção, a âncora da licença.
    Fora do Jetson (PC de teste) cai para o machine-id e, por último, o host.
    """
    for p in ('/proc/device-tree/serial-number', '/etc/machine-id'):
        try:
            with open(p, 'rb') as f:
                v = f.read().strip(b'\x00\n ').decode(errors='ignore')
            if v:
                return ('jetson-' if 'device-tree' in p else 'host-') + v[:32]
        except OSError:
            continue
    return 'host-' + socket.gethostname()


def versao_modelo(engine):
    """Impressão digital do arquivo do modelo: o laudo diz qual modelo o deu."""
    try:
        h = hashlib.sha256()
        with open(engine, 'rb') as f:
            for bloco in iter(lambda: f.read(1 << 20), b''):
                h.update(bloco)
        return f'{os.path.basename(engine)}@{h.hexdigest()[:12]}'
    except OSError:
        return os.path.basename(str(engine))
