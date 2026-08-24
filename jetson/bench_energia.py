#!/usr/bin/env python3
"""Vígil.ia — mede CONSUMO e TEMPERATURA do Jetson rodando o modelo.

fps sozinho não fecha um projeto de produto: a máquina vai operar 14-16 h por
dia, e a conta de energia, a fonte e a dissipação dependem de watts medidos —
não do número da caixa.

    python3 bench_energia.py                          # engine padrão, 60 s
    python3 bench_energia.py --engine X.engine --carga 120
    python3 bench_energia.py --comando "python3 vigil_jetson.py --engine X.engine --source v.mp4 --no-window"

O que ele faz, em três fases:

  1. OCIOSO   — linha de base, com a placa parada
  2. CARGA    — o modelo rodando de verdade
  3. RESFRIA  — depois da carga, para ver se a temperatura volta

O consumo do modelo é a DIFERENÇA entre 2 e 1. Reportar só o total confundiria
o custo do modelo com o da placa ligada, que existiria de qualquer jeito.

Fonte dos números: `tegrastats`, o próprio instrumento da NVIDIA, que lê os
monitores INA3221 da placa. VDD_IN é a entrada total.
"""
import argparse
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

# tegrastats no Orin: "VDD_IN 4523mW/4200mW" = instantâneo/média
RE_TRILHO = re.compile(r'(VDD_\w+)\s+(\d+)mW/(\d+)mW')
RE_TEMP = re.compile(r'(\w+)@([\d.]+)C')
RE_GPU = re.compile(r'GR3D_FREQ\s+(\d+)%')
RE_CPU = re.compile(r'CPU \[([^\]]+)\]')
RE_RAM = re.compile(r'RAM (\d+)/(\d+)MB')


def parse_tegrastats(linha):
    """Extrai potência, temperatura e uso de uma linha do tegrastats."""
    d = {'trilhos': {}, 'temps': {}}
    for nome, inst, _media in RE_TRILHO.findall(linha):
        d['trilhos'][nome] = int(inst)
    for nome, val in RE_TEMP.findall(linha):
        d['temps'][nome] = float(val)
    m = RE_GPU.search(linha)
    if m:
        d['gpu'] = int(m.group(1))
    m = RE_CPU.search(linha)
    if m:
        usos = [int(p.split('%')[0]) for p in m.group(1).split(',')
                if '%' in p and p.split('%')[0].isdigit()]
        if usos:
            d['cpu'] = sum(usos) / len(usos)
    m = RE_RAM.search(linha)
    if m:
        d['ram_mb'], d['ram_total_mb'] = int(m.group(1)), int(m.group(2))
    return d if d['trilhos'] else None


class Monitor:
    """Roda o tegrastats em segundo plano e acumula as amostras."""

    def __init__(self, intervalo_ms=500):
        cmd = ['tegrastats', '--interval', str(intervalo_ms)]
        if os.geteuid() != 0 and shutil.which('sudo'):
            cmd = ['sudo', '-n'] + cmd            # -n: não trava pedindo senha
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True,
                                     bufsize=1)
        self.amostras = []
        import threading
        self._parar = False
        self.t = threading.Thread(target=self._ler, daemon=True)
        self.t.start()

    def _ler(self):
        for linha in self.proc.stdout:
            if self._parar:
                break
            d = parse_tegrastats(linha)
            if d:
                d['t'] = time.time()
                self.amostras.append(d)

    def janela(self, t0, t1):
        return [a for a in self.amostras if t0 <= a['t'] <= t1]

    def fim(self):
        self._parar = True
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def resume(amostras, trilho='VDD_IN'):
    if not amostras:
        return None
    w = [a['trilhos'].get(trilho, 0) / 1000 for a in amostras
         if trilho in a['trilhos']]
    if not w:
        return None
    tj = [a['temps'].get('tj', 0) for a in amostras if 'tj' in a['temps']]
    gpu = [a.get('gpu', 0) for a in amostras if 'gpu' in a]
    cpu = [a.get('cpu', 0) for a in amostras if 'cpu' in a]
    ram = [a.get('ram_mb', 0) for a in amostras if 'ram_mb' in a]
    return {
        'n': len(w),
        'w_med': statistics.mean(w), 'w_max': max(w), 'w_min': min(w),
        'tj_med': statistics.mean(tj) if tj else None,
        'tj_max': max(tj) if tj else None,
        'gpu_med': statistics.mean(gpu) if gpu else None,
        'cpu_med': statistics.mean(cpu) if cpu else None,
        'ram_max': max(ram) if ram else None,
    }


def linha_fase(nome, r):
    if not r:
        return f'  {nome:10s} — sem amostras'
    s = (f'  {nome:10s} {r["w_med"]:6.2f} W  (pico {r["w_max"]:5.2f}, '
         f'mín {r["w_min"]:5.2f})')
    if r['tj_med'] is not None:
        s += f'   tj {r["tj_med"]:.1f}°C (máx {r["tj_max"]:.1f})'
    if r['gpu_med'] is not None:
        s += f'   GPU {r["gpu_med"]:.0f}%'
    return s


def modo_energia():
    try:
        out = subprocess.run(['sudo', '-n', 'nvpmodel', '-q'],
                             capture_output=True, text=True, timeout=10).stdout
        nome = re.search(r'NV Power Mode:\s*(\S+)', out)
        num = re.search(r'^\s*(\d+)\s*$', out, re.M)
        return f'{nome.group(1) if nome else "?"} (modo {num.group(1) if num else "?"})'
    except Exception:
        return 'desconhecido'


def main():
    ap = argparse.ArgumentParser(description='Consumo e temperatura sob carga')
    ap.add_argument('--engine', default=None,
                    help='engine .engine (padrão: a primeira da pasta)')
    ap.add_argument('--comando', default=None,
                    help='comando de carga (padrão: trtexec na engine)')
    ap.add_argument('--ocioso', type=float, default=20, help='segundos de linha de base')
    ap.add_argument('--carga', type=float, default=60, help='segundos sob carga')
    ap.add_argument('--resfria', type=float, default=20, help='segundos de resfriamento')
    ap.add_argument('--horas', type=float, default=15, help='jornada, para a conta de energia')
    ap.add_argument('--kwh', type=float, default=0.90, help='R$ por kWh')
    args = ap.parse_args()

    if not shutil.which('tegrastats'):
        sys.exit('tegrastats não encontrado — ele vem com o JetPack.')

    if args.comando:
        carga = args.comando
    else:
        eng = args.engine
        if not eng:
            cands = sorted(f for f in os.listdir('.') if f.endswith('.engine'))
            if not cands:
                sys.exit('nenhuma engine .engine aqui — use --engine ou --comando')
            eng = cands[0]
        trtexec = next((p for p in ('/usr/src/tensorrt/bin/trtexec',
                                    shutil.which('trtexec') or '')
                        if p and os.path.exists(p)), None)
        if not trtexec:
            sys.exit('trtexec não encontrado — use --comando')
        # duration em segundos; o trtexec fica martelando a GPU o tempo todo,
        # que é o pior caso de consumo do modelo
        carga = (f'{trtexec} --loadEngine={eng} --duration={args.carga:.0f} '
                 f'--warmUp=1000 --avgRuns=100')

    print('=' * 66)
    print(' Consumo e temperatura — Vígil.ia no Jetson')
    print('=' * 66)
    print(f'modo de energia : {modo_energia()}')
    print(f'carga           : {carga[:120]}')
    print(f'fases           : {args.ocioso:.0f}s ocioso + {args.carga:.0f}s carga'
          f' + {args.resfria:.0f}s resfriando')
    print()
    print('IMPORTANTE: o modo de energia (nvpmodel) muda tudo. Número medido em')
    print('15 W não é comparável com número medido em MAXN — anote o modo junto.')
    print()

    mon = Monitor()
    time.sleep(2)                       # deixa o tegrastats engatar
    if not mon.amostras:
        mon.fim()
        sys.exit('tegrastats não produziu amostras.\n'
                 '  Ele costuma precisar de root. Rode:  sudo -v  e tente de novo,\n'
                 '  ou rode este script com sudo.')

    print(f'[1/3] ocioso ({args.ocioso:.0f}s)…', flush=True)
    t0 = time.time(); time.sleep(args.ocioso); t1 = time.time()

    print(f'[2/3] carga ({args.carga:.0f}s)…', flush=True)
    t2 = time.time()
    proc = subprocess.run(carga, shell=True, capture_output=True, text=True)
    t3 = time.time()
    saida = proc.stdout + proc.stderr

    print(f'[3/3] resfriando ({args.resfria:.0f}s)…', flush=True)
    t4 = time.time(); time.sleep(args.resfria); t5 = time.time()
    mon.fim()

    # descarta os 2 primeiros segundos da carga: rampa de clock, não regime
    r_ocioso = resume(mon.janela(t0 + 2, t1))
    r_carga = resume(mon.janela(t2 + 2, t3))
    r_resfria = resume(mon.janela(t4, t5))

    print()
    print('=' * 66)
    print(' Potência de entrada da placa (VDD_IN)')
    print('=' * 66)
    print(linha_fase('ocioso', r_ocioso))
    print(linha_fase('CARGA', r_carga))
    print(linha_fase('resfriando', r_resfria))

    if r_ocioso and r_carga:
        delta = r_carga['w_med'] - r_ocioso['w_med']
        print()
        print(f'  >>> o modelo custa {delta:+.2f} W sobre a linha de base')
        print(f'      (placa ligada sem fazer nada já consome {r_ocioso["w_med"]:.2f} W)')

        wh = r_carga['w_med'] * args.horas
        print()
        print('=' * 66)
        print(f' Jornada de {args.horas:.0f} h')
        print('=' * 66)
        print(f'  energia    : {wh:.0f} Wh/dia = {wh/1000:.2f} kWh/dia')
        print(f'  custo      : R$ {wh/1000*args.kwh:.2f}/dia  '
              f'(R$ {wh/1000*args.kwh*30:.2f}/mês a R$ {args.kwh:.2f}/kWh)')
        print(f'  fonte      : mínimo {r_carga["w_max"]*1.3:.0f} W '
              f'(pico medido {r_carga["w_max"]:.1f} W + 30% de folga)')

    # trilhos separados: diz se o gasto é de GPU/CPU ou do resto da placa
    if r_carga:
        amostras = mon.janela(t2 + 2, t3)
        outros = sorted({k for a in amostras for k in a['trilhos']} - {'VDD_IN'})
        if outros:
            print()
            print('  Por trilho, sob carga:')
            for t in outros:
                rt = resume(amostras, t)
                if rt:
                    print(f'    {t:20s} {rt["w_med"]:6.2f} W  (pico {rt["w_max"]:5.2f})')

    print()
    print('=' * 66)
    print(' Térmico')
    print('=' * 66)
    if r_carga and r_carga['tj_max'] is not None:
        tj = r_carga['tj_max']
        print(f'  tj máximo sob carga: {tj:.1f}°C')
        # o Orin começa a reduzir clock perto de 95-100°C; abaixo de 80 há folga
        if tj >= 95:
            print('  >>> THROTTLING PROVÁVEL: acima de ~95°C o Orin reduz clock.')
            print('      Numa câmara FECHADA vai ser pior. Precisa de ventilação.')
        elif tj >= 80:
            print('  >>> quente. Numa câmara fechada, a 14-16 h isso sobe — meça lá.')
        else:
            print('  >>> folga térmica boa NESTE ambiente.')
        print()
        print('  Este ensaio foi ao ar livre e por poucos minutos. A jornada real')
        print('  é dentro da câmara fechada, por 14-16 h: repita lá antes de')
        print('  considerar o número final.')

    if r_carga and r_carga['ram_max']:
        print()
        print(f'  RAM máxima sob carga: {r_carga["ram_max"]} MB')

    m = re.search(r'Throughput: ([\d.]+) qps', saida)
    if m and r_carga and r_ocioso:
        qps = float(m.group(1))
        delta = r_carga['w_med'] - r_ocioso['w_med']
        print()
        print('=' * 66)
        print(' Eficiência')
        print('=' * 66)
        print(f'  {qps:.1f} qps a {r_carga["w_med"]:.2f} W')
        print(f'  {qps/r_carga["w_med"]:.1f} inferências por watt (placa inteira)')
        if delta > 0:
            print(f'  {qps/delta:.1f} inferências por watt (só o custo do modelo)')

    print()
    print('Anote junto do resultado: modo de energia, temperatura ambiente e se')
    print('a placa estava aberta ou dentro da câmara. Sem isso o número não se')
    print('compara com nada depois.')


if __name__ == '__main__':
    main()
