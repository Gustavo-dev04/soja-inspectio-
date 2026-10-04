#!/usr/bin/env python3
"""Testa o parser do tegrastats do `bench_energia.py` — roda em qualquer máquina.

O parser é a parte que erra em silêncio: uma regex que não casa devolve amostra
vazia, e o relatório sai com 0 W parecendo medição. Aqui ele é conferido contra
linhas reais de Orin Nano e do formato antigo (AGX), incluindo os casos chatos —
núcleo de CPU "off", linha sem potência, e o resumo por trilho.

    python3 testar_bench_energia.py
"""
import os, sys, importlib.util
spec = importlib.util.spec_from_file_location('be', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_energia.py'))
be = importlib.util.module_from_spec(spec); spec.loader.exec_module(be)

# formato do Orin Nano (JetPack 6/7)
ORIN = ("RAM 3244/7620MB (lfb 8x4MB) SWAP 0/3810MB (cached 0MB) "
        "CPU [12%@1420,3%@1420,0%@1420,1%@1420,2%@1420,0%@1420] "
        "EMC_FREQ 8%@2133 GR3D_FREQ 74%@[1020] "
        "cpu@52.5C soc2@50.5C soc0@49.5C gpu@51.5C tj@52.5C soc1@49.5C "
        "VDD_IN 8234mW/7100mW VDD_CPU_GPU_CV 3456mW/2900mW VDD_SOC 1789mW/1700mW")
# formato antigo (AGX Xavier) — nomes de trilho diferentes
AGX = ("RAM 5000/15000MB (lfb 100x4MB) CPU [5%@2265,off,off,4%@2265] "
       "GR3D_FREQ 0% thermal@45C "
       "VDD_IN 5000mW/4900mW VDD_CPU_GPU_CV 1200mW/1100mW")
# linha sem potência (algumas versões imprimem cabeçalho)
LIXO = "tegrastats iniciado"

d = be.parse_tegrastats(ORIN)
assert d is not None
print('trilhos:', d['trilhos'])
assert d['trilhos'] == {'VDD_IN': 8234, 'VDD_CPU_GPU_CV': 3456, 'VDD_SOC': 1789}
assert d['temps']['tj'] == 52.5 and d['temps']['gpu'] == 51.5
assert d['gpu'] == 74, d.get('gpu')
assert abs(d['cpu'] - 3.0) < 0.01, d['cpu']          # média de 12,3,0,1,2,0
assert d['ram_mb'] == 3244 and d['ram_total_mb'] == 7620
print(f"temps: tj={d['temps']['tj']}  gpu={d['gpu']}%  cpu={d['cpu']:.1f}%  ram={d['ram_mb']}MB")
print('OK: linha do Orin Nano')

d2 = be.parse_tegrastats(AGX)
assert d2 and d2['trilhos']['VDD_IN'] == 5000
assert 'tj' not in d2['temps'] and d2['temps']['thermal'] == 45.0
# CPU com núcleos "off" não pode quebrar a média
assert abs(d2['cpu'] - 4.5) < 0.01, d2['cpu']
print(f"OK: formato antigo (núcleos 'off' ignorados, cpu={d2['cpu']}%)")

assert be.parse_tegrastats(LIXO) is None
print('OK: linha sem potência é descartada, não vira amostra zerada')

# --- resumo estatístico ---
import time
agora = time.time()
amostras = []
for i, w in enumerate([5000, 5200, 4900, 12000, 12500, 11800]):
    a = be.parse_tegrastats(ORIN.replace('VDD_IN 8234mW', f'VDD_IN {w}mW'))
    a['t'] = agora + i
    amostras.append(a)
r = be.resume(amostras)
print(f"\nresumo: {r['w_med']:.2f} W média, {r['w_max']:.2f} pico, {r['w_min']:.2f} mín, n={r['n']}")
assert abs(r['w_med'] - 8.5667) < 0.01 and r['w_max'] == 12.5 and r['w_min'] == 4.9
assert r['tj_max'] == 52.5 and r['ram_max'] == 3244
print('OK: média, pico, mínimo e temperatura')

# --- resumo por trilho ---
rg = be.resume(amostras, 'VDD_CPU_GPU_CV')
assert abs(rg['w_med'] - 3.456) < 1e-6, rg['w_med']
print(f"OK: trilho separado (VDD_CPU_GPU_CV = {rg['w_med']:.3f} W)")

# --- janela temporal filtra direito ---
class M: pass
m = M(); m.amostras = amostras; m.janela = be.Monitor.janela.__get__(m)
assert len(m.janela(agora + 3, agora + 5)) == 3
print('OK: janela por tempo seleciona só a fase pedida')

assert be.resume([]) is None and be.linha_fase('x', None).endswith('sem amostras')
print('OK: sem amostras não quebra, avisa')

# --- extração do qps da saída do trtexec ---
import re
saida = "[I] Throughput: 53.1998 qps\n[I] Latency: min = 18.6 ms"
m2 = re.search(r'Throughput: ([\d.]+) qps', saida)
assert m2 and abs(float(m2.group(1)) - 53.1998) < 1e-4
print(f'OK: qps extraído do trtexec ({m2.group(1)})')
print('\n=== parser validado ===')
