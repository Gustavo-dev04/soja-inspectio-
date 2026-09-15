#!/usr/bin/env python3
"""Testes de projetar_qps.py — rodam no PC, sem Jetson.

O risco deste script não é quebrar, é MENTIR com aparência de conta: devolver
um qps plausível a partir de um ajuste errado. Os testes atacam isso.

    python3 testar_projecao.py
"""
import sys

import projetar_qps as pq

falhas = []


def checa(nome, cond, detalhe=''):
    print(f'{"ok  " if cond else "FALHA"}  {nome}' + (f'   {detalhe}' if detalhe else ''))
    if not cond:
        falhas.append(nome)


ref = {n: t4 for n, _, _, _, _, _, t4, _ in pq.REFERENCIA}
a, b = pq.ajustar(pq.MEDIDO_15W)

# 1. o ajuste tem que devolver de volta os pontos que o geraram. Com 2 pontos e
# 2 incógnitas é exato — erro aqui é erro de álgebra, não de modelo.
for nome, qps in pq.MEDIDO_15W.items():
    prev = 1000 / (a * ref[nome] + b)
    checa(f'ajuste reproduz {nome}', abs(prev - qps) < 0.05,
          f'{prev:.2f} vs {qps:.1f} qps')

# 2. a inclinação é o custo marginal de trocar T4 por Orin. Fora de 4-7x algo
# está errado nas medições ou na referência.
checa('inclinação plausível', 4.0 <= a <= 7.0, f'{a:.3f}x')

# 3. mais latência no T4 nunca pode dar mais qps no Orin.
proj = [(n, 1000 / (a * t4 + b))
        for n, fam, _, _, _, _, t4, _ in pq.REFERENCIA if fam == 'vit']
ordenado = sorted(proj, key=lambda p: ref[p[0]])
checa('projeção é monótona',
      all(x[1] >= y[1] - 1e-9 for x, y in zip(ordenado, ordenado[1:])))

# 4. o Large tem que cair na faixa que os três métodos independentes deram
# (28-31 qps). Se sair disso, a decisão de arquitetura precisa ser revista.
large = 1000 / (a * ref['RF-DETR-L'] + b)
checa('RF-DETR-L entre 26 e 32 qps', 26 <= large <= 32, f'{large:.1f} qps')

# 5. conferência independente do T4: o Large é a Small a 704 px, então escalar
# a NOSSA medição pela área tem que bater com a projeção dentro de ~15%.
por_area = pq.MEDIDO_15W['RF-DETR-S'] * (512 / 704) ** 2
checa('bate com escala por área', abs(large - por_area) / por_area < 0.15,
      f'{large:.1f} vs {por_area:.1f} qps')

# 6. medição nova tem que MUDAR o ajuste — se não muda, o parâmetro é ignorado
# e o script estaria fingindo aprender.
a2, _ = pq.ajustar({**pq.MEDIDO_15W, 'RF-DETR-L': 22.0})
checa('medição nova reajusta', abs(a2 - a) > 0.1, f'{a:.3f} -> {a2:.3f}')

# 7. o modo super é teto de clock; não pode ser menor que o 15W.
checa('super > 15W', pq.CLOCK_SUPER > 1.0, f'{pq.CLOCK_SUPER:.3f}x')

# 8. a faixa de ignorância entre arquiteturas precisa ser larga de verdade —
# estreitá-la sem medir seria inventar precisão.
checa('faixa cruzada é honesta', pq.FATOR_MAX / pq.FATOR_MIN > 1.5,
      f'{pq.FATOR_MIN}x a {pq.FATOR_MAX}x')

# 9. nenhum modelo de licença proibida pode estar marcado como utilizável.
proibidas = {lic for *_, lic in pq.REFERENCIA} - pq.LICENCA_OK
checa('só Apache 2.0 é liberado', pq.LICENCA_OK == {'Apache 2.0'},
      f'bloqueadas: {sorted(proibidas)}')

print()
if falhas:
    print(f'{len(falhas)} falha(s): ' + ', '.join(falhas))
    sys.exit(1)
print('tudo certo')
