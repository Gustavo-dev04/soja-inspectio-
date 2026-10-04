#!/usr/bin/env python3
"""Vígil.ia — projeta o qps dos backbones candidatos no Jetson Orin Nano.

Existe porque a pergunta "qual backbone?" só tem resposta depois de duas
outras: (1) a licença permite vender? (2) quantos qps ele dá NO APARELHO?

A tabela pública de latência é medida em T4. O Orin Nano é outra máquina, e o
fator de conversão **não é constante** — depende de quanto o modelo é limitado
por cálculo. Medimos dois pontos do RF-DETR no nosso aparelho, e é a partir
deles que o resto é projetado:

    RF-DETR-N  384px  98,2 qps  (10,18 ms)   <- medido, 15W
    RF-DETR-S  512px  59,3 qps  (16,86 ms)   <- medido, 15W

    python3 projetar_qps.py                  # tabela completa
    python3 projetar_qps.py --modo super     # e no MAXN_SUPER?
    python3 projetar_qps.py --vazao          # converte qps em kg/h no rig
    python3 projetar_qps.py --medido RF-DETR-L=30.1   # refaz o ajuste
"""
import argparse

# ---------------------------------------------------------------------------
# REFERÊNCIA — todos medidos no MESMO arnês (T4, TensorRT, FP16, batch 1) pelo
# benchmark da Roboflow (SAB). Ter o mesmo arnês é o que torna as linhas
# comparáveis entre si; números de fornecedores diferentes não são.
# AP_RF é RF100-VL AP50:95 — transferência para dataset pequeno e customizado,
# que é o NOSSO caso. COCO mede outra coisa.
# ---------------------------------------------------------------------------
REFERENCIA = [
    # nome            fam      res  params  AP_coco AP_RF100VL  ms_T4  licença
    ('RF-DETR-N',    'vit',    384,  30.5,  48.4,  57.7,  2.3, 'Apache 2.0'),
    ('RF-DETR-S',    'vit',    512,  32.1,  53.0,  60.2,  3.5, 'Apache 2.0'),
    ('RF-DETR-M',    'vit',    576,  33.7,  54.7,  61.2,  4.4, 'Apache 2.0'),
    ('RF-DETR-L',    'vit',    704,  33.9,  56.5,  62.2,  6.8, 'Apache 2.0'),
    ('RF-DETR-XL',   'vit',    700, 126.4,  58.6,  62.9, 11.5, 'PML 1.0'),
    ('RF-DETR-2XL',  'vit',    880, 126.9,  60.1,  63.2, 17.2, 'PML 1.0'),
    ('LW-DETR-T',    'vit',    640,  12.1,  42.9,  57.1,  1.9, 'Apache 2.0'),
    ('LW-DETR-S',    'vit',    640,  14.6,  48.0,  57.4,  2.6, 'Apache 2.0'),
    ('LW-DETR-M',    'vit',    640,  28.2,  52.6,  59.8,  4.4, 'Apache 2.0'),
    ('LW-DETR-L',    'vit',    640,  46.8,  56.1,  61.5,  6.9, 'Apache 2.0'),
    ('D-FINE-N',     'cnn',    640,   3.8,  42.7,  58.2,  2.1, 'Apache 2.0'),
    ('D-FINE-S',     'cnn',    640,  10.2,  50.6,  60.3,  3.5, 'Apache 2.0'),
    ('D-FINE-M',     'cnn',    640,  19.2,  55.0,  60.6,  5.4, 'Apache 2.0'),
    ('D-FINE-L',     'cnn',    640,  31.0,  57.2,  61.6,  7.5, 'Apache 2.0'),
    ('D-FINE-X',     'cnn',    640,  62.0,  59.3,  62.2, 11.5, 'Apache 2.0'),
    ('YOLO11-S',     'cnn',    640,   9.4,  44.4,  56.2,  3.2, 'AGPL-3.0'),
    ('YOLO11-X',     'cnn',    640,  56.9,  50.9,  56.2, 10.5, 'AGPL-3.0'),
    ('YOLO26-S',     'cnn',    640,   9.4,  47.7,  57.0,  2.6, 'AGPL-3.0'),
    ('YOLO26-L',     'cnn',    640,  25.3,  54.1,  59.3,  5.7, 'AGPL-3.0'),
]

# medido por nós, no aparelho, 15W, FP16, com o vigil_jetson.py
MEDIDO_15W = {'RF-DETR-N': 98.2, 'RF-DETR-S': 59.3}

# MAXN_SUPER sobe o clock da GPU de 635 para 1020 MHz. É teto: a parte do
# modelo limitada por memória não acompanha o clock.
CLOCK_SUPER = 1020 / 635

# Extremos do fator T4->Orin realmente OBSERVADOS. O piso vem de uma CNN
# pequena (YOLOv8n 640: 4,43 ms em MAXN_SUPER -> ~7,1 ms a 15W, contra 2,5 ms
# de um YOLO11-N no T4 = 2,9x). O teto é a inclinação marginal do nosso próprio
# ajuste (5,6x). A distância entre os dois é o tamanho da nossa ignorância para
# arquiteturas que não medimos.
FATOR_MIN, FATOR_MAX = 2.9, 5.6

LICENCA_OK = {'Apache 2.0'}


def ajustar(medido):
    """Reta ms_orin = a * ms_t4 + b a partir dos pontos medidos.

    Duas incógnitas; com 2 pontos é exato, com mais é mínimos quadrados. O
    coeficiente linear costuma dar NEGATIVO, e isso não é erro: o T4 paga um
    custo fixo por chamada que o Orin não paga proporcionalmente, então a razão
    ms_orin/ms_t4 CRESCE com o tamanho do modelo em vez de ser constante.
    """
    ref = {n: t4 for n, _, _, _, _, _, t4, _ in REFERENCIA}
    pts = [(ref[n], 1000.0 / q) for n, q in medido.items() if n in ref]
    if len(pts) < 2:
        raise SystemExit('preciso de pelo menos 2 modelos medidos para ajustar')
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    det = n * sxx - sx * sx
    a = (n * sxy - sx * sy) / det
    b = (sy * a * 0 + sy - a * sx) / n
    return a, b


def main():
    ap = argparse.ArgumentParser(description='Projeta qps dos backbones no Jetson')
    ap.add_argument('--modo', choices=['15w', 'super'], default='15w',
                    help='modo de energia do Jetson (medimos tudo em 15W)')
    ap.add_argument('--medido', action='append', default=[], metavar='NOME=QPS',
                    help='acrescenta uma medição real e refaz o ajuste')
    ap.add_argument('--vazao', action='store_true',
                    help='converte o qps em kg/h no rig (usa calcular_vazao)')
    ap.add_argument('--tudo', action='store_true',
                    help='mostra também os AGPL e os de licença restrita')
    args = ap.parse_args()

    medido = dict(MEDIDO_15W)
    for m in args.medido:
        nome, _, qps = m.partition('=')
        medido[nome.strip()] = float(qps)

    a, b = ajustar(medido)
    escala = 1.0 / CLOCK_SUPER if args.modo == 'super' else 1.0
    rotulo = 'MAXN_SUPER' if args.modo == 'super' else '15W'

    print('=' * 78)
    print(f' Backbones candidatos -> qps no Jetson Orin Nano 8GB ({rotulo}, FP16)')
    print('=' * 78)
    print(f'Ajuste a partir de {len(medido)} medição(ões) nossa(s): '
          f'ms_orin = {a:.3f} x ms_t4 {b:+.3f}')
    if args.modo == 'super':
        print(f'MAXN_SUPER aplicado como teto de clock ({CLOCK_SUPER:.2f}x). '
              'O ganho real é MENOR e precisa ser medido.')
    print()
    print(f'{"modelo":<13}{"res":>5}{"AP RF100VL":>11}{"ms T4":>7}'
          f'{"qps no Orin":>21}   licença')
    print('-' * 78)

    for nome, fam, res, par, apc, apr, t4, lic in REFERENCIA:
        if not args.tudo and lic not in LICENCA_OK:
            continue
        if nome in medido:
            qps = medido[nome] / escala
            marca = f'{qps:.1f} ' + ('medido' if args.modo == '15w' else 'teto')
        elif fam == 'vit':
            marca = f'{1000 / ((a * t4 + b) * escala):.1f} ±10%'
        else:
            lo = 1000 / (t4 * FATOR_MAX * escala)
            hi = 1000 / (t4 * FATOR_MIN * escala)
            marca = f'{lo:.0f}-{hi:.0f} a medir'
        print(f'{nome:<13}{res:>5}{apr:>11.1f}{t4:>7.1f}{marca:>21}   {lic}')

    print('-' * 78)
    print('MEDIDO   = no nosso aparelho, com o vigil_jetson.py.')
    print('±10%     = projetado pela reta acima; mesma família dos pontos medidos,')
    print('           logo a extrapolação é curta e confiável.')
    print('a medir  = OUTRA arquitetura. O fator T4->Orin observado varia de')
    print(f'           {FATOR_MIN:.1f}x (CNN pequena) a {FATOR_MAX:.1f}x (ViT grande), e essa faixa')
    print('           é larga demais para decidir. Exporte o ONNX e meça.')

    if args.vazao:
        import calcular_vazao as cv
        print()
        print('=' * 78)
        print(' O que cada qps vira de vazão no rig (faixa 100 mm, 2 recortes,')
        print(' sobreposição 20%, rastreamento com compensação)')
        print('=' * 78)
        print(f'{"modelo":<13}{"entrada":>8}{"grão":>7}{"varred/s":>10}'
              f'{"v.max":>9}{"kg/h":>7}   meta 33 kg/h')
        print('-' * 78)
        for nome, fam, res, par, apc, apr, t4, lic in REFERENCIA:
            if not args.tudo and lic not in LICENCA_OK:
                continue
            if nome in medido:
                qps = medido[nome] / escala
            elif fam == 'vit':
                qps = 1000 / ((a * t4 + b) * escala)
            else:
                continue                      # faixa larga demais: não vira kg/h
            d = cv.dist_para_tiles(120, 100.0, 2, 0.2, entrada=res)
            r = cv.analisa(120, d, 100.0, 2, qps, compensado=True,
                           sobrepor_frac=0.2, entrada=res)
            ok = '✅' if r['kg_h'] >= 33.3 else '❌'
            det = '' if r['grao_px'] >= 60 else ' ⚠sem detalhe'
            print(f'{nome:<13}{res:>8}{r["grao_px"]:>5.0f}px{r["varreduras_s"]:>10.1f}'
                  f'{r["v_max"]:>7.0f}mm/s{r["kg_h"]:>7.0f}   {ok} '
                  f'{r["kg_h"]/33.3:.2f}x{det}')
        print('-' * 78)
        print('A coluna "grão" é a armadilha desta tabela. Cobrir os mesmos 100 mm')
        print('com uma entrada menor obriga a AFASTAR a câmera, e o grão chega ao')
        print('modelo com menos pixel. Abaixo de ~60 px a textura do defeito')
        print('(mancha, casca danificada) some antes da inferência: o modelo fica')
        print('rápido e cego. Vazão alta com grão pequeno não é ganho, é troca.')
        print()
        print('Detalhe de qualquer linha: python3 calcular_vazao.py --fps <qps> '
              '--compensado')


if __name__ == '__main__':
    main()
