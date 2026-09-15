#!/usr/bin/env python3
"""Vígil.ia — dimensiona a câmara de inspeção: de geometria até kg/h.

Amarra numa conta só o que estava espalhado em chute: tamanho da câmara,
distância da câmera, quantos recortes (tiles) de entrada cobrem a faixa, fps do
modelo, e a velocidade máxima que o rastreamento aguenta. A saída é a única
pergunta que importa: **essa configuração entrega a vazão alvo?**

    python3 calcular_vazao.py                      # cenário atual
    python3 calcular_vazao.py --tiles 3            # e com 3 recortes?
    python3 calcular_vazao.py --fps 50             # e se o INT8 der 50 qps?
    python3 calcular_vazao.py --meta-kg-dia 1000   # outra meta de vazão

Os dois limites de velocidade são independentes e o menor manda:

  1. VARREDURAS — o grão precisa ser visto LOCK_MIN_FRAMES vezes antes do
     veredito travar. Isso limita quão rápido ele pode cruzar a janela.
  2. RASTREAMENTO — o casamento entre varreduras. Deslocamento grande demais e o
     grão troca de ID no meio da passagem: os votos zeram e a contagem infla,
     em silêncio. São dois regimes:
       * IoU cru        -> precisa de SOBREPOSIÇÃO: ~0,54 grão por varredura.
       * com compensação-> precisa só de NÃO-AMBIGUIDADE: metade do espaçamento
                           entre grãos vizinhos, 0,7 grão por varredura.
     Passando disso, o vizinho da frente fica mais perto do que o grão andou, e
     nenhum algoritmo desempata isso sem outra fonte de informação — é
     ambiguidade de abertura, não falta de código.
"""
import argparse

from calcular_optica import (CANVAS_TREINO, ENTRADA_MODELO, GRAO_MM, ROI_H,
                             ROI_W, SENSOR_DIAG, SENSOR_H, SENSOR_W,
                             focal_de_fov)

# --- grão ---
MASSA_GRAO_G = 0.16       # peso de mil grãos 120-200 g -> 0,12-0,20 g por grão
PASSO = GRAO_MM * 1.4     # centro a centro numa monocamada sem grão encostado

# --- câmara de inspeção (v1 do padrão) ---
FAIXA_MM = 100.0          # largura TRANSVERSAL ao movimento (os 10 cm)
CURSO_MM = 150.0          # comprimento na direção do movimento (os 15 cm)

# --- regras do app (têm que bater com vigil_jetson.py) ---
LOCK_MIN_FRAMES = 8
IOU_MIN = 0.30

# --- meta ---
META_KG_DIA = 500.0
META_HORAS = 15.0


def desloc_max_iou(grao_mm=GRAO_MM, iou_min=IOU_MIN):
    """Maior deslocamento por varredura que ainda casa por IoU.

    Duas caixas quadradas de lado `grao_mm` deslocadas de d na direção do
    movimento: inter = (g-d)*g, união = 2g² - inter. Resolve por bisseção
    porque a inversa analítica não acrescenta nada e a bisseção não erra.
    """
    def iou(d):
        inter = max(0.0, grao_mm - d) * grao_mm
        return inter / (2 * grao_mm * grao_mm - inter)

    lo, hi = 0.0, grao_mm
    for _ in range(60):
        m = (lo + hi) / 2
        if iou(m) >= iou_min:
            lo = m
        else:
            hi = m
    return lo


def px_por_mm(fov_diag, dist_mm):
    """px/mm no sensor CHEIO — é o que o recorte 1:1 entrega ao modelo."""
    f = focal_de_fov(fov_diag, SENSOR_DIAG)
    campo_w = SENSOR_W * dist_mm / f
    return ROI_W / campo_w


def cobertura_px(n_tiles, sobrepor_frac, entrada=ENTRADA_MODELO):
    """Largura útil de n recortes que se sobrepõem em `sobrepor_frac` do lado.

    A sobreposição não é opcional: sem ela o grão em cima da costura é cortado
    ao meio nos dois recortes e nenhum dos dois pedaços vira detecção boa.
    """
    return entrada * (n_tiles - (n_tiles - 1) * sobrepor_frac)


def dist_para_tiles(fov_diag, faixa_mm, n_tiles, sobrepor_frac=0.0,
                    entrada=ENTRADA_MODELO):
    """Distância em que `n_tiles` recortes cobrem exatamente a faixa.

    px/mm cresce quando a câmera aproxima, então existe uma distância única em
    que a faixa ocupa a largura útil dos recortes.
    """
    f = focal_de_fov(fov_diag, SENSOR_DIAG)
    alvo = cobertura_px(n_tiles, sobrepor_frac, entrada) / faixa_mm   # px/mm
    return ROI_W * f / (SENSOR_W * alvo)


def analisa(fov_diag, dist_mm, faixa_mm, n_tiles, fps_modelo,
            compensado=False, sobrepor_frac=0.0, entrada=ENTRADA_MODELO):
    pxmm = px_por_mm(fov_diag, dist_mm)
    grao_px = GRAO_MM * pxmm
    tile_mm = entrada / pxmm                       # janela de UM recorte
    cobertura_mm = min(faixa_mm,
                       cobertura_px(n_tiles, sobrepor_frac, entrada) / pxmm)
    varreduras_s = fps_modelo / n_tiles            # 1 varredura = n_tiles inferências

    # limite 1: o grão precisa de LOCK_MIN_FRAMES varreduras dentro da janela
    v_lock = tile_mm * varreduras_s / LOCK_MIN_FRAMES
    # limite 2: casamento entre varreduras.
    # sem compensação, precisa de sobreposição; com ela, basta não ser ambíguo
    # em relação ao grão vizinho — metade do espaçamento da monocamada.
    desloc = PASSO / 2 if compensado else desloc_max_iou()
    v_iou = desloc * varreduras_s
    v_max = min(v_lock, v_iou)

    graos_s = cobertura_mm * v_max / PASSO ** 2
    return {
        'px_mm': pxmm, 'grao_px': grao_px, 'tile_mm': tile_mm,
        'cobertura_mm': cobertura_mm, 'varreduras_s': varreduras_s,
        'v_lock': v_lock, 'v_iou': v_iou, 'v_max': v_max,
        'graos_s': graos_s, 'kg_h': graos_s * MASSA_GRAO_G * 3600 / 1000,
        'limitante': 'varreduras' if v_lock <= v_iou else 'rastreamento',
        'graos_visiveis': int((tile_mm / PASSO) ** 2 * n_tiles),
        'sobrepor_mm': sobrepor_frac * tile_mm,
    }


def velocidade_para(kg_h, cobertura_mm):
    """Velocidade da esteira que entrega `kg_h` na faixa dada."""
    graos_s = kg_h * 1000 / 3600 / MASSA_GRAO_G
    return graos_s * PASSO ** 2 / cobertura_mm


def imprime(r, meta_kg_h, faixa_mm, n_tiles, dist_mm, fps, alvo_px):
    print(f'  câmera         : {dist_mm/10:.1f} cm  ->  {r["px_mm"]:.2f} px/mm')
    print(f'  recortes       : {n_tiles} x {ENTRADA_MODELO}px  '
          f'(cada um {r["tile_mm"]:.0f} x {r["tile_mm"]:.0f} mm, '
          f'cobrem {r["cobertura_mm"]:.0f} dos {faixa_mm:.0f} mm de faixa)')
    if n_tiles > 1:
        print(f'  sobreposição   : {r["sobrepor_mm"]:.0f} mm entre recortes '
              f'({r["sobrepor_mm"]/GRAO_MM:.1f} grãos)'
              + ('' if r['sobrepor_mm'] >= GRAO_MM else '  ⚠ MENOR que 1 grão'))
    lo, hi = alvo_px
    dentro = 'DENTRO' if lo <= r['grao_px'] <= hi else ('ABAIXO' if r['grao_px'] < lo else 'ACIMA')
    print(f'  grão no modelo : {r["grao_px"]:.0f} px  ({dentro} do alvo {lo}-{hi})')
    print(f'  ~{r["graos_visiveis"]} grãos visíveis por varredura')
    print(f'  varreduras     : {r["varreduras_s"]:.1f}/s  ({fps} qps / {n_tiles} recortes)')
    print(f'  velocidade máx : {r["v_max"]:.0f} mm/s  '
          f'(limitada por {r["limitante"]})')
    print(f'                   varreduras: {r["v_lock"]:.0f} mm/s  |  '
          f'rastreamento: '
          + f'{r["v_iou"]:.0f} mm/s')
    print(f'  vazão máx      : {r["graos_s"]:.0f} grãos/s  =  {r["kg_h"]:.0f} kg/h')
    folga = r['kg_h'] / meta_kg_h if meta_kg_h else 0
    marca = '✅' if folga >= 1 else '❌'
    print(f'  {marca} meta {meta_kg_h:.0f} kg/h -> folga de {folga:.2f}x')
    if folga >= 1:
        v = velocidade_para(meta_kg_h, r['cobertura_mm'])
        print(f'     para bater a meta na mosca: esteira a {v:.0f} mm/s')
    if r['cobertura_mm'] < faixa_mm - 1:
        falta = faixa_mm - r['cobertura_mm']
        print(f'  ⚠ {falta:.0f} mm da faixa ficam FORA do quadro — grão que passar '
              f'por ali não é inspecionado.')


def main():
    ap = argparse.ArgumentParser(description='Dimensiona a câmara de inspeção')
    ap.add_argument('--fov', type=float, default=120, help='FOV diagonal (graus)')
    ap.add_argument('--faixa', type=float, default=FAIXA_MM,
                    help='largura da câmara, transversal ao movimento (mm)')
    ap.add_argument('--tiles', type=int, default=2, help='recortes lado a lado')
    ap.add_argument('--dist', type=float, default=None,
                    help='distância (mm); por padrão calcula a que casa com --tiles')
    ap.add_argument('--fps', type=float, default=28,
                    help='qps do modelo no aparelho (704 FP16 ~28; INT8 a medir)')
    ap.add_argument('--compensado', action='store_true',
                    help='tracking com compensação de movimento (tira o limite de IoU)')
    ap.add_argument('--sobrepor', type=float, default=0.2, metavar='FRAC',
                    help='sobreposição entre recortes, em fração do lado '
                         '(0.2 = 20%%; precisa ser >= 1 grão)')
    ap.add_argument('--meta-kg-dia', type=float, default=META_KG_DIA)
    ap.add_argument('--horas', type=float, default=META_HORAS)
    args = ap.parse_args()

    meta_kg_h = args.meta_kg_dia / args.horas
    graos_dia = args.meta_kg_dia * 1000 / MASSA_GRAO_G
    alvo_px = (round(60 * ENTRADA_MODELO / CANVAS_TREINO),
               round(150 * ENTRADA_MODELO / CANVAS_TREINO))

    print('=' * 70)
    print(' Vazão da câmara de inspeção — Vígil.ia')
    print('=' * 70)
    print(f'META: {args.meta_kg_dia:.0f} kg em {args.horas:.0f} h = '
          f'{meta_kg_h:.0f} kg/h = {graos_dia/1e6:.2f} M grãos/dia = '
          f'{graos_dia/(args.horas*3600):.0f} grãos/s')
    print(f'Grão: {GRAO_MM:.0f} mm, {MASSA_GRAO_G*1000:.0f} mg, passo {PASSO:.1f} mm '
          f'({10000/PASSO**2:.0f} grãos/cm² em monocamada)')
    print(f'Regra: {LOCK_MIN_FRAMES} varreduras para travar o veredito.')
    print(f'Desloc. máx por varredura: {desloc_max_iou():.2f} mm com IoU cru '
          f'(IoU>={IOU_MIN:.2f}) | {PASSO/2:.2f} mm com compensação '
          f'(metade do espaçamento)')

    dist = args.dist or dist_para_tiles(args.fov, args.faixa, args.tiles,
                                        args.sobrepor)
    r = analisa(args.fov, dist, args.faixa, args.tiles, args.fps,
                args.compensado, args.sobrepor)
    print(f'\n>>> CENÁRIO — câmara de {args.faixa:.0f} mm de faixa')
    imprime(r, meta_kg_h, args.faixa, args.tiles, dist, args.fps, alvo_px)

    print('\n' + '-' * 70)
    print(f'ALTERNATIVAS (faixa {args.faixa:.0f} mm; distância ajustada a cada '
          f'nº de recortes)')
    print('-' * 70)
    print(f'{"tiles":>5} {"dist":>7} {"px/mm":>6} {"grão":>6} {"varr/s":>7} '
          f'{"v máx":>9} {"kg/h":>7}  limitante')
    for comp in (False, True):
        print(f'  -- tracking {"COM" if comp else "sem"} compensação de movimento --')
        for n in (1, 2, 3, 4):
            d = dist_para_tiles(args.fov, args.faixa, n, args.sobrepor)
            x = analisa(args.fov, d, args.faixa, n, args.fps, comp, args.sobrepor)
            ok = '✅' if x['kg_h'] >= meta_kg_h else '  '
            print(f'{n:5d} {d/10:6.1f}cm {x["px_mm"]:6.1f} {x["grao_px"]:5.0f}px '
                  f'{x["varreduras_s"]:7.1f} {x["v_max"]:7.0f}mm/s {x["kg_h"]:6.0f} '
                  f'{ok} {x["limitante"]}')

    print('\n' + '=' * 70)
    print('COMO CONFERIR NA PRÁTICA (a conta assume FOV diagonal e spec de M12')
    print('barata, que é aproximada — a régua decide):')
    print('  1. python3 calibrar_rig.py       -> mede px/mm de verdade')
    print('  2. reponha o px/mm medido aqui com --dist até bater')
    print('  3. cronometre a esteira num trecho marcado -> mm/s real')
    print('=' * 70)


if __name__ == '__main__':
    main()
