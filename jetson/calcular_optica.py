#!/usr/bin/env python3
"""Vígil.ia — calcula o campo de visão e o tamanho do grão em pixels.

Serve pra decidir LENTE e DISTÂNCIA antes de imprimir o flange, porque depois
disso a geometria fica congelada. A pergunta que ele responde: com essa lente,
nessa distância, o grão chega ao modelo com quantos pixels?

    python3 calcular_optica.py                 # cenário atual (120°, 15 cm)
    python3 calcular_optica.py --entrada 512   # e se o modelo fosse o Small?
    python3 calcular_optica.py --fov 30        # e se a lente fosse 30°?
    python3 calcular_optica.py --alvo-px 80    # que lente dá 80 px por grão?
"""
import argparse
import math

# IMX219: 3280x2464 px de 1,12 µm -> área ativa 3,674 x 2,760 mm
SENSOR_W, SENSOR_H = 3.674, 2.760
SENSOR_DIAG = math.hypot(SENSOR_W, SENSOR_H)          # 4,59 mm
CAPTURA_W, CAPTURA_H = 1640, 1232                     # modo binado (FOV completo)
ROI_W, ROI_H = 3280, 2464                             # modo cheio, usado no ROI 1:1
ENTRADA_MODELO = 704                                  # RF-DETR Large (704 nativo)
CANVAS_TREINO = 704                                   # canvas do dataset sintético
GRAO_MM = 7.0                                         # soja: ~6-8 mm

# O modelo é treinado com grãos ocupando 60-150 px do canvas — 9% a 23% da
# largura. Como o canvas agora é igual à entrada, a faixa vale direto em px.
ALVO_MIN_PX = round(60 * ENTRADA_MODELO / CANVAS_TREINO)
ALVO_MAX_PX = round(150 * ENTRADA_MODELO / CANVAS_TREINO)


def focal_de_fov(fov_graus, dim_sensor):
    return (dim_sensor / 2) / math.tan(math.radians(fov_graus) / 2)


def fov_de_focal(f, dim_sensor):
    return math.degrees(2 * math.atan((dim_sensor / 2) / f))


def analisa(fov_diag, dist_mm, quadrado=True):
    f = focal_de_fov(fov_diag, SENSOR_DIAG)
    campo_w = SENSOR_W * dist_mm / f
    campo_h = SENSOR_H * dist_mm / f
    # o app recorta o quadrado central: o lado do quadrado é a ALTURA do quadro
    if quadrado:
        campo_util = campo_h
        px_captura = CAPTURA_H
    else:
        campo_util = campo_w
        px_captura = CAPTURA_W
    px_por_mm_captura = px_captura / campo_util
    grao_px_captura = GRAO_MM * px_por_mm_captura
    grao_px_modelo = grao_px_captura * (ENTRADA_MODELO / px_captura)
    graos_por_lado = campo_util / (GRAO_MM * 1.4)   # com folga entre grãos
    return {
        'focal': f, 'campo_w': campo_w, 'campo_h': campo_h,
        'campo_util': campo_util, 'grao_px_captura': grao_px_captura,
        'grao_px_modelo': grao_px_modelo,
        'graos_frame': int(graos_por_lado ** 2),
        'fov_h': fov_de_focal(f, SENSOR_W),
    }


def analisa_roi(fov_diag, dist_mm):
    """Recorte 1:1 no centro do sensor cheio, SEM reescalar.

    É o modo que o rig usa de verdade. Aqui o grão em px é decidido só pela
    ÓPTICA (px/mm do sensor cheio) — a entrada do modelo não muda o detalhe do
    grão, muda a ÁREA que cabe no quadro.
    """
    f = focal_de_fov(fov_diag, SENSOR_DIAG)
    campo_h = SENSOR_H * dist_mm / f
    px_por_mm = ROI_H / campo_h
    grao_px = GRAO_MM * px_por_mm
    janela_mm = ENTRADA_MODELO / px_por_mm
    return {
        'px_por_mm': px_por_mm, 'grao_px': grao_px, 'janela_mm': janela_mm,
        'graos_frame': int((janela_mm / (GRAO_MM * 1.4)) ** 2),
    }


def imprime_roi(fov, dist, r):
    print(f'\n>>> MODO ROI 1:1 (o que o rig usa: --roi {ENTRADA_MODELO})')
    print(f'  sensor cheio   : {ROI_W}x{ROI_H} -> {r["px_por_mm"]:.1f} px/mm')
    print(f'  grão de {GRAO_MM:.0f} mm    : {r["grao_px"]:.0f} px '
          f'(fixo pela óptica, não muda com a entrada do modelo)')
    print(f'  janela útil    : {r["janela_mm"]:.0f} x {r["janela_mm"]:.0f} mm  '
          f'-> ~{r["graos_frame"]} grãos por quadro')
    px = r['grao_px']
    if px < ALVO_MIN_PX:
        print(f'  >>> {px:.0f} px fica ABAIXO do alvo ({ALVO_MIN_PX}-{ALVO_MAX_PX}): '
              f'aproximar a câmera é o único ajuste que sobe esse número')
    elif px > ALVO_MAX_PX:
        print(f'  >>> ACIMA do alvo: dá pra afastar e ganhar área')
    else:
        print(f'  >>> DENTRO do alvo ({ALVO_MIN_PX}-{ALVO_MAX_PX} px)')


def imprime(titulo, fov, dist, r):
    print(f'\n{titulo}')
    print(f'  lente          : FOV {fov:.0f}° diagonal  ->  focal ~{r["focal"]:.1f} mm '
          f'(FOV horizontal {r["fov_h"]:.0f}°)')
    print(f'  distância      : {dist/10:.0f} cm')
    print(f'  campo de visão : {r["campo_w"]:.0f} x {r["campo_h"]:.0f} mm')
    print(f'  após crop quad.: {r["campo_util"]:.0f} x {r["campo_util"]:.0f} mm')
    print(f'  grão de {GRAO_MM:.0f} mm    : {r["grao_px_captura"]:.0f} px na captura  ->  '
          f'{r["grao_px_modelo"]:.0f} px na entrada do modelo')
    print(f'  cabem ~{r["graos_frame"]} grãos no quadro')
    px = r['grao_px_modelo']
    if px < ALVO_MIN_PX:
        print(f'  >>> ABAIXO do alvo ({ALVO_MIN_PX}-{ALVO_MAX_PX} px): o grão chega '
              f'{ALVO_MIN_PX/px:.1f}x menor que no treino')
    elif px > ALVO_MAX_PX:
        print(f'  >>> ACIMA do alvo: poucos grãos por quadro')
    else:
        print(f'  >>> DENTRO do alvo ({ALVO_MIN_PX}-{ALVO_MAX_PX} px)')


def main():
    global GRAO_MM, ENTRADA_MODELO, ALVO_MIN_PX, ALVO_MAX_PX
    ap = argparse.ArgumentParser(description='Óptica do rig de inspeção')
    ap.add_argument('--fov', type=float, default=120, help='FOV diagonal da lente (graus)')
    ap.add_argument('--dist', type=float, default=150, help='distância de trabalho (mm)')
    ap.add_argument('--alvo-px', type=float, default=None,
                    help='quantos px por grão você quer na entrada do modelo')
    ap.add_argument('--grao', type=float, default=None, help='tamanho do grão (mm)')
    ap.add_argument('--entrada', type=int, default=None,
                    help=f'entrada do modelo em px (padrão {ENTRADA_MODELO})')
    args = ap.parse_args()

    if args.grao:
        GRAO_MM = args.grao
    if args.entrada:
        ENTRADA_MODELO = args.entrada
        ALVO_MIN_PX = round(60 * ENTRADA_MODELO / CANVAS_TREINO)
        ALVO_MAX_PX = round(150 * ENTRADA_MODELO / CANVAS_TREINO)

    print('=' * 68)
    print(' Óptica do rig — IMX219 (3,67 x 2,76 mm) + crop quadrado central')
    print('=' * 68)
    print(f'Referência: o modelo foi treinado com grãos de {ALVO_MIN_PX}-{ALVO_MAX_PX} px')
    print(f'na entrada de {ENTRADA_MODELO}px. Muito menor que isso, a textura do')
    print('defeito (mancha, casca danificada) some antes de chegar no modelo.')

    imprime('>>> CENÁRIO ATUAL (quadro inteiro reduzido até a entrada)',
            args.fov, args.dist, analisa(args.fov, args.dist))
    imprime_roi(args.fov, args.dist, analisa_roi(args.fov, args.dist))

    if args.alvo_px:
        alvo = args.alvo_px
        campo = GRAO_MM * ENTRADA_MODELO / alvo
        f = SENSOR_H * args.dist / campo
        fov_d = fov_de_focal(f, SENSOR_DIAG)
        print(f'\n>>> PARA {alvo:.0f} px POR GRÃO a {args.dist/10:.0f} cm')
        print(f'  campo útil necessário : {campo:.0f} mm')
        print(f'  focal necessária      : ~{f:.1f} mm')
        print(f'  FOV diagonal          : ~{fov_d:.0f}°')

    print('\n' + '-' * 68)
    print('ALTERNATIVAS (mesma distância de 15 cm, trocando só a lente M12)')
    print('-' * 68)
    print(f'{"FOV diag":>9s} {"focal":>7s} {"campo útil":>11s} {"grão no modelo":>15s} '
          f'{"grãos/quadro":>13s}')
    for fov in (120, 90, 60, 45, 30, 20):
        r = analisa(fov, args.dist)
        marca = '  <-- alvo' if ALVO_MIN_PX <= r['grao_px_modelo'] <= ALVO_MAX_PX else ''
        print(f'{fov:8.0f}° {r["focal"]:6.1f}mm {r["campo_util"]:9.0f}mm '
              f'{r["grao_px_modelo"]:13.0f}px {r["graos_frame"]:12d}{marca}')

    print('\n' + '-' * 68)
    print('E SE APROXIMAR, mantendo a lente de 120°?')
    print('-' * 68)
    for d in (150, 100, 70, 50, 30):
        r = analisa(120, d)
        marca = '  <-- alvo' if ALVO_MIN_PX <= r['grao_px_modelo'] <= ALVO_MAX_PX else ''
        print(f'  {d/10:4.0f} cm -> campo {r["campo_util"]:5.0f} mm, grão '
              f'{r["grao_px_modelo"]:5.0f} px, ~{r["graos_frame"]:4d} grãos{marca}')
    print('\n  (aproximar tem limite: distância mínima de foco da lente e o ring')
    print('   light começa a sombrear o próprio campo)')

    print('\n' + '=' * 68)
    print('COMO CONFERIR NA PRÁTICA (2 min, vale mais que a conta):')
    print('  1. Ponha uma RÉGUA no fundo, na distância de trabalho')
    print('  2. python3 vigil_jetson.py --camera csi --quadrado --csi-sem-trava')
    print('  3. Veja quantos mm cabem na largura do quadro')
    print(f'  4. grão_px = {GRAO_MM:.0f}mm / (mm no quadro) * {ENTRADA_MODELO}')
    print('=' * 68)


if __name__ == '__main__':
    main()
