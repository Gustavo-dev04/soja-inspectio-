#!/usr/bin/env python3
"""Vígil.ia — calibra o rig: escala (px/mm), exposição e ROI.

Substitui o chute pela medição. Rode com uma RÉGUA no fundo, na distância de
trabalho, e ele responde as três perguntas que travam o rig:

  1. quantos px/mm o rig entrega?      -> define o --roi certo
  2. a exposição está boa?             -> histograma, estouro e fundo
  3. o grão vai chegar com quantos px?  -> compara com o que o modelo espera

    python3 calibrar_rig.py                    # interativo, com a régua
    python3 calibrar_rig.py --imagem foto.jpg  # analisa uma foto já tirada

Teclas na janela: ESPAÇO mede · e exposição · s salva · q sai
"""
import argparse
import sys

import cv2
import numpy as np

ENTRADA_MODELO = 512
GRAO_MM = 7.0
ALVO_MIN_PX, ALVO_MAX_PX = 48, 120     # faixa em que o modelo foi treinado


def analisa_exposicao(img):
    """Fundo cravado em 0 e grão estourado em 255 são informação perdida."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tot = cinza.size
    preto = int((cinza <= 2).sum())
    estourado = int((cinza >= 253).sum())
    # O fundo é escuro e domina; o grão é o que fica acima do Otsu.
    # Usa a MÁSCARA que o OpenCV devolve, não uma comparação própria: o Otsu
    # entrega o limiar como limite inferior e o THRESH_BINARY corta em `> lim`,
    # então reimplementar com `>=` joga o fundo inteiro dentro do "grão".
    _, mascara = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fundo = cinza[mascara == 0]
    grao = cinza[mascara > 0]
    return {
        'preto_pct': 100 * preto / tot,
        'estourado_pct': 100 * estourado / tot,
        'fundo_media': float(fundo.mean()) if fundo.size else 0.0,
        'grao_media': float(grao.mean()) if grao.size else 0.0,
        'grao_pct': 100 * grao.size / tot,
    }


def relatorio_exposicao(e):
    linhas = ['  --- exposição ---',
              f'  fundo  : média {e["fundo_media"]:.0f}/255   '
              f'({e["preto_pct"]:.1f}% cravado em 0)',
              f'  grão   : média {e["grao_media"]:.0f}/255   '
              f'({e["grao_pct"]:.1f}% do quadro)',
              f'  estouro: {e["estourado_pct"]:.2f}% dos pixels em 255']
    if e['estourado_pct'] > 0.5:
        linhas.append('  >>> ESTOURADO: textura perdida no brilho. Reduza a '
                      'exposição ou ponha difusor no ring light.')
    elif e['preto_pct'] > 40:
        linhas.append('  >>> FUNDO CRAVADO em 0: subiu demais o preto. Não é '
                      'grave, mas dificulta separar grão escuro do fundo.')
    elif e['grao_media'] < 90:
        linhas.append('  >>> ESCURO: aumente a exposição ou a luz.')
    else:
        linhas.append('  >>> exposição OK')
    return '\n'.join(linhas)


def recomenda_roi(px_por_mm):
    grao_px = GRAO_MM * px_por_mm
    print(f'\n  --- escala ---')
    print(f'  {px_por_mm:.2f} px/mm')
    print(f'  grão de {GRAO_MM:.0f} mm  ->  {grao_px:.0f} px na captura')
    if grao_px < ALVO_MIN_PX:
        print(f'  >>> Mesmo SEM reduzir, o grão já sai com {grao_px:.0f} px, abaixo '
              f'do alvo ({ALVO_MIN_PX}-{ALVO_MAX_PX}).')
        print(f'      Aproxime a câmera ou use lente mais fechada.')
        return None
    # ROI 1:1 = entrada do modelo; janela útil que isso cobre
    janela_mm = ENTRADA_MODELO / px_por_mm
    graos = int((janela_mm / (GRAO_MM * 1.4)) ** 2)
    print(f'\n  --- ROI recomendado (recorte 1:1, sem reescalar) ---')
    print(f'  --roi {ENTRADA_MODELO}')
    print(f'  janela útil : {janela_mm:.0f} x {janela_mm:.0f} mm')
    print(f'  grão chega  : {grao_px:.0f} px  '
          f'{"(DENTRO do alvo)" if ALVO_MIN_PX <= grao_px <= ALVO_MAX_PX else "(fora do alvo)"}')
    print(f'  cabem ~{graos} grãos por quadro')
    if grao_px > ALVO_MAX_PX:
        # dá pra usar ROI maior e reduzir um pouco, ganhando área
        lado = int(ENTRADA_MODELO * grao_px / ((ALVO_MIN_PX + ALVO_MAX_PX) / 2))
        print(f'\n  Alternativa: --roi {lado} (recorta mais área e reduz até 512;')
        print(f'  grão fica ~{(ALVO_MIN_PX + ALVO_MAX_PX) / 2:.0f} px e cabe mais grão)')
    return ENTRADA_MODELO


def medir_com_regua(img):
    """Clique nas duas pontas de uma distância conhecida da régua."""
    pontos = []
    janela = 'clique 2 pontos na regua (ESC cancela)'

    def clique(evento, x, y, flags, _):
        if evento == cv2.EVENT_LBUTTONDOWN and len(pontos) < 2:
            pontos.append((x, y))

    vis = img.copy()
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, clique)
    print('\n  Clique nas DUAS pontas de uma distância conhecida da régua…')
    while True:
        v = vis.copy()
        for pt in pontos:
            cv2.circle(v, pt, 6, (0, 255, 255), -1)
        if len(pontos) == 2:
            cv2.line(v, pontos[0], pontos[1], (0, 255, 255), 2)
        cv2.imshow(janela, v)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            cv2.destroyWindow(janela)
            return None
        if len(pontos) == 2:
            cv2.waitKey(400)
            break
    cv2.destroyWindow(janela)
    dist_px = float(np.hypot(pontos[1][0] - pontos[0][0], pontos[1][1] - pontos[0][1]))
    try:
        mm = float(input('  Quantos MILÍMETROS há entre os dois pontos? '))
    except (ValueError, EOFError):
        print('  valor inválido')
        return None
    if mm <= 0:
        return None
    return dist_px / mm


def main():
    ap = argparse.ArgumentParser(description='Calibra o rig de inspeção')
    ap.add_argument('--imagem', default=None, help='analisa uma foto em vez da câmera')
    ap.add_argument('--camera', default='csi')
    ap.add_argument('--sem-trava', action='store_true',
                    help='CSI com AE/AWB automáticos (só para achar a exposição)')
    args = ap.parse_args()

    print('=' * 66)
    print(' Calibração do rig — Vígil.ia')
    print('=' * 66)
    print(f'Alvo: o grão deve chegar ao modelo com {ALVO_MIN_PX}-{ALVO_MAX_PX} px')
    print('(faixa em que ele foi treinado). Ponha uma RÉGUA no fundo.')

    if args.imagem:
        img = cv2.imread(args.imagem)
        if img is None:
            sys.exit(f'não abri {args.imagem}')
        print(f'\nimagem: {args.imagem}  ({img.shape[1]}x{img.shape[0]})')
        print(relatorio_exposicao(analisa_exposicao(img)))
        escala = medir_com_regua(img)
        if escala:
            recomenda_roi(escala)
        return

    sys.path.insert(0, __file__.rsplit('/', 1)[0])
    from vigil_jetson import abrir_camera     # reaproveita o pipeline travado
    cap = abrir_camera(args.camera, travar_csi=not args.sem_trava)
    ok, frame = cap.read()
    if not ok or frame is None:
        sys.exit(f'não veio frame de {args.camera}')
    print(f'\ncaptura: {frame.shape[1]}x{frame.shape[0]}')
    print('ESPAÇO mede com a régua · e exposição · s salva · q sai')

    janela = 'calibracao (espaco=medir, e=exposicao, s=salvar, q=sai)'
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        h, w = vis.shape[:2]
        # marca onde ficaria o ROI de 512 e onde é o centro
        lado = min(ENTRADA_MODELO, h, w)
        x, y = (w - lado) // 2, (h - lado) // 2
        cv2.rectangle(vis, (x, y), (x + lado, y + lado), (0, 255, 255), 2)
        cv2.putText(vis, f'ROI {lado}x{lado}', (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(janela, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('e'):
            print('\n' + relatorio_exposicao(analisa_exposicao(frame)))
        if k == ord('s'):
            cv2.imwrite('calibracao.jpg', frame)
            print('  salvo: calibracao.jpg')
        if k == ord(' '):
            escala = medir_com_regua(frame)
            if escala:
                print(relatorio_exposicao(analisa_exposicao(frame)))
                recomenda_roi(escala)
                print('\n  Anote o px/mm na ficha do rig (PADRAO_CAPTURA.md §6).')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
