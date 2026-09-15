#!/usr/bin/env python3
"""Vígil.ia — entra soja, sai dataset pronto para treinar.

    # 1. capturar: uma bandeja de UMA classe por vez
    python3 coletar_dataset.py --classe immature --lote L001
    python3 coletar_dataset.py --classe spotted  --lote L002

    # 2. dividir em train/valid/test quando tiver capturado tudo
    python3 coletar_dataset.py --dividir

    # 3. ver o que já existe
    python3 coletar_dataset.py --status

O resultado é a estrutura que `model/treino_base12k_jetson.ipynb` já sabe ler:

    dataset/pronto/{train,valid,test}/<classe>/*.jpg

Três decisões que definem a qualidade do que sai daqui:

**O rótulo vem da BANDEJA, não do modelo.** Usar o modelo para rotular assa os
erros dele dentro do dataset, e o próximo treino aprende a errar igual. Aqui a
caixa sai do Otsu — o fundo preto fosco do rig torna a segmentação trivial — e a
classe vem de qual lote está passando. Zero anotação manual, zero
realimentação de erro.

**Um grão físico aparece em ~11 quadros, e isso é uma armadilha.** Salvar todos
infla o dataset com cópias quase idênticas, e — pior — uma divisão aleatória
por IMAGEM coloca o mesmo grão no treino e na validação. A validação então mede
memorização e devolve um número bonito e falso. Por isso aqui o rastreamento
agrupa os quadros de cada grão, guarda só os melhores, e a divisão é **por
grão**: todos os recortes de um grão caem no mesmo split.

**Recorte ruim é pior que recorte nenhum.** Grão cortado na borda do quadro,
foto borrada ou grãos encostados viram exemplo errado com rótulo confiante. Os
três são descartados na hora, e o motivo de cada descarte entra no relatório —
se 40% da captura está sendo jogada fora, o problema é o rig, não o filtro.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)

# O vigil_jetson importa tensorrt (do JetPack). A coleta NÃO usa modelo nenhum,
# então num PC sem JetPack ele entra com um stub e o resto funciona igual — dá
# para testar e rodar a divisão fora do Jetson.
try:
    import tensorrt  # noqa: F401
except ImportError:
    sys.modules.setdefault('tensorrt', type(sys)('tensorrt'))
_spec = importlib.util.spec_from_file_location(
    'vigil_jetson', os.path.join(_AQUI, 'vigil_jetson.py'))
vj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vj)

CLASSES = vj.NAMES                      # broken, immature, intact, skin-damaged, spotted
RAIZ = os.path.join(_AQUI, 'dataset')
BRUTO = os.path.join(RAIZ, 'bruto')
PRONTO = os.path.join(RAIZ, 'pronto')
MANIFESTO = os.path.join(BRUTO, 'manifesto.jsonl')

# --- geometria do rig (tem que bater com PADRAO_CAPTURA.md) ---
PX_POR_MM = 12.7
GRAO_MM = 7.0
# grão de 6 a 8 mm, mais folga de calibração: fora disso não é grão
GRAO_PX_MIN = int(5.0 * PX_POR_MM)      # ~63
GRAO_PX_MAX = int(11.0 * PX_POR_MM)     # ~139

# --- filtros de qualidade ---
NITIDEZ_MIN = 60.0      # variância do laplaciano; abaixo disso está borrado
SOLIDEZ_MIN = 0.80      # área/área_do_casco: grãos encostados dão blob irregular
MARGEM_BORDA = 4        # px: grão a menos que isso da borda pode estar cortado
POR_GRAO = 3            # quantos recortes guardar por grão físico (os mais nítidos)
CONTEXTO = 0.15         # borda extra no recorte, em fração do lado do grão

VAL_FRAC, TEST_FRAC = 0.15, 0.10


# ---------------------------------------------------------------- segmentação
def segmentar(frame):
    """Todos os grãos do quadro, por limiarização. Devolve [(x1,y1,x2,y2,solidez)].

    Funciona porque o rig controla o fundo: cartolina preta fosca sob luz
    difusa dá um histograma claramente bimodal, e o Otsu acha o corte sozinho.
    Fora do rig isso não vale — é mais um motivo para o padrão de captura ser
    documento normativo.
    """
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    borrado = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, mascara = cv2.threshold(borrado, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # abertura tira sal-e-pimenta sem comer o grão
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    achados = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        casco = cv2.contourArea(cv2.convexHull(c))
        solidez = area / casco if casco > 0 else 0.0
        achados.append((x, y, x + w, y + h, solidez))
    return achados


def avaliar(frame, caixa, solidez):
    """Este recorte serve? Devolve (serve, motivo, nitidez)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    lado = max(x2 - x1, y2 - y1)

    if lado < GRAO_PX_MIN:
        return False, 'pequeno', 0.0
    if lado > GRAO_PX_MAX:
        # grande demais quase sempre é dois ou mais grãos colados num blob só
        return False, 'grande', 0.0
    if solidez < SOLIDEZ_MIN:
        return False, 'encostado', 0.0
    if (x1 <= MARGEM_BORDA or y1 <= MARGEM_BORDA
            or x2 >= w - MARGEM_BORDA or y2 >= h - MARGEM_BORDA):
        # grão cortado pela borda vira exemplo de "grão quebrado" com rótulo
        # de intacto — exatamente o tipo de erro que o modelo aprende bem
        return False, 'na_borda', 0.0

    recorte = frame[y1:y2, x1:x2]
    nitidez = float(cv2.Laplacian(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY),
                                  cv2.CV_64F).var())
    if nitidez < NITIDEZ_MIN:
        return False, 'borrado', nitidez
    return True, '', nitidez


def recortar(frame, caixa):
    """Recorta com uma borda de contexto, sem sair do quadro."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    pad = int(CONTEXTO * max(x2 - x1, y2 - y1))
    return frame[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)].copy()


# ---------------------------------------------------------------- captura
def capturar(args):
    if args.classe not in CLASSES:
        sys.exit(f'classe inválida: {args.classe}\n  use uma de: {", ".join(CLASSES)}')

    destino = os.path.join(BRUTO, args.classe)
    os.makedirs(destino, exist_ok=True)
    sessao = time.strftime('%Y%m%d-%H%M%S')

    cap = vj.abrir_camera(args.camera, travar_csi=not args.sem_trava, roi=args.roi)
    ok, quadro = cap.read()
    if not ok or quadro is None:
        cap.release()
        sys.exit(f'nenhum quadro de {args.camera}')
    if args.roi:
        quadro = vj.crop_roi(quadro, args.roi)

    print('=' * 66)
    print(f' Coleta — classe {args.classe.upper()} | lote {args.lote}')
    print('=' * 66)
    print(f'sessão : {sessao}')
    print(f'quadro : {quadro.shape[1]}x{quadro.shape[0]}')
    print(f'alvo   : {args.graos} grãos únicos')
    print(f'destino: {destino}')
    if args.sem_trava:
        print('\n*** AE/AWB AUTOMÁTICOS — este dado NÃO é padronizado. ***')
        print('*** Use só para ajustar o rig, nunca para treinar.       ***')
    print('\nEspalhe os grãos SEM SE ENCOSTAR. q encerra · p pausa\n')

    tracker = vj.IoUTracker(compensar=not args.parado)
    # por grão: guarda só os POR_GRAO recortes mais nítidos, em vez dos ~11
    melhores = defaultdict(list)     # track_id -> [(nitidez, recorte, caixa)]
    salvos = {}                      # track_id -> quantos já foram para o disco
    descartes = Counter()
    n_quadros = 0
    t0 = time.time()
    janela = 'coleta (q sai, p pausa)'
    if not args.sem_janela:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    pausado = False

    linhas_manifesto = []
    try:
        while len(melhores) < args.graos:
            if not pausado:
                ok, quadro = cap.read()
                if not ok:
                    print('fim da fonte.')
                    break
                if args.roi:
                    quadro = vj.crop_roi(quadro, args.roi)
                n_quadros += 1

                dets, vis = [], quadro.copy()
                for x1, y1, x2, y2, solidez in segmentar(quadro):
                    serve, motivo, nitidez = avaliar(quadro, (x1, y1, x2, y2), solidez)
                    if not serve:
                        descartes[motivo] += 1
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 160), 1)
                        continue
                    # o tracker espera (x1,y1,x2,y2,classe,conf); classe e conf
                    # não são usadas aqui, a classe vem da bandeja
                    dets.append((x1, y1, x2, y2, 0, nitidez))

                for tid, x1, y1, x2, y2, _cls, nitidez in tracker.update(dets):
                    corte = recortar(quadro, (x1, y1, x2, y2))
                    guardados = melhores[tid]
                    guardados.append((nitidez, corte, (x1, y1, x2, y2)))
                    guardados.sort(key=lambda g: -g[0])
                    del guardados[POR_GRAO:]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (90, 200, 90), 2)
                    cv2.putText(vis, str(tid), (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 200, 90), 1)

                hud = (f'{args.classe} | {len(melhores)}/{args.graos} graos | '
                       f'{n_quadros} quadros | descartes {sum(descartes.values())}')
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 4)
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

            if not args.sem_janela:
                cv2.imshow(janela, vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord('p'):
                    pausado = not pausado
    finally:
        cap.release()
        if not args.sem_janela:
            cv2.destroyAllWindows()

    # grava só no fim: assim um Ctrl-C no meio não deixa recorte órfão sem
    # linha no manifesto
    for tid, guardados in melhores.items():
        for i, (nitidez, corte, caixa) in enumerate(guardados):
            nome = f'{args.lote}_{sessao}_g{tid:05d}_{i}.jpg'
            cv2.imwrite(os.path.join(destino, nome), corte,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            linhas_manifesto.append({
                'arquivo': nome, 'classe': args.classe, 'lote': args.lote,
                'sessao': sessao, 'grao': f'{args.lote}_{sessao}_g{tid:05d}',
                'nitidez': round(nitidez, 1), 'caixa': [int(v) for v in caixa],
                'lado_px': int(max(caixa[2] - caixa[0], caixa[3] - caixa[1])),
                'px_por_mm': PX_POR_MM, 'roi': args.roi,
                'travas': not args.sem_trava,
                'quando': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
        salvos[tid] = len(guardados)

    os.makedirs(BRUTO, exist_ok=True)
    with open(MANIFESTO, 'a') as f:
        for linha in linhas_manifesto:
            f.write(json.dumps(linha, ensure_ascii=False) + '\n')

    dur = time.time() - t0
    print('\n' + '=' * 66)
    print(f'{len(melhores)} grãos únicos, {len(linhas_manifesto)} recortes, '
          f'{n_quadros} quadros em {dur:.0f}s')
    if descartes:
        tot_det = sum(descartes.values()) + sum(salvos.values())
        print(f'\ndescartados ({100*sum(descartes.values())/max(tot_det,1):.0f}% do detectado):')
        for motivo, n in descartes.most_common():
            print(f'  {motivo:12s} {n}')
        dica = {'na_borda': 'normal na esteira — o grão entra e sai pela borda',
                'borrado': 'foco ou exposição: veja calibrar_rig.py',
                'encostado': 'espalhe mais os grãos',
                'grande': 'grãos colados viram um blob só — espalhe mais',
                'pequeno': 'confira o px/mm com a régua'}
        pior = descartes.most_common(1)[0][0]
        if sum(descartes.values()) > sum(salvos.values()):
            print(f'\n  Mais descarte que aproveitamento. Maior causa: {pior}')
            print(f'  -> {dica.get(pior, "")}')
    if len(melhores) < args.graos:
        print(f'\nMeta era {args.graos} grãos. Rode de novo com outra bandeja '
              f'do mesmo lote para completar.')
    print(f'\nQuando tiver todas as classes:  python3 coletar_dataset.py --dividir')


# ---------------------------------------------------------------- divisão
def split_do_grao(grao):
    """Split determinístico a partir do ID do grão.

    Hash em vez de sorteio: rodar a divisão de novo dá o mesmo resultado, e
    capturas novas não remexem o que já estava dividido. E como a chave é o
    GRÃO, todos os recortes dele caem juntos — é o que impede o mesmo grão
    físico de aparecer no treino e na validação ao mesmo tempo.
    """
    h = int(hashlib.md5(grao.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return 'test'
    if h < TEST_FRAC + VAL_FRAC:
        return 'valid'
    return 'train'


def dividir(args):
    if not os.path.exists(MANIFESTO):
        sys.exit(f'nada capturado ainda (sem {MANIFESTO})')
    linhas = [json.loads(l) for l in open(MANIFESTO) if l.strip()]
    if not linhas:
        sys.exit('manifesto vazio')

    # uma captura sem travas não pode se misturar com o dataset padronizado:
    # são domínios diferentes, que é o gargalo estrutural do projeto inteiro
    sem_trava = [l for l in linhas if not l.get('travas', True)]
    if sem_trava and not args.incluir_sem_trava:
        print(f'IGNORANDO {len(sem_trava)} recortes capturados SEM as travas de '
              f'exposição.')
        print('  Eles são de outro domínio e misturá-los recria o domain shift.')
        print('  Use --incluir-sem-trava se souber o que está fazendo.\n')
        linhas = [l for l in linhas if l.get('travas', True)]

    shutil.rmtree(PRONTO, ignore_errors=True)
    for sp in ('train', 'valid', 'test'):
        for c in CLASSES:
            os.makedirs(os.path.join(PRONTO, sp, c), exist_ok=True)

    por_split = defaultdict(Counter)
    graos_split = defaultdict(set)
    faltando = 0
    for l in linhas:
        origem = os.path.join(BRUTO, l['classe'], l['arquivo'])
        if not os.path.exists(origem):
            faltando += 1
            continue
        sp = split_do_grao(l['grao'])
        destino = os.path.join(PRONTO, sp, l['classe'], l['arquivo'])
        try:
            os.link(origem, destino)      # hardlink: não duplica bytes
        except OSError:
            shutil.copy(origem, destino)
        por_split[sp][l['classe']] += 1
        graos_split[sp].add(l['grao'])

    # a garantia que justifica dividir por grão: nenhum grão em dois splits
    for a in ('train', 'valid', 'test'):
        for b in ('train', 'valid', 'test'):
            if a < b:
                comum = graos_split[a] & graos_split[b]
                assert not comum, f'VAZAMENTO: {len(comum)} grãos em {a} e {b}'

    total = sum(sum(c.values()) for c in por_split.values())
    print('=' * 66)
    print(' Dataset dividido')
    print('=' * 66)
    cab = f'{"classe":16s}' + ''.join(f'{s:>9s}' for s in ('train', 'valid', 'test')) + f'{"total":>9s}'
    print(cab)
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        print(f'{c:16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print('-' * len(cab))
    n = [sum(por_split[s].values()) for s in ('train', 'valid', 'test')]
    print(f'{"TOTAL":16s}' + ''.join(f'{v:9d}' for v in n) + f'{total:9d}')
    print(f'{"grãos únicos":16s}'
          + ''.join(f'{len(graos_split[s]):9d}' for s in ('train', 'valid', 'test'))
          + f'{sum(len(v) for v in graos_split.values()):9d}')
    if faltando:
        print(f'\n{faltando} linhas do manifesto sem arquivo no disco (ignoradas)')
    print('\nSem vazamento: nenhum grão físico aparece em dois splits.')

    vazias = [c for c in CLASSES if sum(por_split[s][c] for s in por_split) == 0]
    if vazias:
        print(f'\nATENÇÃO: sem nenhum recorte de {vazias}.')
        print('  Um modelo treinado assim nunca prevê essas classes.')

    _escrever_relatorio(por_split, graos_split, linhas, total)
    print(f'\npronto em: {PRONTO}')
    print('Aponte o notebook de treino para essa pasta (ela já tem o formato')
    print('train/valid/test por classe que o collect_base espera).')


def _escrever_relatorio(por_split, graos_split, linhas, total):
    lados = [l['lado_px'] for l in linhas if 'lado_px' in l]
    nitidez = [l['nitidez'] for l in linhas if 'nitidez' in l]
    lotes = sorted({l['lote'] for l in linhas})
    sessoes = sorted({l['sessao'] for l in linhas})
    md = [
        '# Dataset do rig — relatório',
        '',
        f'Gerado em {time.strftime("%Y-%m-%d %H:%M:%S")}.',
        '',
        '## Conteúdo', '',
        '| classe | train | valid | test | total |', '|---|---|---|---|---|',
    ]
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        md.append(f'| `{c}` | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |')
    md += [
        '',
        f'- **{total} recortes** de '
        f'**{sum(len(v) for v in graos_split.values())} grãos físicos únicos**',
        f'- lotes: {", ".join(lotes)}',
        f'- sessões de captura: {len(sessoes)}',
    ]
    if lados:
        md.append(f'- grão: {np.mean(lados):.0f} px em média '
                  f'(p5 {np.percentile(lados, 5):.0f}, p95 {np.percentile(lados, 95):.0f}) '
                  f'— alvo do rig {GRAO_MM * PX_POR_MM:.0f} px')
    if nitidez:
        md.append(f'- nitidez (variância do laplaciano): mediana '
                  f'{np.median(nitidez):.0f}, mínima {min(nitidez):.0f}')
    md += [
        '',
        '## Como ler estes números', '',
        'A contagem que importa é a de **grãos físicos únicos**, não a de',
        'recortes: cada grão contribui com até '
        f'{POR_GRAO} recortes do mesmo objeto, e eles',
        'não são exemplos independentes.',
        '',
        'A divisão é **por grão**, não por imagem — todos os recortes de um grão',
        'caem no mesmo split. Dividir por imagem colocaria o mesmo grão físico no',
        'treino e na validação, e a validação passaria a medir memorização.',
        '',
        '## Procedência', '',
        'Caixa por limiarização de Otsu (o rig controla o fundo), classe pela',
        'bandeja que estava passando. **Nenhum rótulo veio de modelo** — isso',
        'evita assar os erros do modelo atual dentro do dataset do próximo.',
        '',
        'Só entram capturas com as travas de exposição LIGADAS. Captura em',
        'automático é outro domínio e fica de fora por padrão.',
        '',
        f'Escala do rig no momento da captura: **{PX_POR_MM} px/mm**. Se a',
        'geometria mudar, isto vira um dataset de outra versão do padrão — ver',
        '`PADRAO_CAPTURA.md`.',
    ]
    caminho = os.path.join(PRONTO, 'RELATORIO.md')
    open(caminho, 'w').write('\n'.join(md) + '\n')

    with open(os.path.join(PRONTO, 'dataset.yaml'), 'w') as f:
        f.write(f'# Vígil.ia — dataset do rig ({time.strftime("%Y-%m-%d")})\n')
        f.write(f'path: {PRONTO}\ntrain: train\nval: valid\ntest: test\n\n')
        f.write(f'nc: {len(CLASSES)}\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write(f'  {i}: {c}\n')


# ---------------------------------------------------------------- status
def status(args):
    print('=' * 66)
    print(' Status da coleta')
    print('=' * 66)
    if not os.path.exists(MANIFESTO):
        print('nada capturado ainda.')
        print(f'\nComece por:  python3 coletar_dataset.py --classe intact --lote L001')
        return
    linhas = [json.loads(l) for l in open(MANIFESTO) if l.strip()]
    graos = defaultdict(set)
    for l in linhas:
        graos[l['classe']].add(l['grao'])
    print(f'{"classe":16s} {"grãos":>8s} {"recortes":>10s}   meta 120')
    for c in CLASSES:
        n = len(graos[c])
        rec = sum(1 for l in linhas if l['classe'] == c)
        barra = '#' * min(30, int(30 * n / 120))
        print(f'{c:16s} {n:8d} {rec:10d}   {barra}')
    falta = [c for c in CLASSES if len(graos[c]) < 120]
    if falta:
        print(f'\nfalta capturar: {", ".join(falta)}')
    else:
        print('\ntodas as classes com 120+ grãos — pode dividir.')


def main():
    ap = argparse.ArgumentParser(description='Coleta de dataset no rig')
    ap.add_argument('--classe', help=f'classe da bandeja ({"/".join(CLASSES)})')
    ap.add_argument('--lote', default='L001', help='identificador do lote de soja')
    ap.add_argument('--graos', type=int, default=120, help='meta de grãos únicos')
    ap.add_argument('--camera', default='csi')
    ap.add_argument('--roi', type=int, default=0, metavar='PX',
                    help='recorte 1:1 do sensor cheio (rig: 704)')
    ap.add_argument('--sem-trava', action='store_true',
                    help='AE/AWB automáticos — NÃO use para dataset')
    ap.add_argument('--parado', action='store_true',
                    help='bandeja estática (desliga a compensação de movimento)')
    ap.add_argument('--sem-janela', action='store_true')
    ap.add_argument('--dividir', action='store_true',
                    help='divide o capturado em train/valid/test')
    ap.add_argument('--incluir-sem-trava', action='store_true',
                    help='na divisão, inclui capturas não padronizadas')
    ap.add_argument('--status', action='store_true', help='quanto já foi capturado')
    args = ap.parse_args()

    if args.status:
        return status(args)
    if args.dividir:
        return dividir(args)
    if not args.classe:
        ap.error('diga a classe da bandeja: --classe intact  '
                 '(ou use --status / --dividir)')
    capturar(args)


if __name__ == '__main__':
    main()
