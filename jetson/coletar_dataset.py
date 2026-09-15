#!/usr/bin/env python3
"""Vígil.ia — entra soja multi-grão, sai dataset de DETECÇÃO para treinar do zero.

    # 1. capturar: soja misturada, como ela cai na esteira
    python3 coletar_dataset.py --engine soja_rfdetr_small_CAMPEAO_fp16.engine \
        --camera csi --roi 704 --tiles 2 --lote L001

    # 2. CORRIGIR À MÃO: abra dataset/revisar/ e arraste o que estiver na
    #    pasta errada. Uma correção vale para TODOS os quadros daquele grão.

    # 3. exportar em YOLO + COCO, dividido em train/valid/test
    python3 coletar_dataset.py --exportar

O que sai daqui treina um **detector do zero**, não um classificador:

    dataset/pronto/
        train/images/*.jpg   train/labels/*.txt      <- YOLO
        train/_annotations.coco.json                 <- COCO (rfdetr, DETR…)
        dataset.yaml

Por que o quadro inteiro, e não só o recorte
--------------------------------------------
Recorte solto treina classificador. Detector precisa aprender *onde* o grão
está, quantos existem no quadro e como eles se encostam — e isso só está na
cena inteira. O pipeline antigo montava cenas sintéticas justamente porque só
tinha foto de um grão por imagem; o rig entrega cena densa **real**, com fundo
real e oclusão real. Guardar o quadro é trocar a imitação pelo original.

Como a correção se propaga
--------------------------
O rastreamento dá um ID a cada grão físico. Você corrige **um** recorte daquele
grão, e a correção vale para todas as caixas dele, em todos os quadros salvos.
Mover um arquivo pode acertar dez anotações — é o mesmo princípio do
`model/aprendizado_ativo.ipynb`.

Divisão sem vazamento
---------------------
Quadros seguidos da esteira são quase idênticos: o grão anda ~4 mm entre
varreduras. Jogar o quadro N no treino e o N+1 na validação é vazamento
disfarçado. Por isso a captura é cortada em **blocos** de tempo, os blocos
inteiros vão para um split, e há uma **banda de guarda** entre eles maior que a
travessia de um grão — assim nenhum grão físico aparece em dois splits.
"""
import argparse
import csv
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
_spec = importlib.util.spec_from_file_location(
    'vigil_jetson', os.path.join(_AQUI, 'vigil_jetson.py'))
vj = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vj)

CLASSES = vj.NAMES
DESCARTE = 'descartar'
RAIZ = os.path.join(_AQUI, 'dataset')
QUADROS = os.path.join(RAIZ, 'quadros')     # cena inteira + caixas
REVISAR = os.path.join(RAIZ, 'revisar')     # recortes, para a revisão humana
PRONTO = os.path.join(RAIZ, 'pronto')       # dataset exportado

NITIDEZ_MIN = 50.0      # variância do laplaciano no recorte
MARGEM_BORDA = 3        # px: grão a menos que isso da borda pode estar cortado
CONTEXTO = 0.15         # borda extra no recorte de revisão
VAL_FRAC, TEST_FRAC = 0.15, 0.10


def recortar(frame, caixa):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caixa
    pad = int(CONTEXTO * max(x2 - x1, y2 - y1))
    return frame[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)].copy()


def nitidez_de(recorte):
    if recorte.size == 0:
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY),
                               cv2.CV_64F).var())


# ---------------------------------------------------------------- captura
def capturar(args):
    modelo = vj.RFDetrTRT(args.engine, conf=args.conf, offset=args.class_offset)
    tracker = vj.IoUTracker(compensar=not args.parado)
    lado = args.roi or modelo.W
    sobrepor = int(lado * 0.2) if args.tiles > 1 else 0
    recorte_hw = ((lado * args.tiles - sobrepor * (args.tiles - 1), lado)
                  if args.roi else None)

    cap = (cv2.VideoCapture(args.source) if args.source
           else vj.abrir_camera(args.camera, travar_csi=not args.sem_trava,
                                roi=args.roi, recorte=recorte_hw))
    ok, quadro = cap.read()
    if not ok or quadro is None:
        cap.release()
        sys.exit(f'nenhum quadro de {args.source or args.camera}')

    sessao = time.strftime('%Y%m%d-%H%M%S')
    dir_q = os.path.join(QUADROS, sessao)
    os.makedirs(dir_q, exist_ok=True)
    for c in CLASSES + [DESCARTE]:
        os.makedirs(os.path.join(REVISAR, c), exist_ok=True)

    print('=' * 70)
    print(' Coleta multi-grão — dataset de DETECÇÃO')
    print('=' * 70)
    print(f'sessão  : {sessao} | lote {args.lote}')
    print(f'modelo  : {os.path.basename(args.engine)} (propõe as caixas e as classes)')
    if args.tiles > 1:
        print(f'recortes: {args.tiles} x {lado}px, sobreposição {sobrepor}px')
    print(f'guarda   : 1 quadro a cada {args.passo} varreduras '
          f'(quadros seguidos são quase idênticos)')
    print(f'blocos   : {args.bloco}s, com {args.guarda}s de banda de guarda')
    print(f'alvo     : {args.graos} grãos únicos')
    if args.sem_trava:
        print('\n*** AE/AWB AUTOMÁTICOS — dado NÃO padronizado, não treine com ele.')
    print('\nEspalhe os grãos SEM SE ENCOSTAR. q encerra · p pausa\n')

    votos = defaultdict(Counter)
    visto = Counter()
    travado = {}
    melhor = {}                     # grao -> (nitidez, recorte)
    quadros_salvos = []             # {'arquivo','bloco','t','caixas':[...]}
    descartes = Counter()
    n_varreduras = 0
    t0 = time.time()
    janela = 'coleta (q sai, p pausa)'
    if not args.sem_janela:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    pausado = False
    vis = quadro

    try:
        while len(travado) < args.graos:
            if not pausado:
                ok, quadro = cap.read()
                if not ok:
                    print('fim da fonte.')
                    break
                if args.roi and args.tiles <= 1:
                    quadro = vj.crop_roi(quadro, args.roi)
                n_varreduras += 1
                agora = time.time() - t0

                if args.tiles > 1:
                    dets = []
                    for tile, dx, dy in vj.crop_tiles(quadro, lado, args.tiles, sobrepor):
                        for x1, y1, x2, y2, ci, cf in modelo(tile):
                            dets.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, ci, cf))
                    dets = vj.nms_global(dets)
                else:
                    dets = modelo(quadro)

                vis = quadro.copy()
                h, w = quadro.shape[:2]
                caixas_do_quadro = []
                for tid, x1, y1, x2, y2, ci, cf in tracker.update(dets):
                    grao = f'{args.lote}_{sessao}_g{tid:05d}'
                    nome_cls = CLASSES[ci] if 0 <= ci < len(CLASSES) else 'intact'
                    votos[grao][nome_cls] += cf
                    visto[grao] += 1

                    na_borda = (x1 <= MARGEM_BORDA or y1 <= MARGEM_BORDA
                                or x2 >= w - MARGEM_BORDA or y2 >= h - MARGEM_BORDA)
                    if na_borda:
                        # grão cortado não vira anotação: ensinaria o detector a
                        # chamar meio grão de grão inteiro
                        descartes['na_borda'] += 1
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 160), 1)
                        continue
                    caixas_do_quadro.append({'grao': grao,
                                             'caixa': [int(x1), int(y1), int(x2), int(y2)]})

                    corte = recortar(quadro, (x1, y1, x2, y2))
                    nit = nitidez_de(corte)
                    if nit >= NITIDEZ_MIN and (grao not in melhor or nit > melhor[grao][0]):
                        melhor[grao] = (nit, corte)
                    if grao not in travado and visto[grao] >= vj.LOCK_MIN_FRAMES:
                        travado[grao] = vj.veredito(votos[grao])
                    cls = travado.get(grao)
                    cor = vj.COLORS[cls] if cls else (160, 160, 160)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)

                # guarda a CENA a cada `passo` varreduras: quadros consecutivos
                # são quase o mesmo dado, e salvar todos só enche o disco
                if caixas_do_quadro and n_varreduras % args.passo == 0:
                    bloco = int(agora // args.bloco)
                    nome = f'{args.lote}_{sessao}_b{bloco:03d}_{n_varreduras:06d}.jpg'
                    cv2.imwrite(os.path.join(dir_q, nome), quadro,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    quadros_salvos.append({'arquivo': nome, 'bloco': bloco,
                                           't': round(agora, 2),
                                           'largura': w, 'altura': h,
                                           'caixas': caixas_do_quadro})

                d = Counter(travado.values())
                hud = (f'{len(travado)}/{args.graos} graos | {len(quadros_salvos)} quadros | '
                       + '  '.join(f'{vj.PT_LABEL[c][:5]} {d[c]}' for c in CLASSES if d[c]))
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
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

    # um recorte por grão, para a revisão. A classe proposta vai no NOME: depois
    # que você mover o arquivo, o nome ainda diz o que o modelo tinha achado.
    salvos = Counter()
    for grao, classe in sorted(travado.items()):
        if grao not in melhor:
            continue
        cv2.imwrite(os.path.join(REVISAR, classe, f'{grao}__{classe}.jpg'),
                    melhor[grao][1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        salvos[classe] += 1

    sessao_meta = {
        'sessao': sessao, 'lote': args.lote, 'engine': os.path.basename(args.engine),
        'travas': not args.sem_trava, 'roi': args.roi, 'tiles': args.tiles,
        'passo': args.passo, 'bloco_s': args.bloco, 'guarda_s': args.guarda,
        'quando': time.strftime('%Y-%m-%d %H:%M:%S'),
        'quadros': quadros_salvos,
        'graos': {g: {'classe_prevista': c, 'n_varreduras': visto[g],
                      'confianca': round(votos[g][c] / max(sum(votos[g].values()), 1e-9), 3)}
                  for g, c in travado.items()},
    }
    json.dump(sessao_meta, open(os.path.join(dir_q, 'sessao.json'), 'w'),
              ensure_ascii=False, indent=1)
    with open(os.path.join(dir_q, 'revisao.csv'), 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['grao', 'classe_prevista', 'confianca', 'n_varreduras'])
        for g, c in sorted(travado.items()):
            w_.writerow([g, c, sessao_meta['graos'][g]['confianca'], visto[g]])

    n_caixas = sum(len(q['caixas']) for q in quadros_salvos)
    print('\n' + '=' * 70)
    print(f'{len(travado)} grãos | {len(quadros_salvos)} quadros | {n_caixas} caixas '
          f'| {n_varreduras} varreduras em {time.time()-t0:.0f}s')
    print('\nComo o MODELO separou (proposta, não verdade):')
    for c in CLASSES:
        if salvos[c]:
            print(f'  {c:14s} {salvos[c]:5d}')
    if descartes:
        print(f'\ncaixas descartadas: {dict(descartes)}')
    print('\n' + '-' * 70)
    print('AGORA A PARTE MANUAL:')
    print(f'  1. abra  {REVISAR}')
    print('  2. ligue as miniaturas grandes no gerenciador de arquivos')
    print('  3. arraste o que estiver na pasta errada')
    print(f'  4. o que não for grão, ou estiver em dúvida -> {DESCARTE}/')
    print('\n  Você corrige UM recorte por grão, e a correção vale para todas as')
    print('  caixas dele em todos os quadros. Um arquivo movido pode acertar')
    print('  dezenas de anotações.')
    print('\n  Depois:  python3 coletar_dataset.py --exportar')
    print('-' * 70)


# ---------------------------------------------------------------- revisão
def ler_revisao():
    """Estado ATUAL das pastas: onde você deixou o arquivo é o rótulo."""
    mapa = {}
    for pasta in CLASSES + [DESCARTE]:
        d = os.path.join(REVISAR, pasta)
        if not os.path.isdir(d):
            continue
        for nome in sorted(os.listdir(d)):
            if not nome.lower().endswith('.jpg'):
                continue
            grao, _, prevista = nome[:-4].rpartition('__')
            if grao:
                mapa[grao] = {'corrigida': pasta, 'prevista': prevista}
    return mapa


def _sessoes():
    if not os.path.isdir(QUADROS):
        return []
    saida = []
    for d in sorted(os.listdir(QUADROS)):
        p = os.path.join(QUADROS, d, 'sessao.json')
        if os.path.exists(p):
            saida.append((os.path.join(QUADROS, d), json.load(open(p))))
    return saida


def split_do_bloco(chave):
    """Split determinístico por BLOCO de tempo, não por quadro.

    Quadros seguidos da esteira são quase idênticos — o grão anda ~4 mm entre
    varreduras. Dividir por quadro poria praticamente a mesma imagem no treino e
    na validação. Blocos inteiros vão juntos, e a banda de guarda garante que
    nenhum grão atravesse a fronteira.
    """
    h = int(hashlib.md5(chave.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return 'test'
    if h < TEST_FRAC + VAL_FRAC:
        return 'valid'
    return 'train'


# ---------------------------------------------------------------- exportação
def exportar(args):
    sessoes = _sessoes()
    if not sessoes:
        sys.exit(f'nada capturado ainda (sem sessões em {QUADROS})')
    revisao = ler_revisao()
    if not revisao:
        sys.exit(f'nada em {REVISAR} — capture primeiro.')

    shutil.rmtree(PRONTO, ignore_errors=True)
    for sp in ('train', 'valid', 'test'):
        os.makedirs(os.path.join(PRONTO, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(PRONTO, sp, 'labels'), exist_ok=True)

    coco = {sp: {'images': [], 'annotations': [], 'categories':
                 [{'id': i + 1, 'name': c, 'supercategory': 'soja'}
                  for i, c in enumerate(CLASSES)]}
            for sp in ('train', 'valid', 'test')}
    ids = {sp: [0, 0] for sp in coco}          # [img_id, ann_id]
    por_split = defaultdict(Counter)
    quadros_split = Counter()
    graos_split = defaultdict(set)
    sem_revisao = set()
    descartadas = 0
    na_guarda = 0
    pulados_sem_trava = 0

    for dir_q, meta in sessoes:
        if not meta.get('travas', True) and not args.incluir_sem_trava:
            pulados_sem_trava += len(meta['quadros'])
            continue
        bloco_s, guarda_s = meta.get('bloco_s', 20), meta.get('guarda_s', 3)
        for q in meta['quadros']:
            # banda de guarda: quadro perto da fronteira entre blocos é
            # descartado, porque um grão pode aparecer dos dois lados
            resto = q['t'] % bloco_s
            if resto < guarda_s / 2 or resto > bloco_s - guarda_s / 2:
                na_guarda += 1
                continue
            chave = f"{meta['sessao']}_b{q['bloco']:03d}"
            sp = split_do_bloco(chave)

            linhas, anns = [], []
            for cx in q['caixas']:
                r = revisao.get(cx['grao'])
                if r is None:
                    sem_revisao.add(cx['grao'])
                    continue
                if r['corrigida'] == DESCARTE:
                    descartadas += 1
                    continue
                ci = CLASSES.index(r['corrigida'])
                x1, y1, x2, y2 = cx['caixa']
                W, H = q['largura'], q['altura']
                linhas.append(f'{ci} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} '
                              f'{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}')
                anns.append((ci, x1, y1, x2 - x1, y2 - y1))
                por_split[sp][r['corrigida']] += 1
                graos_split[sp].add(cx['grao'])

            if not linhas:            # quadro que ficou sem nenhuma caixa válida
                continue
            origem = os.path.join(dir_q, q['arquivo'])
            if not os.path.exists(origem):
                continue
            destino = os.path.join(PRONTO, sp, 'images', q['arquivo'])
            try:
                os.link(origem, destino)
            except OSError:
                shutil.copy(origem, destino)
            open(os.path.join(PRONTO, sp, 'labels',
                              q['arquivo'][:-4] + '.txt'), 'w').write('\n'.join(linhas))

            iid, aid = ids[sp]
            coco[sp]['images'].append({'id': iid, 'file_name': q['arquivo'],
                                       'width': q['largura'], 'height': q['altura']})
            for ci, x, y, bw, bh in anns:
                coco[sp]['annotations'].append(
                    {'id': aid, 'image_id': iid, 'category_id': ci + 1,
                     'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
                aid += 1
            ids[sp] = [iid + 1, aid]
            quadros_split[sp] += 1

    for sp in coco:
        json.dump(coco[sp], open(os.path.join(PRONTO, sp, '_annotations.coco.json'), 'w'))

    # a garantia que justifica dividir por bloco
    for a in ('train', 'valid', 'test'):
        for b in ('train', 'valid', 'test'):
            if a < b:
                comum = graos_split[a] & graos_split[b]
                assert not comum, f'VAZAMENTO: {len(comum)} grãos em {a} e {b}'

    # --- acurácia do modelo, medida pelas correções ---
    com_prev = [r for r in revisao.values() if r['prevista'] in CLASSES]
    acertos = sum(1 for r in com_prev if r['prevista'] == r['corrigida'])
    confusao = Counter((r['prevista'], r['corrigida'])
                       for r in com_prev if r['prevista'] != r['corrigida'])

    cab = f'{"classe":16s}' + ''.join(f'{s:>9s}' for s in ('train', 'valid', 'test')) + f'{"total":>9s}'
    print('=' * 70)
    print(' Dataset de detecção exportado')
    print('=' * 70)
    print(cab)
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        print(f'{c:16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print('-' * len(cab))
    n = [sum(por_split[s].values()) for s in ('train', 'valid', 'test')]
    print(f'{"CAIXAS":16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print(f'{"quadros":16s}'
          + ''.join(f'{quadros_split[s]:9d}' for s in ('train', 'valid', 'test'))
          + f'{sum(quadros_split.values()):9d}')
    print(f'{"grãos únicos":16s}'
          + ''.join(f'{len(graos_split[s]):9d}' for s in ('train', 'valid', 'test'))
          + f'{sum(len(v) for v in graos_split.values()):9d}')
    if descartadas:
        print(f'\n{descartadas} caixas de grãos que você mandou para {DESCARTE}/')
    if na_guarda:
        print(f'{na_guarda} quadros na banda de guarda entre blocos (descartados '
              f'para não vazar grão entre splits)')
    if pulados_sem_trava:
        print(f'{pulados_sem_trava} quadros de sessões SEM travas de exposição '
              f'(outro domínio; --incluir-sem-trava para forçar)')
    if sem_revisao:
        print(f'{len(sem_revisao)} grãos sem recorte na revisão (não travaram a '
              f'classe) — as caixas deles ficaram de fora')
    print('\nSem vazamento: nenhum grão físico aparece em dois splits.')

    vazias = [c for c in CLASSES if sum(por_split[s][c] for s in por_split) == 0]
    if vazias:
        print(f'\nATENÇÃO: nenhuma caixa de {vazias}.')
        print('  Um detector treinado assim nunca prevê essas classes.')

    if com_prev:
        print('\n' + '=' * 70)
        print(' Quanto o modelo atual acertou — medido pelas SUAS correções')
        print('=' * 70)
        print(f'  {acertos}/{len(com_prev)} grãos ficaram onde o modelo pôs '
              f'= {100*acertos/len(com_prev):.1f}%')
        if confusao:
            print('\n  onde errou (modelo -> você corrigiu para):')
            for (p, r), k in confusao.most_common(8):
                print(f'    {p:14s} -> {r:14s} {k:4d}')
        print('\n  É a primeira acurácia do projeto medida NO RIG contra rótulo')
        print('  humano — e é a linha de base que o modelo novo precisa bater.')

    _relatorio(por_split, quadros_split, graos_split, com_prev, acertos, confusao,
               descartadas, na_guarda)
    _yaml()
    print(f'\npronto em: {PRONTO}')
    print('  YOLO : train/images + train/labels')
    print('  COCO : train/_annotations.coco.json')
    print('  Os dois formatos saem do mesmo dado — treine com o que preferir.')


def _yaml():
    with open(os.path.join(PRONTO, 'dataset.yaml'), 'w') as f:
        f.write(f'# Vígil.ia — dataset de detecção do rig ({time.strftime("%Y-%m-%d")})\n')
        f.write(f'path: {PRONTO}\ntrain: train/images\nval: valid/images\n'
                f'test: test/images\n\nnc: {len(CLASSES)}\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write(f'  {i}: {c}\n')


def _relatorio(por_split, quadros_split, graos_split, com_prev, acertos, confusao,
               descartadas, na_guarda):
    md = ['# Dataset de detecção do rig — relatório', '',
          f'Gerado em {time.strftime("%Y-%m-%d %H:%M:%S")}.', '',
          '## Conteúdo', '', '| classe | train | valid | test | total |',
          '|---|---|---|---|---|']
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        md.append(f'| `{c}` | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |')
    md += ['',
           f'- **{sum(sum(v.values()) for v in por_split.values())} caixas** em '
           f'**{sum(quadros_split.values())} quadros**',
           f'- **{sum(len(v) for v in graos_split.values())} grãos físicos únicos**',
           f'- {descartadas} caixas descartadas na revisão, '
           f'{na_guarda} quadros na banda de guarda']
    if com_prev:
        md += ['', '## Acurácia do modelo atual no rig', '',
               f'**{acertos}/{len(com_prev)} = {100*acertos/len(com_prev):.1f}%** '
               'dos grãos ficaram onde o modelo os colocou.', '',
               'Sai da revisão manual: cada grão não movido é um acerto. É medido',
               '**no rig**, contra rótulo humano — e é a linha de base que o modelo',
               'treinado do zero precisa bater para justificar a troca.']
        if confusao:
            md += ['', '| modelo previu | corrigido para | n |', '|---|---|---|']
            for (p, r), k in confusao.most_common(12):
                md.append(f'| `{p}` | `{r}` | {k} |')
    md += ['', '## Formato', '',
           'Sai nos dois formatos, do mesmo dado:', '',
           '- **YOLO**: `<split>/images/*.jpg` + `<split>/labels/*.txt`',
           '- **COCO**: `<split>/_annotations.coco.json`', '',
           'É dataset de **detecção**, com a cena inteira e todas as caixas — não',
           'recorte solto. Detector precisa aprender onde o grão está, quantos há',
           'no quadro e como eles se encostam, e isso só existe na cena.', '',
           '## Sem vazamento', '',
           'Quadros consecutivos da esteira são quase idênticos. A divisão é por',
           '**bloco de tempo**, com banda de guarda entre blocos maior que a',
           'travessia de um grão — nenhum grão físico aparece em dois splits, e a',
           'exportação verifica isso antes de terminar.', '',
           '## Procedência', '',
           'Caixas do detector atual em cena multi-grão, classe por voto temporal',
           'e **corrigida à mão**. O rótulo final é a pasta em que o grão foi',
           'deixado na revisão; uma correção vale para todas as caixas daquele',
           'grão, em todos os quadros.', '',
           'Só entram sessões com as travas de exposição ligadas.']
    open(os.path.join(PRONTO, 'RELATORIO.md'), 'w').write('\n'.join(md) + '\n')


# ---------------------------------------------------------------- status
def status(args):
    print('=' * 70)
    print(' Status')
    print('=' * 70)
    sessoes = _sessoes()
    revisao = ler_revisao()
    if not sessoes:
        print(f'nada capturado ainda.')
        return
    nq = sum(len(m['quadros']) for _, m in sessoes)
    nc = sum(len(q['caixas']) for _, m in sessoes for q in m['quadros'])
    print(f'{len(sessoes)} sessões | {nq} quadros | {nc} caixas')
    por_pasta = Counter(r['corrigida'] for r in revisao.values())
    print(f'\n{"pasta":16s} {"grãos":>8s}   meta 120')
    for c in CLASSES:
        barra = '#' * min(30, int(30 * por_pasta[c] / 120))
        print(f'{c:16s} {por_pasta[c]:8d}   {barra}')
    print(f'{DESCARTE:16s} {por_pasta[DESCARTE]:8d}')
    movidos = sum(1 for r in revisao.values()
                  if r['prevista'] in CLASSES and r['prevista'] != r['corrigida'])
    print(f'\n{movidos} de {len(revisao)} corrigidos por você '
          f'({100*movidos/max(len(revisao),1):.0f}%)')
    falta = [c for c in CLASSES if por_pasta[c] < 120]
    if falta:
        print(f'falta capturar mais: {", ".join(falta)}')


def main():
    ap = argparse.ArgumentParser(
        description='Coleta multi-grão -> dataset de detecção para treinar do zero')
    ap.add_argument('--engine', default=None)
    ap.add_argument('--lote', default='L001')
    ap.add_argument('--graos', type=int, default=400, help='meta de grãos por sessão')
    ap.add_argument('--camera', default='csi')
    ap.add_argument('--source', default=None, help='vídeo gravado em vez da câmera')
    ap.add_argument('--roi', type=int, default=0, metavar='PX')
    ap.add_argument('--tiles', type=int, default=1, metavar='N')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='confiança mínima; baixa de propósito — você filtra na revisão')
    ap.add_argument('--passo', type=int, default=6,
                    help='guarda 1 quadro a cada N varreduras (padrão 6)')
    ap.add_argument('--bloco', type=float, default=20,
                    help='tamanho do bloco de tempo, em segundos')
    ap.add_argument('--guarda', type=float, default=3,
                    help='banda de guarda entre blocos, em segundos')
    ap.add_argument('--class-offset', type=int, default=None)
    ap.add_argument('--sem-trava', action='store_true')
    ap.add_argument('--parado', action='store_true')
    ap.add_argument('--sem-janela', action='store_true')
    ap.add_argument('--exportar', action='store_true')
    ap.add_argument('--incluir-sem-trava', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    if args.status:
        return status(args)
    if args.exportar:
        return exportar(args)
    if not args.engine:
        cands = sorted(f for f in os.listdir('.') if f.endswith('.engine'))
        if len(cands) != 1:
            ap.error('diga qual engine usar: --engine X.engine'
                     + (f'\n  disponíveis: {", ".join(cands)}' if cands else ''))
        args.engine = cands[0]
    capturar(args)


if __name__ == '__main__':
    main()
