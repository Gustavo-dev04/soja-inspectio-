#!/usr/bin/env python3
"""Vígil.ia — entra soja multi-grão, sai dataset pré-separado para você corrigir.

    # 1. capturar: soja misturada, como ela cai na esteira
    python3 coletar_dataset.py --engine soja_rfdetr_small_CAMPEAO_fp16.engine \
        --camera csi --roi 704 --tiles 2

    # 2. CORRIGIR À MÃO: abra dataset/revisar/ no gerenciador de arquivos e
    #    arraste o que estiver na pasta errada. Só isso.

    # 3. dividir em train/valid/test
    python3 coletar_dataset.py --dividir

O fluxo é o do `model/aprendizado_ativo.ipynb`, trazido para o rig: **o modelo
propõe, você dispõe**. Ele detecta os grãos no quadro multi-grão, rastreia,
fecha a classe por voto (as mesmas regras do app ao vivo), recorta cada grão
pela caixa e joga o recorte na pasta da classe que ele acha que é.

    dataset/revisar/
        broken/  immature/  intact/  skin-damaged/  spotted/
        descartar/     <- não é grão, está cortado, ou você não tem certeza

Corrigir é **arrastar arquivo entre pastas**, não editar planilha. Num
gerenciador com miniaturas dá para varrer centenas de grãos em minutos, porque
a maioria já está no lugar certo.

E aí vem o que isso rende de graça: a diferença entre onde o modelo pôs e onde
você moveu **é a acurácia do modelo no rig**. É a medição que a documentação diz
não existir, e ela cai no colo como subproduto da anotação.

Duas coisas que o script protege sozinho:

**Um grão físico aparece em ~11 quadros.** Salvar todos infla o dataset com
cópias quase iguais e, pior, uma divisão por IMAGEM colocaria o mesmo grão no
treino e na validação — a validação passaria a medir memorização. Aqui o
rastreamento agrupa os quadros de cada grão e a divisão é **por grão**.

**Recorte cortado na borda ou borrado vira exemplo errado com rótulo
confiante.** Os dois são barrados antes de chegar na sua revisão.
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
REVISAR = os.path.join(RAIZ, 'revisar')
PRONTO = os.path.join(RAIZ, 'pronto')
MANIFESTO = os.path.join(RAIZ, 'manifesto.jsonl')

# --- filtros: recorte ruim não deve nem chegar na sua revisão ---
NITIDEZ_MIN = 50.0      # variância do laplaciano
MARGEM_BORDA = 3        # px da borda do quadro: menos que isso pode estar cortado
POR_GRAO = 1            # recortes por grão físico (o mais nítido)
CONTEXTO = 0.15         # borda extra no recorte, em fração do lado

VAL_FRAC, TEST_FRAC = 0.15, 0.10


def recortar(frame, caixa):
    """Recorta a caixa com uma borda de contexto, sem sair do quadro."""
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
    sobrepor = int((args.roi or modelo.W) * 0.2) if args.tiles > 1 else 0
    _lado = args.roi or modelo.W
    recorte_hw = ((_lado * args.tiles - sobrepor * (args.tiles - 1), _lado)
                  if args.roi else None)

    cap = vj.abrir_camera(args.camera, travar_csi=not args.sem_trava,
                          roi=args.roi, recorte=recorte_hw) if not args.source \
        else cv2.VideoCapture(args.source)
    ok, quadro = cap.read()
    if not ok or quadro is None:
        cap.release()
        sys.exit(f'nenhum quadro de {args.source or args.camera}')

    for c in CLASSES + [DESCARTE]:
        os.makedirs(os.path.join(REVISAR, c), exist_ok=True)
    sessao = time.strftime('%Y%m%d-%H%M%S')

    print('=' * 68)
    print(' Coleta multi-grão — o modelo propõe, você corrige')
    print('=' * 68)
    print(f'sessão : {sessao} | lote {args.lote}')
    print(f'modelo : {os.path.basename(args.engine)} (entrada {modelo.W}px)')
    if args.tiles > 1:
        print(f'recortes: {args.tiles} x {_lado}px, sobreposição {sobrepor}px')
    print(f'alvo   : {args.graos} grãos únicos')
    if args.sem_trava:
        print('\n*** AE/AWB AUTOMÁTICOS — este dado NÃO é padronizado.')
        print('*** Serve para ajustar o rig, não para treinar.')
    print('\nEspalhe os grãos SEM SE ENCOSTAR. q encerra · p pausa\n')

    # mesma máquina de voto do app ao vivo: a classe proposta aqui é exatamente
    # a que o vigil_jetson daria, então a taxa de correção mede o modelo de
    # verdade, não uma variante dele
    votos = defaultdict(Counter)
    visto = Counter()
    travado = {}
    melhor = {}                      # tid -> (nitidez, recorte, caixa)
    descartes = Counter()
    n_quadros = 0
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
                n_quadros += 1

                if args.tiles > 1:
                    dets = []
                    for tile, dx, dy in vj.crop_tiles(quadro, _lado, args.tiles, sobrepor):
                        for x1, y1, x2, y2, ci, cf in modelo(tile):
                            dets.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, ci, cf))
                    dets = vj.nms_global(dets)
                else:
                    dets = modelo(quadro)

                vis = quadro.copy()
                h, w = quadro.shape[:2]
                for tid, x1, y1, x2, y2, ci, cf in tracker.update(dets):
                    nome = CLASSES[ci] if 0 <= ci < len(CLASSES) else 'intact'
                    votos[tid][nome] += cf
                    visto[tid] += 1

                    # grão tocando a borda está cortado: não vira exemplo
                    if (x1 <= MARGEM_BORDA or y1 <= MARGEM_BORDA
                            or x2 >= w - MARGEM_BORDA or y2 >= h - MARGEM_BORDA):
                        descartes['na_borda'] += 1
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 160), 1)
                        continue

                    corte = recortar(quadro, (x1, y1, x2, y2))
                    nit = nitidez_de(corte)
                    if nit < NITIDEZ_MIN:
                        descartes['borrado'] += 1
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 120, 160), 1)
                        continue
                    if tid not in melhor or nit > melhor[tid][0]:
                        melhor[tid] = (nit, corte, (x1, y1, x2, y2))

                    if tid not in travado and visto[tid] >= vj.LOCK_MIN_FRAMES:
                        travado[tid] = vj.veredito(votos[tid])
                    cls = travado.get(tid)
                    cor = vj.COLORS[cls] if cls else (160, 160, 160)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), cor, 2)
                    if cls:
                        cv2.putText(vis, f'{tid}:{vj.PT_LABEL[cls][:6]}',
                                    (x1, max(12, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)

                d = Counter(travado.values())
                hud = (f'{len(travado)}/{args.graos} graos  |  '
                       + '  '.join(f'{vj.PT_LABEL[c][:5]} {d[c]}' for c in CLASSES if d[c])
                       + f'  |  {n_quadros}q')
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(vis, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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

    # grava no fim: Ctrl-C no meio não deixa recorte órfão sem linha no manifesto
    linhas, salvos = [], Counter()
    for tid, classe in sorted(travado.items()):
        if tid not in melhor:
            continue
        nit, corte, caixa = melhor[tid]
        grao = f'{args.lote}_{sessao}_g{tid:05d}'
        # a classe entra NO NOME do arquivo além da pasta: depois que você mover,
        # o nome ainda diz o que o modelo tinha proposto — é assim que dá para
        # medir a taxa de correção sem depender só do manifesto
        nome = f'{grao}__{classe}.jpg'
        cv2.imwrite(os.path.join(REVISAR, classe, nome), corte,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        salvos[classe] += 1
        linhas.append({
            'arquivo': nome, 'grao': grao, 'classe_prevista': classe,
            'confianca': round(votos[tid][classe] / max(sum(votos[tid].values()), 1e-9), 3),
            'n_quadros': visto[tid], 'nitidez': round(nit, 1),
            'caixa': [int(v) for v in caixa],
            'lado_px': int(max(caixa[2] - caixa[0], caixa[3] - caixa[1])),
            'lote': args.lote, 'sessao': sessao, 'travas': not args.sem_trava,
            'engine': os.path.basename(args.engine),
            'quando': time.strftime('%Y-%m-%d %H:%M:%S'),
        })

    os.makedirs(RAIZ, exist_ok=True)
    with open(MANIFESTO, 'a') as f:
        for l in linhas:
            f.write(json.dumps(l, ensure_ascii=False) + '\n')
    # revisao.csv no mesmo formato do deck/vigil_deck.py e do aprendizado_ativo
    csv_path = os.path.join(RAIZ, f'revisao_{sessao}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['grao', 'classe_prevista', 'confianca', 'n_quadros', 'nitidez'])
        for l in linhas:
            w.writerow([l['grao'], l['classe_prevista'], l['confianca'],
                        l['n_quadros'], l['nitidez']])

    print('\n' + '=' * 68)
    print(f'{len(linhas)} grãos salvos em {n_quadros} quadros ({time.time()-t0:.0f}s)')
    print('\nComo o MODELO separou (é proposta, não verdade):')
    for c in CLASSES:
        if salvos[c]:
            print(f'  {c:14s} {salvos[c]:5d}')
    if descartes:
        print(f'\ndescartados antes da revisão: {dict(descartes)}')
        if descartes['borrado'] > len(linhas):
            print('  Muito borrado — confira o foco (./setup_camera.sh --ver).')
    print(f'\ncsv: {os.path.basename(csv_path)}')
    print('\n' + '-' * 68)
    print('AGORA A PARTE MANUAL:')
    print(f'  1. abra  {REVISAR}')
    print('  2. ligue as miniaturas grandes no gerenciador de arquivos')
    print('  3. arraste o que estiver na pasta errada para a pasta certa')
    print(f'  4. o que não for grão (ou você não tiver certeza) -> {DESCARTE}/')
    print('\n  A maioria já vai estar certa: você só mexe no que o modelo errou.')
    print('  Depois:  python3 coletar_dataset.py --dividir')
    print('-' * 68)


# ---------------------------------------------------------------- divisão
def split_do_grao(grao):
    """Split determinístico a partir do ID do GRÃO.

    Hash em vez de sorteio: dividir de novo dá o mesmo resultado, e capturas
    novas não remexem o que já estava dividido. E como a chave é o grão, todos
    os recortes dele caem juntos — é o que impede o mesmo grão físico de
    aparecer no treino e na validação ao mesmo tempo.
    """
    h = int(hashlib.md5(grao.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return 'test'
    if h < TEST_FRAC + VAL_FRAC:
        return 'valid'
    return 'train'


def ler_revisao():
    """Lê o estado ATUAL das pastas — é onde você moveu que vale, não o manifesto.

    O nome do arquivo guarda a classe que o modelo propôs (`..__classe.jpg`),
    então comparar a pasta com o nome dá a taxa de correção de graça.
    """
    itens = []
    for pasta in CLASSES + [DESCARTE]:
        d = os.path.join(REVISAR, pasta)
        if not os.path.isdir(d):
            continue
        for nome in sorted(os.listdir(d)):
            if not nome.lower().endswith('.jpg'):
                continue
            base = nome[:-4]
            grao, _, prevista = base.rpartition('__')
            itens.append({'arquivo': nome, 'pasta': pasta,
                          'grao': grao or base,
                          'prevista': prevista if grao else None,
                          'caminho': os.path.join(d, nome)})
    return itens


def dividir(args):
    itens = ler_revisao()
    if not itens:
        sys.exit(f'nada em {REVISAR} — capture primeiro.')

    meta = {}
    if os.path.exists(MANIFESTO):
        for l in open(MANIFESTO):
            if l.strip():
                d = json.loads(l)
                meta[d['arquivo']] = d

    # captura sem travas é outro domínio: misturar recria o domain shift que o
    # rig existe para eliminar
    if not args.incluir_sem_trava:
        fora = [i for i in itens if meta.get(i['arquivo'], {}).get('travas', True) is False]
        if fora:
            print(f'IGNORANDO {len(fora)} recortes capturados SEM as travas de exposição.')
            print('  São de outro domínio. Use --incluir-sem-trava para forçar.\n')
            itens = [i for i in itens if i not in fora]

    descartados = [i for i in itens if i['pasta'] == DESCARTE]
    itens = [i for i in itens if i['pasta'] != DESCARTE]

    shutil.rmtree(PRONTO, ignore_errors=True)
    for sp in ('train', 'valid', 'test'):
        for c in CLASSES:
            os.makedirs(os.path.join(PRONTO, sp, c), exist_ok=True)

    por_split = defaultdict(Counter)
    graos_split = defaultdict(set)
    for i in itens:
        sp = split_do_grao(i['grao'])
        destino = os.path.join(PRONTO, sp, i['pasta'], i['arquivo'])
        try:
            os.link(i['caminho'], destino)
        except OSError:
            shutil.copy(i['caminho'], destino)
        por_split[sp][i['pasta']] += 1
        graos_split[sp].add(i['grao'])

    for a in ('train', 'valid', 'test'):
        for b in ('train', 'valid', 'test'):
            if a < b:
                comum = graos_split[a] & graos_split[b]
                assert not comum, f'VAZAMENTO: {len(comum)} grãos em {a} e {b}'

    # --- a medição que sai de graça ---
    com_previsao = [i for i in itens if i['prevista'] in CLASSES]
    acertos = sum(1 for i in com_previsao if i['prevista'] == i['pasta'])
    confusao = Counter((i['prevista'], i['pasta'])
                       for i in com_previsao if i['prevista'] != i['pasta'])

    cab = f'{"classe":16s}' + ''.join(f'{s:>9s}' for s in ('train', 'valid', 'test')) + f'{"total":>9s}'
    print('=' * 68)
    print(' Dataset dividido — rótulo é a pasta em que VOCÊ deixou')
    print('=' * 68)
    print(cab)
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        print(f'{c:16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print('-' * len(cab))
    n = [sum(por_split[s].values()) for s in ('train', 'valid', 'test')]
    print(f'{"TOTAL":16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print(f'{"grãos únicos":16s}'
          + ''.join(f'{len(graos_split[s]):9d}' for s in ('train', 'valid', 'test'))
          + f'{sum(len(v) for v in graos_split.values()):9d}')
    if descartados:
        print(f'\n{len(descartados)} em {DESCARTE}/ (fora do dataset)')
    print('\nSem vazamento: nenhum grão físico aparece em dois splits.')

    vazias = [c for c in CLASSES if sum(por_split[s][c] for s in por_split) == 0]
    if vazias:
        print(f'\nATENÇÃO: nenhuma amostra de {vazias}.')
        print('  Um modelo treinado assim nunca prevê essas classes.')

    if com_previsao:
        taxa = acertos / len(com_previsao)
        print('\n' + '=' * 68)
        print(' Quanto o modelo acertou — medido pelas SUAS correções')
        print('=' * 68)
        print(f'  {acertos}/{len(com_previsao)} grãos você deixou onde o modelo pôs '
              f'= {100*taxa:.1f}% de acerto')
        if confusao:
            print('\n  onde ele errou (modelo -> você corrigiu para):')
            for (p, r), k in confusao.most_common(8):
                print(f'    {p:14s} -> {r:14s} {k:4d}')
        print('\n  Este é o primeiro número de acurácia do projeto medido NO RIG,')
        print('  contra rótulo humano. Ele sai de graça do trabalho de anotar —')
        print('  e é o que a documentação lista como pendente.')

    _relatorio(por_split, graos_split, itens, descartados, com_previsao, acertos, confusao)
    print(f'\npronto em: {PRONTO}')


def _relatorio(por_split, graos_split, itens, descartados, com_previsao, acertos, confusao):
    md = ['# Dataset do rig — relatório', '',
          f'Gerado em {time.strftime("%Y-%m-%d %H:%M:%S")}.', '',
          '## Conteúdo', '', '| classe | train | valid | test | total |',
          '|---|---|---|---|---|']
    for c in CLASSES:
        n = [por_split[s][c] for s in ('train', 'valid', 'test')]
        md.append(f'| `{c}` | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |')
    md += ['',
           f'- **{len(itens)} recortes** de '
           f'**{sum(len(v) for v in graos_split.values())} grãos físicos únicos**',
           f'- {len(descartados)} descartados na revisão']
    if com_previsao:
        md += ['', '## Acurácia do modelo no rig', '',
               f'**{acertos}/{len(com_previsao)} = {100*acertos/len(com_previsao):.1f}%** '
               'dos grãos ficaram onde o modelo os colocou.', '',
               'Este número sai da revisão manual: cada grão que você não moveu é',
               'um acerto, cada um que moveu é um erro. É medido **no rig**, contra',
               'rótulo humano — diferente de toda avaliação anterior do projeto,',
               'que era em outro domínio ou por inspeção visual.']
        if confusao:
            md += ['', '| modelo previu | você corrigiu para | n |', '|---|---|---|']
            for (p, r), k in confusao.most_common(12):
                md.append(f'| `{p}` | `{r}` | {k} |')
    md += ['', '## Como ler estes números', '',
           'A contagem que importa é a de **grãos físicos únicos**: recortes do',
           'mesmo grão não são exemplos independentes.', '',
           'A divisão é **por grão**, não por imagem — todos os recortes de um',
           'grão caem no mesmo split. Dividir por imagem colocaria o mesmo grão',
           'no treino e na validação, e a validação mediria memorização.', '',
           '## Procedência', '',
           'Caixas do detector em quadro multi-grão, classe proposta por voto',
           'temporal (as mesmas regras do app ao vivo) e **corrigida à mão**. O',
           'rótulo final é a pasta em que o grão foi deixado na revisão.', '',
           'Só entram capturas com as travas de exposição ligadas: captura em',
           'automático é outro domínio e fica de fora por padrão.']
    open(os.path.join(PRONTO, 'RELATORIO.md'), 'w').write('\n'.join(md) + '\n')
    with open(os.path.join(PRONTO, 'dataset.yaml'), 'w') as f:
        f.write(f'# Vígil.ia — dataset do rig ({time.strftime("%Y-%m-%d")})\n')
        f.write(f'path: {PRONTO}\ntrain: train\nval: valid\ntest: test\n\n')
        f.write(f'nc: {len(CLASSES)}\nnames:\n')
        for i, c in enumerate(CLASSES):
            f.write(f'  {i}: {c}\n')


# ---------------------------------------------------------------- status
def status(args):
    print('=' * 68)
    print(' Status da revisão')
    print('=' * 68)
    itens = ler_revisao()
    if not itens:
        print(f'nada em {REVISAR} — capture primeiro.')
        return
    por_pasta = Counter(i['pasta'] for i in itens)
    graos = {i['grao'] for i in itens if i['pasta'] != DESCARTE}
    print(f'{"pasta":16s} {"grãos":>8s}   meta 120')
    for c in CLASSES:
        barra = '#' * min(30, int(30 * por_pasta[c] / 120))
        print(f'{c:16s} {por_pasta[c]:8d}   {barra}')
    print(f'{DESCARTE:16s} {por_pasta[DESCARTE]:8d}')
    print(f'\n{len(graos)} grãos úteis no total')
    movidos = sum(1 for i in itens
                  if i['prevista'] in CLASSES and i['prevista'] != i['pasta'])
    print(f'{movidos} já foram movidos por você '
          f'({100*movidos/max(len(itens),1):.0f}% do total)')
    falta = [c for c in CLASSES if por_pasta[c] < 120]
    if falta:
        print(f'\nfalta capturar mais: {", ".join(falta)}')


def main():
    ap = argparse.ArgumentParser(
        description='Coleta multi-grão: o modelo propõe, você corrige')
    ap.add_argument('--engine', default=None, help='engine .engine do detector')
    ap.add_argument('--lote', default='L001', help='identificador do lote de soja')
    ap.add_argument('--graos', type=int, default=300, help='meta de grãos por sessão')
    ap.add_argument('--camera', default='csi')
    ap.add_argument('--source', default=None, help='vídeo gravado em vez da câmera')
    ap.add_argument('--roi', type=int, default=0, metavar='PX')
    ap.add_argument('--tiles', type=int, default=1, metavar='N')
    ap.add_argument('--conf', type=float, default=0.25,
                    help='confiança mínima; baixa de propósito — você filtra na revisão')
    ap.add_argument('--class-offset', type=int, default=None)
    ap.add_argument('--sem-trava', action='store_true',
                    help='AE/AWB automáticos — NÃO use para dataset')
    ap.add_argument('--parado', action='store_true',
                    help='bandeja estática (desliga a compensação de movimento)')
    ap.add_argument('--sem-janela', action='store_true')
    ap.add_argument('--dividir', action='store_true')
    ap.add_argument('--incluir-sem-trava', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    if args.status:
        return status(args)
    if args.dividir:
        return dividir(args)
    if not args.engine:
        cands = sorted(f for f in os.listdir('.') if f.endswith('.engine'))
        if len(cands) != 1:
            ap.error('diga qual engine usar: --engine X.engine'
                     + (f'\n  disponíveis: {", ".join(cands)}' if cands else ''))
        args.engine = cands[0]
    capturar(args)


if __name__ == '__main__':
    main()
