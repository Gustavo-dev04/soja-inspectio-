#!/usr/bin/env python3
"""Vígil.ia — dataset do rig: status, exportação para treinar do zero, e captura.

A CAPTURA AGORA É AUTOMÁTICA. O `vigil_jetson.py` grava sozinho toda vez que
inspeciona no padrão do rig (CSI com exposição travada) — ver
`gravador_dataset.py`. Este script cuida do que vem depois:

    # 1. inspecionar = coletar: as pastas aparecem sozinhas
    python3 vigil_jetson.py --engine X.engine --camera csi:1 --roi 704 \\
        --tiles 2 --esteira --lote L001

    # 2. CORRIGIR À MÃO: abra dataset/revisar/<sessao>/ e arraste o que estiver
    #    na pasta errada. Uma correção vale para TODAS as caixas daquele grão.

    # 3. quanto já tem, por classe
    python3 coletar_dataset.py --status

    # 4. exportar em YOLO + COCO, dividido em train/valid/test
    python3 coletar_dataset.py --exportar

O atalho de captura antigo continua valendo — agora é o vigil com os padrões de
coleta (confiança baixa, meta de 400 grãos):

    python3 coletar_dataset.py --engine X.engine --camera csi:1 --roi 704 --tiles 2

Na revisão, duas pastas além das classes:

    descartar/   não é grão (sujeira, reflexo, caixa dupla) -> a caixa some
    duvida/      é grão, mas não dá pra saber a classe -> a IMAGEM inteira sai

Deixar o recorte onde está é concordar com o modelo.

O que sai daqui treina um **detector do zero**, não um classificador:

    dataset/pronto/
        train/images/*.jpg   train/labels/*.txt      <- YOLO
        train/_annotations.coco.json                 <- COCO (rfdetr, DETR…)
        dataset.yaml   RELATORIO.md

Por que a cena, e não só o recorte
----------------------------------
Recorte solto treina classificador. Detector precisa aprender *onde* o grão
está, quantos existem e como eles se encostam — e isso só está na cena. Cada
imagem é exatamente uma janela de entrada do modelo (704x704 no rig), para o
treino ver o grão na mesma escala da inferência.

Grão sem rótulo é pior que grão a menos
---------------------------------------
Um grão que aparece na imagem sem caixa ensina o detector que aquilo é FUNDO.
Por isso a imagem inteira sai se algum grão dela ficou sem rótulo (recorte
perdido, ou mandado para `duvida/`). `descartar/` é diferente: ali não há grão,
então não falta caixa nenhuma.

Divisão sem vazamento
---------------------
Quadros seguidos da esteira são quase idênticos: jogar o N no treino e o N+1 na
validação é vazamento disfarçado. A captura é cortada em blocos de tempo, e
blocos que compartilham um grão vão juntos para o MESMO split. Na esteira cada
bloco fica sozinho; na bandeja parada, onde o grão fica a sessão inteira, a
sessão vira um bloco só — em vez de quebrar a exportação. Uma banda de guarda
nas fronteiras entre splits diferentes cobre a troca de ID do rastreamento.
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

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)
import gravador_dataset as gd  # noqa: E402  (sem TensorRT: exporta em qualquer PC)

CLASSES = gd.CLASSES
DESCARTE, DUVIDA = gd.DESCARTE, gd.DUVIDA
RAIZ = gd.RAIZ_PADRAO
QUADROS = os.path.join(RAIZ, 'quadros')     # o que o modelo viu + caixas
REVISAR = os.path.join(RAIZ, 'revisar')     # recortes, para a revisão humana
PRONTO = os.path.join(RAIZ, 'pronto')       # dataset exportado

VAL_FRAC, TEST_FRAC = 0.15, 0.10
SPLITS = ('train', 'valid', 'test')


def usar_raiz(raiz):
    global RAIZ, QUADROS, REVISAR, PRONTO
    RAIZ = raiz
    QUADROS = os.path.join(raiz, 'quadros')
    REVISAR = os.path.join(raiz, 'revisar')
    PRONTO = os.path.join(raiz, 'pronto')


def _vigil():
    """Importa o vigil_jetson só para capturar: ele precisa de TensorRT, e
    status/exportação não — dá para exportar num notebook sem GPU."""
    nome = 'vigil_jetson'
    if nome in sys.modules:
        return sys.modules[nome]
    spec = importlib.util.spec_from_file_location(nome, os.path.join(_AQUI, 'vigil_jetson.py'))
    vj = importlib.util.module_from_spec(spec)
    sys.modules[nome] = vj
    spec.loader.exec_module(vj)
    return vj


# ---------------------------------------------------------------- captura
def capturar(args):
    """Apelido do vigil com os padrões de coleta — um laço de captura só."""
    argv = ['--engine', args.engine, '--conf', str(args.conf), '--esteira',
            '--lote', args.lote, '--dataset-dir', RAIZ,
            '--dataset-passo', str(args.passo), '--dataset-graos', str(args.graos),
            '--dataset-bloco', str(args.bloco), '--dataset-guarda', str(args.guarda),
            '--roi', str(args.roi), '--tiles', str(args.tiles)]
    argv += ['--source', args.source] if args.source else ['--camera', args.camera]
    if args.class_offset is not None:
        argv += ['--class-offset', str(args.class_offset)]
    if args.sem_trava:
        argv += ['--csi-sem-trava']
    if args.sem_trava or args.source:
        argv += ['--dataset']          # grava, mas a sessão sai marcada sem trava
    if args.parado:
        argv += ['--parado']
    if args.sem_janela:
        argv += ['--no-window']
    _vigil().main(argv)


# ---------------------------------------------------------------- leitura
def ler_revisao(conflitos=None):
    """Estado ATUAL das pastas: onde você deixou o arquivo é o rótulo.

    Lê o layout novo (`revisar/<sessao>/<pasta>/`) e o antigo
    (`revisar/<pasta>/`). Um grão em DUAS pastas — copiado em vez de movido —
    é ambíguo e vira `duvida`; passe uma lista em `conflitos` para recebê-los.
    """
    mapa = {}
    if not os.path.isdir(REVISAR):
        return mapa
    diretorios = []
    for nome in sorted(os.listdir(REVISAR)):
        d = os.path.join(REVISAR, nome)
        if not os.path.isdir(d):
            continue
        if nome in gd.PASTAS:
            diretorios.append((d, nome))                     # layout antigo
        else:
            diretorios += [(os.path.join(d, p), p) for p in gd.PASTAS
                           if os.path.isdir(os.path.join(d, p))]
    for d, pasta in diretorios:
        for arq in sorted(os.listdir(d)):
            base, ext = os.path.splitext(arq)
            if ext.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            grao, _, prevista = base.rpartition('__')
            if not grao:
                continue
            if grao in mapa and mapa[grao]['corrigida'] != pasta:
                if conflitos is not None:
                    conflitos.append(grao)
                mapa[grao]['corrigida'] = DUVIDA
                continue
            mapa[grao] = {'corrigida': pasta, 'prevista': prevista,
                          'arquivo': os.path.join(d, arq)}
    return mapa


def _sessoes():
    """[(dir, meta)] com meta['quadros'] completo, dos dois formatos."""
    if not os.path.isdir(QUADROS):
        return []
    saida = []
    for d in sorted(os.listdir(QUADROS)):
        dq = os.path.join(QUADROS, d)
        p = os.path.join(dq, 'sessao.json')
        if not os.path.exists(p):
            continue
        meta = json.load(open(p))
        quadros = list(meta.get('quadros', []))             # formato antigo
        pj = os.path.join(dq, 'quadros.jsonl')
        if os.path.exists(pj):
            for linha in open(pj):
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    quadros.append(json.loads(linha))
                except json.JSONDecodeError:
                    pass     # última linha cortada: o app caiu no meio da escrita
        meta['quadros'] = quadros
        saida.append((dq, meta))
    return saida


def split_do_bloco(chave):
    """Split determinístico por hash: exportar de novo dá o mesmo resultado."""
    h = int(hashlib.md5(chave.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_FRAC:
        return 'test'
    if h < TEST_FRAC + VAL_FRAC:
        return 'valid'
    return 'train'


def _componentes(sessoes, revisao):
    """Blocos que compartilham um grão rotulado vão juntos (union-find).

    Na esteira, cada bloco fica sozinho — mesma divisão de sempre. Na bandeja
    parada, o mesmo grão atravessa a sessão inteira e ela vira um bloco só. Só
    grão com classe liga blocos: o que foi descartado não é grão, e a imagem
    com grão em dúvida nem entra.

    Devolve {(sessao, bloco): representante}, com o MENOR nó como
    representante — determinístico, e igual ao próprio bloco quando ele está
    sozinho (é o que mantém a divisão antiga para sessões de esteira).
    """
    pai = {}

    def raiz(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    onde = {}
    for _, meta in sessoes:
        for q in meta['quadros']:
            no = (meta['sessao'], int(q['bloco']))
            pai.setdefault(no, no)
            for cx in q['caixas']:
                r = revisao.get(cx['grao'])
                if r is None or r['corrigida'] not in CLASSES:
                    continue
                g = cx['grao']
                if g in onde:
                    a, b = raiz(onde[g]), raiz(no)
                    if a != b:
                        pai[max(a, b)] = min(a, b)
                else:
                    onde[g] = no
    return {no: raiz(no) for no in pai}


# ---------------------------------------------------------------- exportação
def exportar(args):
    sessoes = _sessoes()
    if not sessoes:
        sys.exit(f'nada capturado ainda (sem sessões em {QUADROS})')
    conflitos = []
    revisao = ler_revisao(conflitos)
    if not revisao:
        sys.exit(f'nada em {REVISAR} — capture primeiro.')

    shutil.rmtree(PRONTO, ignore_errors=True)
    for sp in SPLITS:
        os.makedirs(os.path.join(PRONTO, sp, 'images'), exist_ok=True)
        os.makedirs(os.path.join(PRONTO, sp, 'labels'), exist_ok=True)

    coco = {sp: {'images': [], 'annotations': [], 'categories':
                 [{'id': i + 1, 'name': c, 'supercategory': 'soja'}
                  for i, c in enumerate(CLASSES)]}
            for sp in SPLITS}
    ids = {sp: [0, 0] for sp in coco}          # [img_id, ann_id]
    por_split = defaultdict(Counter)
    quadros_split = Counter()
    graos_split = defaultdict(set)
    descartadas = na_guarda = pulados_sem_trava = 0
    incompletos = Counter()                    # motivo -> imagens que saíram

    usadas = []
    for dir_q, meta in sessoes:
        if not meta.get('travas', True) and not args.incluir_sem_trava:
            pulados_sem_trava += len(meta['quadros'])
            continue
        usadas.append((dir_q, meta))

    comp = _componentes(usadas, revisao)
    split_de = {no: split_do_bloco(f'{rep[0]}_b{rep[1]:03d}') for no, rep in comp.items()}

    for dir_q, meta in usadas:
        sess = meta['sessao']
        bloco_s, guarda_s = meta.get('bloco_s', 20), meta.get('guarda_s', 3)
        for q in meta['quadros']:
            no = (sess, int(q['bloco']))
            sp = split_de[no]
            # banda de guarda só onde a fronteira separa splits DIFERENTES:
            # é ali que uma troca de ID poria o mesmo grão dos dois lados
            resto = q['t'] % bloco_s
            viz = None
            if resto < guarda_s / 2:
                viz = (sess, no[1] - 1)
            elif resto > bloco_s - guarda_s / 2:
                viz = (sess, no[1] + 1)
            if viz in split_de and split_de[viz] != sp:
                na_guarda += 1
                continue

            linhas, anns, motivo = [], [], None
            for cx in q['caixas']:
                r = revisao.get(cx['grao'])
                if r is None:
                    motivo = 'grão sem recorte para revisar'
                    break
                if r['corrigida'] == DUVIDA:
                    motivo = 'grão em duvida/'
                    break
                if r['corrigida'] == DESCARTE:
                    descartadas += 1
                    continue
                ci = CLASSES.index(r['corrigida'])
                x1, y1, x2, y2 = cx['caixa']
                W, H = q['largura'], q['altura']
                linhas.append(f'{ci} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} '
                              f'{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}')
                anns.append((ci, x1, y1, x2 - x1, y2 - y1, r['corrigida'], cx['grao']))
            if motivo:
                incompletos[motivo] += 1
                continue
            if not linhas:            # só sobrou o que foi descartado
                continue
            origem = os.path.join(dir_q, q['arquivo'])
            if not os.path.exists(origem):
                incompletos['imagem não está no disco'] += 1
                continue
            destino = os.path.join(PRONTO, sp, 'images', q['arquivo'])
            try:
                os.link(origem, destino)
            except OSError:
                shutil.copy(origem, destino)
            with open(os.path.join(PRONTO, sp, 'labels', q['arquivo'][:-4] + '.txt'), 'w') as f:
                f.write('\n'.join(linhas))

            iid, aid = ids[sp]
            coco[sp]['images'].append({'id': iid, 'file_name': q['arquivo'],
                                       'width': q['largura'], 'height': q['altura']})
            for ci, x, y, bw, bh, nome, grao in anns:
                coco[sp]['annotations'].append(
                    {'id': aid, 'image_id': iid, 'category_id': ci + 1,
                     'bbox': [x, y, bw, bh], 'area': bw * bh, 'iscrowd': 0})
                aid += 1
                por_split[sp][nome] += 1
                graos_split[sp].add(grao)
            ids[sp] = [iid + 1, aid]
            quadros_split[sp] += 1

    for sp in coco:
        with open(os.path.join(PRONTO, sp, '_annotations.coco.json'), 'w') as f:
            json.dump(coco[sp], f)

    # a garantia que justifica dividir por componente
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                comum = graos_split[a] & graos_split[b]
                assert not comum, f'VAZAMENTO: {len(comum)} grãos em {a} e {b}'

    # --- acurácia do modelo, medida pelas correções ---
    # `duvida` fica de fora: ali ninguém sabe a resposta certa
    com_prev = [r for r in revisao.values()
                if r['prevista'] in CLASSES and r['corrigida'] != DUVIDA]
    acertos = sum(1 for r in com_prev if r['prevista'] == r['corrigida'])
    confusao = Counter((r['prevista'], r['corrigida'])
                       for r in com_prev if r['prevista'] != r['corrigida'])

    cab = f'{"classe":16s}' + ''.join(f'{s:>9s}' for s in SPLITS) + f'{"total":>9s}'
    print('=' * 70)
    print(' Dataset de detecção exportado')
    print('=' * 70)
    print(cab)
    for c in CLASSES:
        n = [por_split[s][c] for s in SPLITS]
        print(f'{c:16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print('-' * len(cab))
    n = [sum(por_split[s].values()) for s in SPLITS]
    print(f'{"CAIXAS":16s}' + ''.join(f'{v:9d}' for v in n) + f'{sum(n):9d}')
    print(f'{"imagens":16s}' + ''.join(f'{quadros_split[s]:9d}' for s in SPLITS)
          + f'{sum(quadros_split.values()):9d}')
    print(f'{"grãos únicos":16s}' + ''.join(f'{len(graos_split[s]):9d}' for s in SPLITS)
          + f'{sum(len(v) for v in graos_split.values()):9d}')
    n_comp = len(set(comp.values()))
    print(f'\n{len(comp)} blocos de tempo em {n_comp} grupos independentes '
          f'(blocos com grão em comum andam juntos)')
    if descartadas:
        print(f'{descartadas} caixas de não-grão removidas ({DESCARTE}/)')
    for motivo, k in incompletos.most_common():
        print(f'{k} imagens fora: {motivo} (grão sem rótulo ensinaria "fundo")')
    if na_guarda:
        print(f'{na_guarda} imagens na banda de guarda entre splits')
    if pulados_sem_trava:
        print(f'{pulados_sem_trava} imagens de sessões SEM travas de exposição '
              f'(outro domínio; --incluir-sem-trava para forçar)')
    if conflitos:
        print(f'{len(conflitos)} grãos em DUAS pastas (copiados em vez de movidos) '
              f'— tratados como dúvida')
    print('\nSem vazamento: nenhum grão físico aparece em dois splits.')

    vazias = [c for c in CLASSES if sum(por_split[s][c] for s in por_split) == 0]
    if vazias:
        print(f'\nATENÇÃO: nenhuma caixa de {vazias}.')
        print('  Um detector treinado assim nunca prevê essas classes.')
    sem_img = [s for s in ('valid', 'test') if not quadros_split[s]]
    if sem_img and sum(quadros_split.values()):
        print(f'\nATENÇÃO: {" e ".join(sem_img)} ficaram vazios — só há {n_comp} '
              f'grupo(s) independente(s).')
        print('  Na bandeja parada a sessão inteira é um grupo: faça mais sessões')
        print('  curtas (uma por bandeja) em vez de uma longa.')

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
        print('\n  É a acurácia do modelo atual medida NO RIG contra rótulo')
        print('  humano — a linha de base que o modelo novo precisa bater.')
        print('  (vale como medida se você revisou TODOS os recortes, inclusive')
        print('   os que já estavam certos; recorte não olhado conta como acerto)')

    _relatorio(por_split, quadros_split, graos_split, com_prev, acertos, confusao,
               descartadas, na_guarda, incompletos, n_comp)
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
               descartadas, na_guarda, incompletos, n_comp):
    md = ['# Dataset de detecção do rig — relatório', '',
          f'Gerado em {time.strftime("%Y-%m-%d %H:%M:%S")}.', '',
          '## Conteúdo', '', '| classe | train | valid | test | total |',
          '|---|---|---|---|---|']
    for c in CLASSES:
        n = [por_split[s][c] for s in SPLITS]
        md.append(f'| `{c}` | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |')
    md += ['',
           f'- **{sum(sum(v.values()) for v in por_split.values())} caixas** em '
           f'**{sum(quadros_split.values())} imagens**',
           f'- **{sum(len(v) for v in graos_split.values())} grãos físicos únicos**, '
           f'em {n_comp} grupos independentes',
           f'- {descartadas} caixas de não-grão removidas na revisão, '
           f'{na_guarda} imagens na banda de guarda']
    for motivo, k in incompletos.most_common():
        md.append(f'- {k} imagens fora: {motivo}')
    if com_prev:
        md += ['', '## Acurácia do modelo atual no rig', '',
               f'**{acertos}/{len(com_prev)} = {100*acertos/len(com_prev):.1f}%** '
               'dos grãos ficaram onde o modelo os colocou.', '',
               'Sai da revisão manual: cada grão não movido é um acerto. É medido',
               '**no rig**, contra rótulo humano — e é a linha de base que o modelo',
               'treinado do zero precisa bater para justificar a troca. Só vale se',
               'todos os recortes foram olhados, inclusive os que estavam certos.']
        if confusao:
            md += ['', '| modelo previu | corrigido para | n |', '|---|---|---|']
            for (p, r), k in confusao.most_common(12):
                md.append(f'| `{p}` | `{r}` | {k} |')
    md += ['', '## Formato', '',
           'Sai nos dois formatos, do mesmo dado:', '',
           '- **YOLO**: `<split>/images/*.jpg` + `<split>/labels/*.txt`',
           '- **COCO**: `<split>/_annotations.coco.json`', '',
           'Cada imagem é uma janela de entrada do modelo, na escala da inferência,',
           'com todas as caixas da cena — dataset de **detecção**, não recorte solto.', '',
           '## Sem vazamento', '',
           'Quadros consecutivos da esteira são quase idênticos. A divisão é por',
           '**bloco de tempo**; blocos que compartilham um grão vão para o mesmo',
           'split, e há banda de guarda nas fronteiras entre splits diferentes.',
           'A exportação verifica que nenhum grão físico aparece em dois splits.', '',
           '## Procedência', '',
           'Caixas do detector atual em cena multi-grão, classe por voto temporal',
           'e **corrigida à mão**. O rótulo final é a pasta em que o grão foi',
           'deixado na revisão; uma correção vale para todas as caixas daquele',
           'grão, em todas as imagens. Imagem com grão sem rótulo é descartada',
           'inteira. Só entram sessões com as travas de exposição ligadas.', '',
           '**Limite conhecido:** grão que o detector atual NÃO achou fica sem',
           'caixa na imagem, e a revisão de recortes não enxerga isso. Para',
           'auditar, abra uma amostra do COCO no CVAT ou Label Studio.']
    with open(os.path.join(PRONTO, 'RELATORIO.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')


# ---------------------------------------------------------------- status
def status(args):
    print('=' * 70)
    print(f' Status do dataset — {RAIZ}')
    print('=' * 70)
    sessoes = _sessoes()
    conflitos = []
    revisao = ler_revisao(conflitos)
    if not sessoes:
        print('nada capturado ainda.')
        print('Rode o vigil_jetson.py no rig (CSI com travas): ele grava sozinho.')
        return
    graos_sessao = Counter(g.rsplit('_g', 1)[0] for g in revisao)
    print(f'{"sessão":24s} {"travas":>6s} {"imagens":>8s} {"caixas":>8s} {"grãos":>7s}')
    for _, m in sessoes:
        nq = len(m['quadros'])
        nc = sum(len(q['caixas']) for q in m['quadros'])
        ng = graos_sessao.get(f"{m.get('lote', '')}_{m['sessao']}", 0)
        print(f'{m["sessao"]:24s} {"sim" if m.get("travas", True) else "NÃO":>6s} '
              f'{nq:8d} {nc:8d} {ng:7d}')
    por_pasta = Counter(r['corrigida'] for r in revisao.values())
    print(f'\n{"pasta":16s} {"grãos":>8s}   meta {args.meta}')
    for c in CLASSES:
        barra = '#' * min(30, int(30 * por_pasta[c] / max(args.meta, 1)))
        print(f'{c:16s} {por_pasta[c]:8d}   {barra}')
    print(f'{DESCARTE:16s} {por_pasta[DESCARTE]:8d}')
    print(f'{DUVIDA:16s} {por_pasta[DUVIDA]:8d}')
    movidos = sum(1 for r in revisao.values()
                  if r['prevista'] in CLASSES and r['prevista'] != r['corrigida'])
    print(f'\n{movidos} de {len(revisao)} corrigidos por você '
          f'({100*movidos/max(len(revisao), 1):.0f}%)')
    if conflitos:
        print(f'{len(conflitos)} grãos em duas pastas (copiados em vez de movidos)')
    falta = [c for c in CLASSES if por_pasta[c] < args.meta]
    if falta:
        print(f'falta capturar mais: {", ".join(falta)}')


def main():
    ap = argparse.ArgumentParser(
        description='Dataset do rig: status, exportação e (atalho de) captura')
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
                    help='guarda 1 varredura a cada N (padrão 6)')
    ap.add_argument('--bloco', type=float, default=20,
                    help='tamanho do bloco de tempo, em segundos')
    ap.add_argument('--guarda', type=float, default=3,
                    help='banda de guarda entre blocos, em segundos')
    ap.add_argument('--class-offset', type=int, default=None)
    ap.add_argument('--sem-trava', action='store_true')
    ap.add_argument('--parado', action='store_true')
    ap.add_argument('--sem-janela', action='store_true')
    ap.add_argument('--dataset-dir', default=None, metavar='DIR',
                    help='raiz do dataset (padrão: $VIGIL_DATASET ou ./dataset)')
    ap.add_argument('--exportar', action='store_true')
    ap.add_argument('--incluir-sem-trava', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--meta', type=int, default=120,
                    help='grãos por classe considerados suficientes no --status')
    args = ap.parse_args()

    if args.dataset_dir:
        usar_raiz(args.dataset_dir)
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
