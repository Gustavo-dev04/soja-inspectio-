#!/usr/bin/env python3
"""Testa o pipeline de dataset do `treino_base12k_jetson.ipynb` — sem GPU.

Monta um 12,5k FALSO com a mesma estrutura de pastas do Roboflow, executa as
células de função e de construção do notebook, e confere o que costuma quebrar
em silêncio: pasta que não é classe virando classe, vazamento entre splits,
escala do grão fora da do rig, prior de defeito fixo, caixa deslocada.

    python3 testar_base12k.py
"""
import json, os, shutil, sys
import numpy as np, cv2

nb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'treino_base12k_jetson.ipynb')))
codigo = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']

# --- monta um dataset falso com a MESMA estrutura do Roboflow ---
RAIZ = 'fake12k'
for _d in ('det_out', 'coco_out'):   # estado de rodada anterior engana o guard
    shutil.rmtree(_d, ignore_errors=True)
shutil.rmtree(RAIZ, ignore_errors=True)
PASTAS = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans',
          'Skin-damaged soybeans', 'Spotted soybeans',
          'Part of the original soybean images']   # <- tem que ser IGNORADA
rng = np.random.default_rng(0)
n_por = {'train': 12, 'valid': 5}
for sp, n in n_por.items():
    for f in PASTAS:
        d = f'{RAIZ}/{sp}/{f}'; os.makedirs(d)
        for k in range(n):
            img = np.full((400, 400, 3), 12, np.uint8)     # fundo escuro
            # tom de soja em BGR: bege/marrom, com saturação real (o
            # extract_cutout segmenta por saturação, não por brilho)
            cor = (int(rng.integers(60, 110)), int(rng.integers(150, 190)),
                   int(rng.integers(190, 230)))
            cv2.ellipse(img, (200, 200), (95, 70), float(rng.uniform(0, 180)),
                        0, 360, cor, -1)
            cv2.imwrite(f'{d}/{k:03d}.jpg', img)

# --- config equivalente à célula 4, sem Colab/torch ---
g = {'os': os, 'shutil': shutil, 'np': np, 'cv2': cv2, 'json': json}
exec("VARIANTE='large'; RES=704; SIZE=RES\n"
     "PX_POR_MM=12.7; GRAO_MM=7.0; ESPACO_MM=GRAO_MM*1.4\n"
     "GRAO_PX=GRAO_MM*PX_POR_MM; GRAO_PX_MIN=int(GRAO_PX*0.75); GRAO_PX_MAX=int(GRAO_PX*1.35)\n"
     "_lado_mm=SIZE/PX_POR_MM; _max_grao=int((_lado_mm/ESPACO_MM)**2)\n"
     "GRAOS_MIN,GRAOS_MAX=max(3,int(_max_grao*0.4)),_max_grao\n"
     "FUNDO_MIN,FUNDO_MAX=8,45; VINHETA_MAX=30\n"
     "DEFEITO_MIN,DEFEITO_MAX=0.02,0.65\n"
     "N_CENAS_TRAIN=12; N_CENAS_VAL=4; BLUR_FRAC=0.4; FRAC_SOLTAS=1.0\n"
     f"CLS_BASE={RAIZ!r}; DET_DIR='det_out'; COCO_DIR='coco_out'\n", g)

exec(codigo[2], g)          # funções de dataset
exec(codigo[3], g)          # construir + yolo_to_coco

print(f'\nescala: grão alvo {g["GRAO_PX"]:.0f} px, cenas com '
      f'{g["GRAOS_MIN"]}-{g["GRAOS_MAX"]} grãos')

# ---------------- verificações ----------------
import glob
from collections import Counter
NAMES = g['NAMES']

# 1. a pasta "Part of the original" NÃO pode ter virado classe
itens = g['collect_base'](RAIZ)
assert len(itens) == 5 * (12 + 5), f'esperava 85 imagens (5 classes), veio {len(itens)}'
print('OK: pasta "Part of the original soybean images" excluída')

# 2. splits: nenhum grão do val aparece em cena de treino
tr = glob.glob('det_out/images/train/*.jpg'); va = glob.glob('det_out/images/val/*.jpg')
assert tr and va, (len(tr), len(va))
assert all('_val_' not in os.path.basename(p) for p in tr)
assert all('_train_' not in os.path.basename(p) for p in va)
print(f'OK: {len(tr)} imagens de train, {len(va)} de val, sem vazamento de nome')

# 3. canvas e escala do grão nas cenas
lados, contagens, classes = [], [], Counter()
for p in glob.glob('det_out/labels/train/cena_*.txt'):
    img = cv2.imread(p.replace('/labels/', '/images/').replace('.txt', '.jpg'))
    assert img.shape[:2] == (704, 704), img.shape
    linhas = [l.split() for l in open(p).read().splitlines() if l.strip()]
    contagens.append(len(linhas))
    for c, cx, cy, w, h in linhas:
        classes[NAMES[int(c)]] += 1
        lados.append((float(w) + float(h)) / 2 * 704)
        for v in (cx, cy, w, h):
            assert 0 <= float(v) <= 1, f'coordenada fora de [0,1]: {v}'
print(f'OK: canvas 704x704, {len(contagens)} cenas, '
      f'{int(np.median(contagens))} grãos por cena (mediana)')
assert g['GRAOS_MIN'] <= max(contagens) <= g['GRAOS_MAX'], (min(contagens), max(contagens))
med = np.mean(lados)
assert g['GRAO_PX_MIN'] * 0.9 <= med <= g['GRAO_PX_MAX'] * 1.1, med
print(f'OK: grão a {med:.0f} px em média (alvo do rig: {g["GRAO_PX"]:.0f} px, '
      f'faixa {g["GRAO_PX_MIN"]}-{g["GRAO_PX_MAX"]})')

# 4. o prior de defeito varia entre cenas (não é fixo)
taxas = []
for p in glob.glob('det_out/labels/*/cena_*.txt'):
    ls = [int(l.split()[0]) for l in open(p).read().splitlines() if l.strip()]
    if ls:
        taxas.append(sum(1 for c in ls if NAMES[c] != 'intact') / len(ls))
assert max(taxas) - min(taxas) > 0.25, f'taxa de defeito quase fixa: {min(taxas):.2f}-{max(taxas):.2f}'
print(f'OK: taxa de defeito varia de {100*min(taxas):.0f}% a {100*max(taxas):.0f}% '
      f'entre cenas (lote bom e lote ruim)')
assert classes['intact'] > 0 and len(classes) >= 4, dict(classes)
print(f'OK: todas as classes aparecem: {dict(classes)}')

# 5. fundo da câmara: escuro, com centro mais claro que a borda
img = cv2.imread(sorted(glob.glob('det_out/images/train/cena_*.jpg'))[0])
cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fundo = cinza[cinza < 90]
assert fundo.mean() < 70, f'fundo claro demais: {fundo.mean():.0f}'
print(f'OK: fundo escuro (média {fundo.mean():.0f}/255) como a cartolina preta')
vin = g['fundo_camara'](704, np.random.default_rng(3))[:, :, 0].astype(float)
centro, borda = vin[302:402, 302:402].mean(), np.r_[vin[:40].ravel(), vin[-40:].ravel()].mean()
assert centro > borda, (centro, borda)
print(f'OK: vinheta do ring light — centro {centro:.0f} > borda {borda:.0f}')

# 6. caixa cola no grão nas fotos soltas
p = sorted(glob.glob('det_out/labels/train/solta_*.txt'))[0]
img = cv2.imread(p.replace('/labels/', '/images/').replace('.txt', '.jpg'))
c, cx, cy, w, h = open(p).read().split()
cx, cy, w, h = (float(v) * 704 for v in (cx, cy, w, h))
x1, y1, x2, y2 = int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dentro = gr[y1:y2, x1:x2]; fora = gr.copy(); fora[y1:y2, x1:x2] = 0
assert dentro.mean() > 90 and fora.max() < 60, (dentro.mean(), fora.max())
print(f'OK: caixa da foto solta cola no grão (dentro {dentro.mean():.0f}, fora max {fora.max()})')

# 7. COCO: category_id 1-indexado, caixas em pixels
cj = json.load(open('coco_out/valid/_annotations.coco.json'))
assert [c['name'] for c in cj['categories']] == NAMES
assert min(a['category_id'] for a in cj['annotations']) >= 1
assert max(a['category_id'] for a in cj['annotations']) <= 5
a = cj['annotations'][0]
assert a['bbox'][2] > 2 and a['bbox'][3] > 2, a['bbox']
print(f'OK: COCO com {len(cj["images"])} imgs, {len(cj["annotations"])} caixas, '
      f'category_id 1..5 e bbox em pixels')

shutil.rmtree(RAIZ); shutil.rmtree('det_out'); shutil.rmtree('coco_out')
print('\n=== pipeline de dataset do notebook validado ===')
