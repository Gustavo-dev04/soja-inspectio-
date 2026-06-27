# =============================================================================
#  Vígil.ia — Treino DEFINITIVO do classificador YOLO11s-cls no domínio real
#  -----------------------------------------------------------------------------
#  Self-contained para Google Colab (GPU). Roda de cima a baixo.
#
#  O que faz:
#    1. instala deps e monta o Drive
#    2. carrega o modelo base treinado no Roboflow (do Drive)
#    3. junta as fotos reais das DUAS pastas (Lotes + "Soja pra completar"),
#       mapeia os nomes em português para as classes do modelo,
#       recorta cada grão (OpenCV) e faz split estratificado train/val/test
#    4. mede o baseline (antes do fine-tuning)
#    5. TREINA (receita definitiva: mais épocas + early-stopping + aug forte)
#    6. avalia em val e test, imprime relatório e salva a matriz de confusão
#    7. salva o modelo no Drive SE o val melhorar (seleção por val, test = honesto)
#
#  Pré-requisito: o modelo base do Roboflow precisa estar no Drive
#  (soja_yolo11s_best.pt OU soja_yolo11s_finetuned.pt), gerado pelo train_yolo.ipynb.
# =============================================================================

# ----------------------------------------------------------------------------
# 1) SETUP
# ----------------------------------------------------------------------------
!pip -q install ultralytics

import os, shutil, random, glob, pathlib, json, unicodedata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix

from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive'

# >>> CONFIG — pastas no Drive com as fotos reais (uma subpasta por classe). <<<
# Várias raízes são unidas. Subpastas podem estar em PT ou EN (CLASS_ALIASES resolve).
REAL_SRCS = [
    '/content/drive/MyDrive/Soja total/Soja total/Lotes',
    '/content/drive/MyDrive/Soja pra completar',
]

CLASS_NAMES = [
    'Broken soybeans', 'Immature soybeans', 'Intact soybeans',
    'Skin-damaged soybeans', 'Spotted soybeans',
]
SHORT = {
    'Broken soybeans': 'broken', 'Immature soybeans': 'immature',
    'Intact soybeans': 'intact', 'Skin-damaged soybeans': 'skin-damaged',
    'Spotted soybeans': 'spotted',
}
CLASS_ALIASES = {
    'broken':       ['broken', 'quebrado', 'quebrada'],
    'immature':     ['immature', 'imaturo', 'imatura', 'nao maduro', 'verde'],
    'intact':       ['intact', 'intacto', 'intacta'],
    'skin-damaged': ['skin-damaged', 'skin damaged', 'casca danificada'],
    'spotted':      ['spotted', 'manchado', 'manchada'],
}
short_names = [SHORT[c] for c in CLASS_NAMES]
idx_of = {name: i for i, name in enumerate(CLASS_NAMES)}

def _norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(s.split())

def crop_single_grain(arr):
    """Mesmo recorte do app/treino: maior contorno via Otsu -> bounding box."""
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return arr
    largest = max(contours, key=cv2.contourArea)
    ratio = cv2.contourArea(largest) / (h * w)
    if ratio <= 0.02 or ratio >= 0.95:
        return arr
    x, y, bw, bh = cv2.boundingRect(largest)
    pad = max(15, int(min(bw, bh) * 0.12))
    return arr[max(0, y-pad):min(h, y+bh+pad), max(0, x-pad):min(w, x+bw+pad)]

# ----------------------------------------------------------------------------
# 2) MODELO BASE (Roboflow) — do Drive
# ----------------------------------------------------------------------------
BASE_CANDS = [f'{SAVE_DIR}/soja_yolo11s_best.pt', f'{SAVE_DIR}/soja_yolo11s_finetuned.pt']
BASE_PT = next((p for p in BASE_CANDS if os.path.exists(p)), None)
assert BASE_PT, (f'❌ NÃO achei o modelo base no Drive. Esperado um destes:\n   ' +
                 '\n   '.join(BASE_CANDS) +
                 '\n   Gere com o train_yolo.ipynb (Célula 8).')
model = YOLO(BASE_PT)
assert [model.names[i] for i in range(len(CLASS_NAMES))] == CLASS_NAMES, \
    'Ordem das classes do modelo diverge de CLASS_NAMES!'
print('✅ Modelo base:', BASE_PT)

# ----------------------------------------------------------------------------
# 3) DATASET — junta as raízes, casa nomes PT/EN, recorta, split 70/15/15
# ----------------------------------------------------------------------------
REAL_DST = '/content/soja_real'
if os.path.exists(REAL_DST):
    shutil.rmtree(REAL_DST)
rng = random.Random(42)
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
VAL_FRAC, TEST_FRAC = 0.15, 0.15

def _resolve_roots(cands):
    roots = []
    for c in cands:
        if os.path.isdir(c):
            roots.append(c)
        else:
            roots += [d for d in glob.glob(c.rstrip('/').rstrip() + '*') if os.path.isdir(d)]
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

ROOTS = _resolve_roots(REAL_SRCS)
assert ROOTS, f'❌ Nenhuma raiz encontrada. Ajuste REAL_SRCS. Tentei: {REAL_SRCS}'
print('Raízes:'); [print('  -', r) for r in ROOTS]

alias_to_cls = {}
for cls in CLASS_NAMES:
    for a in CLASS_ALIASES[SHORT[cls]]:
        alias_to_cls[_norm(a)] = cls

def _images_for_class(cls):
    out = []
    for root in ROOTS:
        for name in sorted(os.listdir(root)):
            sub = os.path.join(root, name)
            if os.path.isdir(sub) and alias_to_cls.get(_norm(name)) == cls:
                out += [os.path.join(sub, f) for f in os.listdir(sub)
                        if pathlib.Path(f).suffix.lower() in IMG_EXTS]
    return sorted(out)

summary = {}
for cls in CLASS_NAMES:
    fs = _images_for_class(cls)
    rng.shuffle(fs)
    n = len(fs)
    n_val  = max(1, round(n * VAL_FRAC))  if n >= 3 else 0
    n_test = max(1, round(n * TEST_FRAC)) if n >= 4 else 0
    val_set, test_set = set(fs[:n_val]), set(fs[n_val:n_val + n_test])
    for p in fs:
        split = 'val' if p in val_set else ('test' if p in test_set else 'train')
        try:
            arr = np.array(Image.open(p).convert('RGB'))
        except Exception as e:
            print('  ⚠️ pulei:', p, e); continue
        crop = crop_single_grain(arr)
        outdir = os.path.join(REAL_DST, split, cls); os.makedirs(outdir, exist_ok=True)
        Image.fromarray(crop).save(os.path.join(outdir, pathlib.Path(p).stem + '.jpg'), quality=95)
    summary[SHORT[cls]] = (n - n_val - n_test, n_val, n_test)

print('\nclasse        -> (train, val, test)   [total]')
total = [0, 0, 0]
for k, v in summary.items():
    print(f'  {k:>13}: {v}   [{sum(v)}]'); total = [a + b for a, b in zip(total, v)]
print(f'  {"TOTAL":>13}: {tuple(total)}   [{sum(total)} fotos]')
for k, v in summary.items():
    if sum(v) == 0:
        print(f'⚠️  classe "{k}" ZERADA — confira o nome da pasta/alias.')
trains = [v[0] for v in summary.values()]
if min(trains) and max(trains) > 2 * min(trains):
    print('⚠️  desbalanceado no train (maior > 2x menor) — nivele a captura.')

# ----------------------------------------------------------------------------
# 4) BASELINE (antes do fine-tuning)
# ----------------------------------------------------------------------------
def eval_split(m, split):
    paths, true = [], []
    for cls in CLASS_NAMES:
        for p in sorted(glob.glob(os.path.join(REAL_DST, split, cls, '*.jpg'))):
            paths.append(p); true.append(idx_of[cls])
    if not paths:
        return None, None, None
    imgs = [Image.open(p).convert('RGB') for p in paths]
    pred = [idx_of[m.names[int(r.probs.top1)]]
            for r in m.predict(imgs, imgsz=224, verbose=False)]
    true, pred = np.array(true), np.array(pred)
    return (true == pred).mean(), true, pred

base_tr,  _, _ = eval_split(model, 'train')
base_val, _, _ = eval_split(model, 'val')
base_test, _, _ = eval_split(model, 'test')
print(f'\nBASELINE (sem fine-tuning): train {base_tr:.1%} | val {base_val:.1%} | '
      f'test {base_test:.1%}' if base_test is not None else f'\nBASELINE: train {base_tr:.1%} | val {base_val:.1%}')

# ----------------------------------------------------------------------------
# 5) TREINO DEFINITIVO
#    GPU forte: mais épocas + early-stopping (patience) + EMA + aug forte.
#    FREEZE: 9 = último bloco + cabeça (recomendado). Com dataset bem nivelado
#    e >=80/classe, dá pra testar 8 (mais capacidade). 10 = só a cabeça.
# ----------------------------------------------------------------------------
FREEZE = 9
ft = YOLO(BASE_PT)
ft.train(
    data=REAL_DST,
    epochs=120,          # teto alto; o early-stopping corta no melhor
    patience=30,         # para se o val não melhorar por 30 épocas
    imgsz=224,           # igual à inferência de produção
    batch=32,            # bom p/ ~500 fotos; suba p/ 64 se tiver MUITA foto
    freeze=FREEZE,
    lr0=0.0008, lrf=0.01, cos_lr=True,
    dropout=0.1,         # regulariza a cabeça de classificação
    seed=42,
    cache=True,          # cacheia em RAM (GPU forte) -> épocas rápidas
    # --- augmentation forte: a alavanca contra o domain shift (fundo/luz) ---
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.6,
    degrees=15, flipud=0.5, fliplr=0.5,
    translate=0.1, scale=0.5, erasing=0.4,
    project='/content/runs_soja', name='yolo11s_definitivo',
    exist_ok=True, plots=True, verbose=True,
)
print('\nTreino concluído. save_dir:', ft.trainer.save_dir)

# ----------------------------------------------------------------------------
# 6) AVALIAÇÃO — antes x depois (val + test) + matriz de confusão (no test)
# ----------------------------------------------------------------------------
new_tr,  _, _                 = eval_split(ft, 'train')
new_val, val_true, val_pred   = eval_split(ft, 'val')
new_test, test_true, test_pred = eval_split(ft, 'test')

print('\n=== Domínio real: ANTES -> DEPOIS ===')
print(f'  train: {base_tr:.1%} -> {new_tr:.1%}')
print(f'  val:   {base_val:.1%} -> {new_val:.1%}   (seleção/gate)')
if new_test is not None:
    print(f'  test:  {base_test:.1%} -> {new_test:.1%}   (headline honesto)')
gap = new_tr - new_val
print(f'  gap train-val: {gap:.1%} ' + ('(overfitting)' if gap > 0.15 else '(saudável)'))

rep_true, rep_pred, rep_split = (
    (test_true, test_pred, 'test') if new_test is not None else (val_true, val_pred, 'val'))
print(f'\n=== Relatório no {rep_split} real ===')
print(classification_report(rep_true, rep_pred, target_names=short_names, zero_division=0))

cm = confusion_matrix(rep_true, rep_pred, labels=list(range(len(CLASS_NAMES))))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=short_names, yticklabels=short_names,
            cmap='Greens', linewidths=0.5)
plt.xlabel('Predito'); plt.ylabel('Real')
plt.title(f'Vígil.ia — {rep_split} {(new_test or new_val):.0%}')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/vigil_matriz_confusao.png', dpi=130, bbox_inches='tight')
plt.show()

# ----------------------------------------------------------------------------
# 7) SALVAR no Drive — só se o val melhorou (gate); versionado
# ----------------------------------------------------------------------------
from pathlib import Path
best_ft = Path(ft.trainer.best)
if new_val is not None and new_val > (base_val or 0):
    dest    = f'{SAVE_DIR}/soja_yolo11s_finetuned.pt'      # o que o Space carrega
    dest_v1 = f'{SAVE_DIR}/soja_yolo11s_finetuned_v1.pt'   # cópia versionada
    shutil.copy(best_ft, dest); shutil.copy(best_ft, dest_v1)
    metrics = {'baseline': {'train': base_tr, 'val': base_val, 'test': base_test},
               'finetuned': {'train': new_tr, 'val': new_val, 'test': new_test},
               'counts': summary, 'freeze': FREEZE}
    json.dump(metrics, open(f'{SAVE_DIR}/vigil_metrics.json', 'w'), indent=2, default=float)
    print(f'\n✅ val melhorou ({base_val:.1%} -> {new_val:.1%}) — salvo em:')
    print(f'   {dest}\n   {dest_v1}\n   + vigil_metrics.json + vigil_matriz_confusao.png')
    if new_test is not None:
        print(f'   test honesto: {base_test:.1%} -> {new_test:.1%}')
    print('\nPróximo: subir o .pt na raiz do HF Space soja-inspection-api (rebuild).')
else:
    print(f'\n❌ val NÃO melhorou ({base_val:.1%} -> {new_val:.1%}) — não salvei.')
    print('   Tente freeze=8, mais fotos na classe fraca, ou confirme o recorte.')
