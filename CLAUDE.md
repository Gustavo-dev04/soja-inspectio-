# CLAUDE.md — Classificador de Grãos de Soja (Demo)

> Arquivo de contexto para o Claude Code. Leia tudo antes de gerar código.
> Resume o estado atual do projeto, as decisões já fechadas (com os porquês) e o que falta fazer.

---

## 1. O que é o projeto AGORA

Demo de **classificação de grãos de soja por imagem**, feita em casa, usando a **câmera do celular**.

Objetivo imediato e único: **levar para o professor da FATEC e provar que o modelo reconhece cada classe de grão.** Tirar uma foto de grãos espalhados → o sistema marca cada grão com sua classe na tela.

**NÃO é** (ainda) um sistema industrial. Esquecer, por enquanto:
- esteira transportadora
- sensor NIR
- soprador pneumático / ejeção
- Jetson / hardware de borda
- integração com qualquer empresa

Esses itens são evolução futura, não fazem parte da demo. Não introduzir essa complexidade no código da demo.

---

## 2. O resultado esperado da demo

Fluxo, do começo ao fim:

```
foto do celular (vários grãos, fundo escuro)
   → segmentação (recorta cada grão)
   → cada recorte vai pro classificador
   → modelo devolve a classe
   → desenha caixa + rótulo sobre cada grão na foto original
   → tela final que o professor vê
```

As **5 classes** (rótulos exatos, em inglês, como no dataset):
`intact` · `immature` · `broken` · `skin-damaged` · `spotted`

(PT para a UI: Intacto · Imaturo · Quebrado · Casca danificada · Manchado)

---

## 3. Stack — DECIDIDO, não relitigar

| Camada | Escolha | Por quê |
|---|---|---|
| Framework de treino | **Ultralytics / PyTorch** | Substituiu Keras/TensorFlow após experimento controlado (ver §9). Em igualdade de condições o YOLO11s-cls venceu o EfficientNet-B0. |
| Modelo | **YOLO11s-cls via transfer learning** | Variante de **classificação** do YOLO11 (não detecção — dataset é por pastas, sem caixas). Em experimento controlado bateu o EfficientNet-B0 no domínio real (91,7% vs 75%). Trocar só a cabeça para 5 classes. |
| Segmentação (recorte) | **OpenCV clássico** (grayscale → threshold → findContours → bounding box → resize 224×224) | NÃO usar detector treinado na demo. O dataset original foi gerado exatamente assim (>98% de acerto de segmentação) e o fundo escuro torna o threshold trivial. O recorte alimenta o classificador YOLO11s-cls. |
| Treino | **Google Colab** (GPU grátis) | Sem custo, sem setup local. |
| App / interface | **Gradio** | Abre no navegador do celular; botão de foto; mostra imagem anotada. |
| Deploy | **Hugging Face Spaces** | O dono já usa. Vira um link compartilhável. |

---

## 4. O dataset (ATUALIZADO — dataset real no Drive)

- **Nome:** SoyaBeans Classifications v2 (Roboflow export)
- **Fonte:** Roboflow Universe — hansaka-sudusinghe  
  `https://universe.roboflow.com/hansaka-sudusinghe/soyabeans-classifications-yjxdd`
- **Licença:** MIT (uso comercial liberado)
- **Tamanho:** 12.528 imagens (com augmentation Roboflow: flip, rotação, crop, shear)
- **Dimensão:** 400×400 px (redimensionado para 224×224 no pipeline Keras)
- **Splits prontos:** `train/` + `valid/` + `test/`
- **Localização no Drive:** `Meu Drive/SoyaBeans Classifications.v2i.folder (Unzipped Files)/`
- **Drive ID do zip original:** `1NsDtGLDwzPftRrmeRbmbokkp0gsQgJge`

### Estrutura de pastas (dentro de cada split):
```
train/
  Broken soybeans/          ← índice 0
  Immature soybeans/        ← índice 1
  Intact soybeans/          ← índice 2
  Skin-damaged soybeans/    ← índice 3
  Spotted soybeans/         ← índice 4
  Part of the original soybean images/   ← IGNORAR (excluído via class_names explícito)
```

⚠️ **Importante:** Ao usar `image_dataset_from_directory`, sempre passar `class_names` explicitamente
para excluir o folder "Part of the original soybean images". Ver `model/train.ipynb` Célula 3.

---

## 5. O ponto crítico: domain shift

O modelo aprende com câmera do dataset. A foto do celular do dono vai parecer diferente.

**Mitigações (obrigatórias):**
1. **Imitar o setup do dataset:** fundo **preto fosco**, grãos **sem se encostar**, **luz difusa de cima**.
2. **Validar com os próprios grãos:** tirar 10–15 fotos por classe antes do dia do professor.
3. **Fine-tuning leve** com fotos próprias se o acerto cair.

---

## 6. Ordem de construção (plano de risco mínimo)

1. **Treino** do YOLO11s-cls no Colab (`model/train_yolo.ipynb`) → salvar `soja_yolo11s_best.pt`
2. **Fine-tuning no domínio real** (`model/finetune_yolo.ipynb`) → `soja_yolo11s_finetuned.pt`
3. **Versão "1 grão"** primeiro — classificação pura, fallback garantido para a demo.
4. **Versão multi-grão** — segmentação OpenCV por cima.
5. **Deploy HF Spaces** — `model/app.py` + `model/requirements.txt` + modelo `.pt`

---

## 7. Arquivos do projeto

| Arquivo | Propósito |
|---|---|
| `model/train_yolo.ipynb` | **(produção)** Treina o YOLO11s-cls no Roboflow, avalia, salva `soja_yolo11s_best.pt` |
| `model/finetune_yolo.ipynb` | **(produção)** Fine-tuning do YOLO no domínio real (fotos do celular) → `soja_yolo11s_finetuned.pt` |
| `model/train.ipynb` | (histórico) Treino do EfficientNet-B0 — base do experimento comparativo |
| `model/finetune.ipynb` | (histórico) Fine-tuning do EfficientNet com holdout + safety gate |
| `model/finetune_eff_fair.ipynb` | (experimento) EfficientNet em condições idênticas ao YOLO — prova arquitetura vs receita |
| `model/app.py` | Gradio app: modo 1 grão + modo multi-grão com OpenCV — **carrega o `.pt` do YOLO** |
| `model/requirements.txt` | Dependências para HF Spaces (Ultralytics, não TensorFlow) |
| `model/soja_classes.json` | Lista de classes na ordem correta (índice = folder alfabético) |

**Para deploy no HF Space, enviar:**
- `model/app.py` → renomear para `app.py` na raiz do Space
- `model/requirements.txt` → `requirements.txt` na raiz do Space
- `soja_yolo11s_finetuned.pt` → gerado pelo Colab (`finetune_yolo.ipynb` FT-5), na raiz do Space
- `soja_classes.json` → na raiz do Space

---

## 8. Decisões fechadas (resumo para não reabrir)

- ✅ Ultralytics/PyTorch (YOLO11s-cls) — **substituiu** Keras/EfficientNet após experimento controlado (§9)
- ✅ YOLO11s-cls + transfer learning (não ML clássico como motor)
- ✅ OpenCV para recorte (segmentação clássica; o YOLO é só classificador, não detector)
- ✅ Gradio + HF Spaces (não FastAPI/Next.js nesta fase)
- ✅ Dataset Roboflow, 5 classes, 12.528 imagens, 400×400 → resize 224×224
- ✅ Fine-tuning no domínio real (fotos do celular) é obrigatório — sem ele, domain shift derruba para ~3-8%
- ✅ Foco só na demo de classificação — sem esteira/NIR/sopro/Jetson
- ✅ Fotografar imitando fundo escuro + luz de cima
- ✅ Validar com grãos próprios; fine-tuning como fallback

---

## 9. Experimento EfficientNet vs YOLO11s-cls (por que trocamos)

Comparação empírica, mesmas 57 fotos reais, mesmo crop OpenCV, mesmo split (val = 12 fotos),
augmentation equivalente. Mede acurácia no **domínio real** (fotos do celular, fundo cinza):

| Modelo | Val real | Gap train-val |
|---|---|---|
| EfficientNet — aug branda + mix Roboflow (run antiga) | 64% | 25% (overfitting) |
| EfficientNet — aug forte + freeze parcial (justo) | 75% | 7,2% |
| **YOLO11s-cls — aug forte (produção)** | **91,7%** | **3,9%** |

**Conclusões:**
1. A **receita** explica ~11pp (64→75): a run antiga do EfficientNet tinha `lr=1e-5` (baixo demais),
   misturava 60% de dados do Roboflow (diluía a adaptação) e media num holdout diferente.
2. A **arquitetura** explica outros ~17pp (75→91,7): em igualdade total, o YOLO ainda venceu.
   Atribuído ao pipeline AdamW+weight-decay+EMA do Ultralytics e ao bloco de atenção espacial
   (C2PSA) do YOLO11s, que ajudam a generalizar com pouca foto e ignorar o fundo.
3. **Domain shift é o gargalo real:** sem fine-tuning, o modelo (treinado em fundo preto) cai para
   ~3-8% nas fotos de fundo cinza. O fine-tuning no domínio real recupera para 91,7%.
