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
| Framework de treino | **TensorFlow / Keras** | Decisão do dono do projeto. |
| Modelo | **EfficientNet-B0 via transfer learning** | Dataset é pequeno/médio e balanceado; transfer learning é o caminho mais simples E mais confiável aqui. Trocar só a cabeça de classificação para 5 classes. |
| Segmentação (recorte) | **OpenCV clássico** (grayscale → threshold → findContours → bounding box → resize 224×224) | NÃO usar YOLO/detector treinado na demo. O dataset original foi gerado exatamente assim (>98% de acerto de segmentação) e o fundo escuro torna o threshold trivial. |
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

1. **Treino** do EfficientNet-B0 no Colab (`model/train.ipynb`) → salvar `soja_model_final.keras`
2. **Versão "1 grão"** primeiro — classificação pura, fallback garantido para a demo.
3. **Versão multi-grão** — segmentação OpenCV por cima.
4. **Deploy HF Spaces** — `model/app.py` + `model/requirements.txt` + modelo `.keras`
5. **Validação** com grãos próprios → fine-tuning se necessário.

---

## 7. Arquivos do projeto

| Arquivo | Propósito |
|---|---|
| `model/train.ipynb` | Notebook Colab completo: carrega dataset, treina EfficientNet-B0, avalia, salva |
| `model/app.py` | Gradio app: modo 1 grão + modo multi-grão com OpenCV |
| `model/requirements.txt` | Dependências para HF Spaces |
| `model/soja_classes.json` | Lista de classes na ordem correta (índice = folder alfabético) |

**Para deploy no HF Space, enviar:**
- `model/app.py` → renomear para `app.py` na raiz do Space
- `model/requirements.txt` → `requirements.txt` na raiz do Space
- `soja_model_final.keras` → gerado pelo Colab, adicionar na raiz do Space
- `soja_classes.json` → gerado pelo Colab (ou usar o do repo), na raiz do Space

---

## 8. Decisões fechadas (resumo para não reabrir)

- ✅ Keras/TensorFlow (não PyTorch)
- ✅ EfficientNet-B0 + transfer learning (não ML clássico como motor)
- ✅ OpenCV para recorte (não YOLO na demo)
- ✅ Gradio + HF Spaces (não FastAPI/Next.js nesta fase)
- ✅ Dataset Roboflow, 5 classes, 12.528 imagens, 400×400 → resize 224×224
- ✅ Foco só na demo de classificação — sem esteira/NIR/sopro/Jetson
- ✅ Fotografar imitando fundo escuro + luz de cima
- ✅ Validar com grãos próprios; fine-tuning como fallback
