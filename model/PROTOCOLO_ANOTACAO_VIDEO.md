# Protocolo de anotação — detecção em vídeo (Vígil.ia)

> Objetivo: sair de **classificação (1 grão, foto)** para **detecção (vários grãos, vídeo)**.
> Um detector acha **e** classifica cada grão num passo só — sem o recorte Otsu por fora,
> que quebra com vários grãos / fundo não-uniforme.
>
> Este documento é a parte **manual** (captura + anotação). O treino comparativo
> (YOLO11-detect vs RT-DETR) já está pronto em `model/treino_deteccao_video.ipynb`.

---

## 0. Por que isso e não "só trocar o backbone"

O vídeo está fraco **não** por causa da arquitetura, e sim porque 100% do treino atual
é **foto estática, fundo preto, 1 grão**. Vídeo é luz variável, distância, leve borrão e
**vários grãos juntos**. Nenhuma troca de modelo conserta isso — o que conserta é
**dado do domínio de vídeo**. Por isso o passo crítico é capturar e anotar frames reais.

RT-DETR (transformer) **não** "precisa de menos dado" — a família DETR é mais faminta por
dado que CNN. A vantagem dele é detecção em tempo real. Por isso vamos **comparar** RT-DETR
com YOLO11-detect no mesmo dataset e deixar o mAP decidir, em vez de apostar no hype.

---

## 1. Captura dos vídeos/fotos (imitando a esteira)

Replicar o ambiente de produção o máximo possível:

- **Fundo:** preto fosco (cartolina/EVA preto), igual ao dataset original.
- **Luz:** difusa, vinda de **cima** (evitar sombra dura e reflexo no grão).
- **Distância:** a **mesma** que a câmera vai ficar na esteira (ex.: ~10–20 cm). Variar um pouco.
- **Densidade:** grãos espalhados, **vários no quadro** (10–40 por quadro), alguns encostando —
  é justamente isso que o detector precisa aprender (o Otsu não dava conta disso).
- **Variedade:** misturar as 5 classes no mesmo quadro. Gravar lotes "premium" (quase só intacto)
  e lotes "ruins" (vários defeitos) — reflete o caso real de QC.
- **Movimento:** se possível, mover a bandeja devagar (simula esteira) — captura borrão real.

### Quantidade-alvo (mínimo viável → bom)
| Item | Mínimo | Bom |
|---|---|---|
| Frames anotados | ~150 | ~300–400 |
| Caixas (grãos) totais | ~2.000 | ~5.000+ |
| Caixas por classe (a menor) | ≥ 150 | ≥ 400 |

> Como cada frame tem dezenas de grãos, **150 frames já geram milhares de caixas**. É bem
> mais eficiente que a coleta de classificação (1 grão por foto).

### Extrair frames de vídeo (sem quase-duplicatas)
Grave 3–5 vídeos curtos (20–40 s) e extraia 1 frame a cada ~0,5–1 s:
```bash
# 1 frame a cada 15 (≈ a cada 0.5s num vídeo 30fps)
ffmpeg -i lote01.mp4 -vf "select=not(mod(n\,15))" -vsync vfr frames/lote01_%04d.jpg
```
Frames muito parecidos não ajudam (e ainda vazam entre train/val) — por isso o subsampling.

---

## 2. Anotação no Roboflow

1. Criar projeto **Object Detection** (não Classification).
2. Classes — usar **exatamente** estes nomes (minúsculas, com hífen):
   ```
   broken
   immature
   intact
   skin-damaged
   spotted
   ```
   ⚠️ `intact` precisa ser idêntico — a regra de "Premium" no front depende dessa string.
3. Subir os frames. Desenhar **uma caixa por grão**, justa (sem sobra grande de fundo).
   - Usar o **Label Assist / SAM** do Roboflow pra acelerar (clica no grão, ele fecha a caixa).
   - Grão cortado na borda do quadro: anotar mesmo assim a parte visível.
   - Grãos encostados: uma caixa **para cada**, mesmo que se sobreponham.
4. **Split:** 70% train / 20% valid / 10% test. Garantir que frames do **mesmo vídeo**
   não fiquem espalhados entre train e val (senão a métrica infla). Se o Roboflow não
   separar por vídeo, dá pra subir os vídeos em "batches" e dividir por batch.

### Augmentation (no Roboflow, leve — o A100 aguenta treinar mais)
- Flip horizontal ✅ / vertical ✅ (grão não tem orientação canônica)
- Rotação ±15°, brilho ±20%, leve blur (simula vídeo) ✅
- **Mosaic já vem do treino (Ultralytics)** — não precisa duplicar no Roboflow.
- Evitar crop agressivo que corte o grão ao meio sem necessidade.

---

## 3. Exportar pro treino

Duas opções (o notebook aceita as duas):

**A) Via API do Roboflow (recomendado)** — copiar o snippet "YOLOv11" do botão *Export*:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="SUA_KEY")
ds = rf.workspace("...").project("...").version(N).download("yolov11")
# -> gera um data.yaml + train/valid/test
```

**B) Via Google Drive** — *Export → format YOLOv11 → download zip*, subir o zip pro Drive,
e apontar o caminho no notebook. Igual fizemos com o dataset de classificação.

O export já vem com `data.yaml` no formato de detecção:
```yaml
train: ../train/images
val:   ../valid/images
test:  ../test/images
nc: 5
names: [broken, immature, intact, skin-damaged, spotted]
```

---

## 4. Próximo passo (automático)

Com o dataset pronto, abrir **`model/treino_deteccao_video.ipynb`** no Colab (A100):
treina YOLO11-detect e RT-DETR, compara mAP50-95 / por-classe / velocidade, e exporta
o vencedor (`.pt` + `.onnx`). Aí integramos no app — provavelmente servidor primeiro
(detecção em vídeo no browser é mais pesada que classificação), com a UI de QC premium
contando intacto vs fora-do-padrão sobre as caixas detectadas.
