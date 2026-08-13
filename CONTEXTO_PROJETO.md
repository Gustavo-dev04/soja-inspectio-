# Vígil.ia — Contexto completo do projeto

> **Propósito deste arquivo:** reunir em um só lugar tudo o que existe hoje no
> repositório — decisões, experimentos, resultados, arquitetura, apps e histórico —
> para servir de fonte única na hora de escrever a documentação final (relatório
> pro professor, PDF, slides etc). Não é a documentação em si; é o material bruto
> organizado.
>
> **Como este arquivo foi montado:** lendo `CLAUDE.md`, todo o histórico de commits
> (68 no branch atual), os notebooks e scripts de `model/`, o app do Steam Deck
> (`deck/`), o backend (`backend/`), o frontend (`frontend/`) e os documentos
> `HANDOFF.md` / `docs/ESPECIFICACAO.md` / `ROADMAP.md` / `README.md`.
>
> ⚠️ **Aviso importante:** `ROADMAP.md`, `docs/ESPECIFICACAO.md`, `docs/README.md`
> e `README.md` descrevem uma versão **anterior e diferente** do projeto (rótulos
> `soja_boa`/`soja_verde`/etc, modelos "Sinnet"/"Magnus", EfficientNet, Gradio,
> roadmap industrial com esteira/NIR/Jetson). O `CLAUDE.md` já deixa claro que o
> escopo atual da demo **não inclui** esteira/NIR/sopro/Jetson. Este arquivo
> descreve o estado **real e atual** do código, que é ainda mais recente que o
> `CLAUDE.md`: o projeto pivotou de classificação-de-1-grão para
> **detecção-multi-grão-em-vídeo** nos últimos ~50 commits. Trate os documentos
> antigos como histórico, não como verdade atual.

---

## 1. O que o projeto é, na prática, hoje

O Vígil.ia tem **duas frentes de produto que coexistem**, nascidas em momentos
diferentes do projeto:

1. **Modo foto / classificação de 1 grão** (mais antigo, em produção real,
   web pública) — o usuário fotografa **um grão**, o sistema classifica em uma
   das 5 classes. É o que está no ar em `Guguinhaxd/soja-inspection-api` (HF
   Spaces) + Vercel.
2. **Modo vídeo / detecção multi-grão** (mais recente, em fase de demo/validação)
   — vários grãos no mesmo quadro, ao vivo, com caixa + classe por grão e
   veredito travado por rastreamento. Roda local (Steam Deck) ou via GPU do
   Colab (túnel Cloudflare), porque o modelo campeão (YOLO11x ou RT-DETR-l) é
   pesado demais para CPU/edge.

As **5 classes** são as mesmas nas duas frentes (rótulos do dataset, em inglês):
`intact` · `immature` · `broken` · `skin-damaged` · `spotted`
(PT: Intacto · Imaturo · Quebrado · Casca danificada · Manchado)

---

## 2. Linha do tempo — como o projeto evoluiu (3 eras)

### Era 1 — EfficientNet-B0 + Gradio (histórico, arquivada)
- Framework: TensorFlow/Keras, modelo `soja_model_final.keras` (29 MB, Git LFS)
- Segmentação: OpenCV (Otsu) recorta 1 grão
- App: Gradio, deploy Hugging Face Spaces (`Guguinhaxd/soja-inspection`, SDK Gradio)
- Feedback loop: usuário corrige → salva em dataset HF `Guguinhaxd/soja-correction`
- Resultado real observado: **~29% de acurácia** em fotos reais (domain shift
  severo — o modelo praticamente só respondia "Quebrado"). Um fine-tuning com
  57 correções levou a 64%, mas com overfitting (treino 89% / val 64%).
- Arquivos que ainda existem no repo desse período: `model/train.ipynb`,
  `model/finetune.ipynb`, `gerar_relatorio.py`, `gerar_versao.py`.

### Era 2 — YOLO11s-cls + FastAPI/Next.js (é o que está em produção no site hoje)
- Migração motivada por um **experimento controlado**: mesmas 57 fotos reais,
  mesmo recorte, mesmo split → YOLO11s-cls bateu o EfficientNet-B0 mesmo em
  igualdade de condições (ver §5).
- Arquitetura separada em 3 peças (limite de 250MB da Vercel não comporta
  torch/ultralytics):
  - **Frontend** Next.js 14 na Vercel (UI + `/api/explain`, que chama Groq/Llama
    3.3 70B — não usa Python)
  - **Backend de visão** FastAPI + YOLO11s-cls + OpenCV, em Docker no Hugging
    Face Spaces (`Guguinhaxd/soja-inspection-api`)
  - **Banco** Supabase (tabelas `inspecoes`, `lotes`)
- Modelo em produção: `soja_yolo11s_finetuned.pt`, fine-tunado no domínio real
  (fotos de celular). Acurácia observada: **91,7%** (mas em val de só 12 fotos —
  ver limitações, §9).
- Arquivos-chave: `backend/main.py`, `backend/inference.py`,
  `backend/database.py`, `model/train_yolo.ipynb`, `model/finetune_yolo.ipynb`.
- Documentado em detalhe (embora hoje desatualizado no resto) em `HANDOFF.md`.

### Era 3 — Detecção multi-grão em vídeo (a mais recente, ~50 commits)
Motivação: o cenário real de inspeção (esteira/lote) tem **vários grãos por
quadro**, e a segmentação Otsu clássica não escala pra isso (exige fundo
controlado e 1 grão por imagem). A solução é trocar classificação por
**detecção**: um modelo só que acha *e* classifica cada grão numa passada.

Isso disparou uma sequência grande de experimentos, documentados em detalhe em
`model/PROTOCOLO_ANOTACAO_VIDEO.md` e `model/COMPARATIVO_YOLO11S_VS_RTDETR.md`
(resumidos na íntegra na seção 5 abaixo). Produtos que nasceram dessa era:
- `deck/vigil_deck.py` — app local de inferência ao vivo no Steam Deck (sem
  internet, sem servidor)
- `model/demo_servidor_colab.ipynb` + `model/testar_gpu_colab.py` — servidor
  FastAPI rodando no Colab (GPU) exposto via túnel Cloudflare, para rodar os
  modelos pesados (YOLO11x, RT-DETR-l) a partir da câmera do celular sem
  precisar de GPU local
- `frontend/src/app/ao-vivo/page.tsx` — modo ao vivo **no navegador**, via
  ONNX Runtime Web (inferência client-side, sem servidor) — ver §7
- `model/aprendizado_ativo.ipynb` — loop de active learning (humano corrige →
  modelo reaprende)
- `model/autotreino_video_v4.ipynb` — auto-treino: vídeos brutos rotulados
  automaticamente pelo modelo campeão (RT-DETR/YOLO11x como "professor"),
  organizados por pasta = classe

---

## 3. Arquitetura atual — visão geral das peças

```
                         ┌─────────────────────────┐
                         │   Frontend Next.js 14    │  Vercel
                         │   (site público)          │
                         └───────┬─────────┬─────────┘
                                 │         │
                   modo foto     │         │  modo ao-vivo (navegador)
                   POST /inspect │         │  ONNX Runtime Web (client-side,
                                 │         │  sem servidor — ver §7)
                                 ▼         ▼
                    ┌──────────────────┐  (roda 100% no browser do usuário)
                    │ Backend FastAPI   │
                    │ YOLO11s-cls +     │  Hugging Face Spaces (Docker)
                    │ OpenCV            │  Guguinhaxd/soja-inspection-api
                    └────────┬──────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Supabase (Postgres)│  tabelas inspecoes, lotes
                    └──────────────────┘

  ─────────────────────────────────────────────────────────────────
  Frente separada: detecção multi-grão / vídeo (mais pesada, fora do
  fluxo web público, usada em demo/validação)

  Steam Deck (deck/vigil_deck.py)          Celular (câmera local)
  100% local, sem internet                       │
  carrega .pt direto, OpenVINO no iGPU            │ (DroidCam / USB)
                                                   ▼
                                    testar_gpu_colab.py (PC local)
                                                   │ POST /inspect (frame JPEG)
                                                   ▼
                              demo_servidor_colab.ipynb (Colab, GPU)
                              FastAPI + YOLO11x/RT-DETR + túnel Cloudflare
                              (mesmo contrato /inspect do backend de produção)
```

**Por que essa separação existe:** o modelo de detecção campeão (YOLO11x,
56,9M parâmetros) é pesado demais para CPU/edge e para o free tier do HF
Spaces. Ele só faz sentido com GPU. Como o Colab não enxerga a rede Wi-Fi
local do usuário, a solução foi inverter a conexão: o Colab expõe a API via
túnel Cloudflare (link público tipo `https://xxxx.trycloudflare.com`) e um
cliente local (`testar_gpu_colab.py`) lê a câmera do celular e manda cada
frame pra lá. O mesmo padrão é reaproveitado no site: `?api=<url-do-túnel>`
na Vercel troca o backend de visão sem redeploy (persistido em
`localStorage`, ver `frontend/src/lib/api.ts`).

---

## 4. As 5 classes e a regra de negócio

| Índice | Rótulo (dataset) | Rótulo (UI) | Observação |
|---|---|---|---|
| 0 | `broken` | Quebrado | |
| 1 | `immature` | Imaturo | |
| 2 | `intact` | Intacto | única classe "Premium"; string usada literalmente no front |
| 3 | `skin-damaged` | Casca danificada | fronteira mais ambígua com `broken` |
| 4 | `spotted` | Manchado | classe onde os detectores mais erram no vídeo |

Regra de veredito no modo ao vivo (site e Steam Deck): grão é **Premium** só
se a classe for `intact` **e** a confiança bater um piso (0,80 no modo
navegador — `PREMIUM_CONF` em `ao-vivo/page.tsx`). Nos apps de vídeo, existe
adicionalmente um **voto exigente por classe** (`RATIOS` em
`vigil_deck.py`): um defeito só é aceito se acumular ≥ 75–85% dos votos
ponderados por confiança ao longo dos frames rastreados; caso contrário o
grão recebe benefício da dúvida e vira `intact`. Isso existe especificamente
para matar alucinação de defeito (falso positivo) em vídeo.

---

## 5. Modelos — o que foi testado, o que venceu, e por quê

### 5.1 Classificação (1 grão) — decisão já fechada, em produção

| Modelo | Acurácia (domínio real) | Gap treino-val |
|---|---|---|
| EfficientNet-B0 — receita antiga | 64,0% | 25,0% (overfitting) |
| EfficientNet-B0 — receita justa (mesmas condições do YOLO) | 75,0% | 7,2% |
| **YOLO11s-cls — produção** | **91,7%** | **3,9%** |

Conclusão do experimento controlado (mesmas 57 fotos reais, mesmo recorte
OpenCV, mesmo split, augmentation equivalente): a receita de treino explica
~11pp da diferença (LR baixo demais, mix de dados diluindo a adaptação); a
**arquitetura** explica outros ~17pp — atribuído ao AdamW+weight-decay+EMA do
Ultralytics e ao bloco de atenção espacial (C2PSA) do YOLO11s, que ajuda a
focar no grão e ignorar o fundo (justamente o problema de domain shift).
Modelo em produção: `soja_yolo11s_finetuned.pt`.

### 5.2 Detecção (multi-grão, vídeo) — a comparação mais extensa do projeto

Contexto: a hipótese inicial do dono era que **RT-DETR** (detector
transformer da Baidu) se sairia melhor que a família YOLO em vídeo. Em vez de
apostar no hype, o protocolo do projeto foi treinar os dois nas mesmas
condições e deixar o resultado decidir (`model/COMPARATIVO_YOLO11S_VS_RTDETR.md`).

**Dados usados (idênticos nos dois modelos):**
- Base: 12.528 imagens do dataset Roboflow (classificação), convertidas pra
  detecção via **pseudo-rótulo Otsu** (1 grão/imagem, fundo preto → caixa
  automática, zero anotação manual)
- Fine-tune v3: ~600 fotos reais balanceadas por oversampling (a coleta original
  tinha 122 `intact` vs 21–32 por classe de defeito) ×2 com motion blur
  sintético, + ~600 **cenas multi-grão sintéticas** (grãos recortados por
  segmentação de saturação, colados em fundos preto→cinza, 6–25 grãos por
  cena, com sobreposição/encosto entre grãos), 40% das cenas borradas
- Validação: 40 fotos reais

**Histórico de treinos (RT-DETR):**

| Rodada | Resultado | Observação |
|---|---|---|
| base_12k | mAP50 0,982 / mAP50-95 0,980 (val 12,5k) | salto grande no close_mosaic (ép. 41) |
| ft_real (1ª) | mAP 0 (falha) | bug: letterbox aplicado antes do Otsu → caixa virava a foto inteira |
| ft_real2 (2ª) | mAP 0 → colapso → corrigido → mAP50 ~0,79 | colapso causado pelo warmup padrão; corrigido com `amp=False, warmup_epochs=0` |
| ft_v3 | campeão da rodada | curou o "tudo vira broken" no vídeo borrado |
| ft_v3_disc (LR discriminativo) | pior que ft_v3 (resultado negativo, arquivado) | dataset já saturado — gargalo era dado, não otimização |

**Particularidades técnicas do RT-DETR** (relevantes se for retreinar): não
roda NMS nativo no Ultralytics 8.4.80 (só filtro de confiança — verificado no
código-fonte), precisou de patch de NMS agnóstico de classe; mosaic treina em
tela 2×imgsz por design; AMP pode gerar NaN (aviso do próprio Ultralytics);
warmup padrão (bias-lr 0,1) colapsa um modelo já convergido em dataset
pequeno.

**Validação em vídeo (o juiz de verdade):** mesmo vídeo real
(`teste_soja.mp4`, ~809 frames), mesmo pipeline (NMS agnóstico de classe, IoU
0,6, conf 0,35 + rastreamento ByteTrack + veredito travado por grão — trava
com ≥8 frames e ≥60% de consenso). RT-DETR ft_v3 rodou a **14,4 ms/frame
(~70 fps) na GPU**.

**Cara a cara ft_v3 (RT-DETR) vs YOLO11s v3** *(avaliação visual, não
benchmark rotulado)*: RT-DETR teve menos alucinação e ~80-85% de acurácia
premium/não-premium estimada, vs ~65-75% do YOLO11s — mas esse comparativo
tinha uma **assimetria metodológica**: os YOLOs não tinham passado pelo
estágio base de 12,5k imagens que o RT-DETR teve.

**ADENDO decisivo — tira-teima de capacidade** (`tira_teima_capacidade.ipynb`):
corrigindo a assimetria (YOLO11l e YOLO11x treinados pelo mesmo caminho de 2
estágios: COCO → base 12,5k → fine-tune v3):

| Modelo | Params | GFLOPs | Veredito no vídeo *(estimativa visual)* |
|---|---|---|---|
| YOLO11s (sem estágio base) | 9,4 M | 21,5 | ainda atrás (~65-75%) |
| YOLO11l | 25,3 M | 86,9 | ≈ RT-DETR; alucina mais, porém melhor em `spotted` |
| RT-DETR-l | 32,0 M | 103,5 | ~80-85% |
| **YOLO11x** | **56,9 M** | **194,9** | **~95% — novo campeão** |

**Conclusão final:** o gap não era de arquitetura, era de **capacidade +
caminho de dados**. Em capacidade pareada (11l vs RT-DETR-l), empatam com
perfis de erro diferentes; com ~2× a capacidade, a CNN com atenção (C2PSA)
supera o transformer, além de ter engenharia mais simples (NMS nativo, treino
estável, sem risco de NaN). A tese "transformer generaliza melhor com dado
escasso" fica restrita ao regime de capacidade igual/menor.

**Papéis atuais dos modelos:**

| Papel | Modelo | Status |
|---|---|---|
| Modo foto (produção, web) | YOLO11s-cls (`soja_yolo11s_finetuned.pt`) | no ar, intocado |
| Vídeo / demo (GPU) | **YOLO11x_v3** (`soja_yolo11x_v3.pt`, com fine-tune multi-grão `soja_yolo11x_multi_v3.pt`) | campeão atual; servido via Colab+túnel |
| Reserva / segunda opinião | RT-DETR-l ft_v3 | arquivado no Drive |
| **Candidato a edge (Jetson Orin Nano)** | **RF-DETR — FT4 (fonte de recorte por classe)**. Small/512 validado no aparelho (53,2 qps); migrando pra **Large/704** por vazão (§5.4) | ver §5.3-5.4; engine FP16 rodando no Orin Nano, app ao vivo funcionando |
| Local / edge (futuro, CPU/iGPU) | YOLO11s ou 11m via destilação | professor da destilação passa a ser o YOLO11x |
| Steam Deck (local, sem GPU) | `soja_yolo11n_base12k_v2.pt` (nano) | roda em CPU/iGPU, ~10-20 fps |

**11m testado e descartado por ora:** treinado do zero sem ajuste de receita
(mais épocas/LR que o 11m precisaria), teve desempenho pior que o 11s no
mesmo teste. Decisão prática: 11s pra uso local, 11x pra servidor GPU — 11m
fica em standby até valer a pena reabrir com receita ajustada.

**Próximo passo já planejado:** `autotreino_video_v4.ipynb` usa vídeos
organizados por pasta (`treino/<classe>/`, luz normal + flash) rotulados
automaticamente pelo YOLO11x (como "professor" de caixas — a classe vem da
pasta, não do modelo), pra treinar um modelo leve (destilação) que rode bem
em edge/CPU sem perder tanta qualidade.

### 5.3 RF-DETR Small — teste dedicado pro Jetson Orin Nano

Missão separada: os campeões de vídeo (RT-DETR-l, YOLO11x) são modelos de GPU
de servidor, inviáveis num Jetson Orin Nano (8GB, ~67 TOPS INT8). O RF-DETR
(Roboflow, mesma família DETR do RT-DETR, mas backbone DINOv2 com NAS de
arquitetura) publica variantes Nano/Small/Medium com poucos ms de latência na
T4 via TensorRT — candidato natural a edge. Testado o **Small** (32,1M
params, patch 16, resolução 512/672) pelo mesmo caminho: COCO → base 12,5k →
fine-tune no domínio real (`model/treino_rfdetr_small_completo.ipynb`).

Sequência de estágios testada, todos julgados no mesmo `teste_soja.mp4`:

1. **Estágio 1** (base 12,5k, pseudo-rótulo Otsu): saturou rápido (mAP50-95
   0,985 no val) — mesmo padrão de tarefa fácil já visto no RT-DETR base_12k.
2. **FT1** (fotos reais + vídeo de defeito + cenas sintéticas): mAP 0,826,
   perfil saudável em todas as classes; no vídeo, forte em intacto (~84%) mas
   fraco em achar defeito.
3. **FT2** (só capturas do `vigil_deck`, 100% cena sintética): mAP 0,375,
   classes colapsadas — **catastrophic forgetting**, perdeu no vídeo até pro
   FT1 sozinho. Cena sintética feita de recorte de captura já recortada
   (recorte de recorte, fundo gerado) é dado mais artificial que o do FT1.
4. **FT3** (capturas + 30% *experience replay* do FT1, val misto das duas
   fontes — mesma técnica de replay 70/30 usada na era EfficientNet, ver
   `gerar_relatorio.py` §7-8): **melhor dos dois mundos no vídeo** — resolveu
   o esquecimento do FT2 sem perder o que o FT1 sabia; caixas boas também
   *(avaliação visual do dono)*.
5. **FT4** (fonte do recorte escolhida **por classe** nas cenas sintéticas —
   `broken`/`skin-damaged` da foto real, `spotted`/`immature` da captura,
   `intact` meio a meio): **campeão no vídeo** — parou de confundir `immature`
   com `intact` e manteve as outras classes bem. **Mas o val expõe uma
   ressalva importante:** `immature` ficou com **recall 0,041** (AP 0,147) e
   `spotted` com AP 0,299, enquanto `broken` 0,859 / `intact` 0,810 /
   `skin-damaged` 0,674. Recall 4% significa que o modelo *parou de prever*
   `immature`, não que aprendeu a distinguir. **O `teste_soja.mp4` não tem
   nenhum grão imaturo** (confirmado), então o vídeo não valida essa classe —
   decisão consciente: o FT4 é o melhor ponto **para o dado de hoje**, e
   `immature` vira backlog. Se um lote com imaturo entrar na linha, este
   modelo deixa passar. Causa: só **56 recortes únicos** de `immature` e
   **35** de `spotted` nas capturas (contra 122/classe nas fotos reais),
   oversampled 7× e 11× pro pool. O gargalo agora é **quantidade de grão
   real distinto**, não receita.

**O achado mais útil da sequência** veio do FT4, e não é o que parecia: a
melhora não foi por qualidade de *imagem*, foi por qualidade de **rótulo**. As
fotos de `immature` do FT1 estavam mal anotadas e ensinavam o modelo que
"imaturo se parece com intacto"; trocar a fonte dessa classe removeu o
contra-exemplo errado do treino. Isso é um **contorno, não uma correção** — o
dado ruim continua na pasta de origem e volta a atrapalhar em qualquer receita
que a use. Pode haver contaminação parecida em outras classes ainda não
isoladas; o fluxo pra caçar isso já existe no projeto
(`model/aprendizado_ativo.ipynb`: rodar o campeão sobre as fotos, revisar onde
ele discorda da pasta com confiança alta, corrigir e re-treinar).

**Decidido não corrigir o rótulo de `immature`, e sim capturar grão novo:** o
erro ali foi sistemático (as fotos não são de grãos imaturos), então relabelar
só as moveria pra `intact` — que já sobra — sem devolver nada a `immature`.
Correção de rótulo recupera dado quando o rótulo está *trocado*, não quando o
objeto fotografado é outro. Alvo da coleta: ~120 grãos únicos de `immature` e
~120 de `spotted` (hoje: 56 e 35), o que iguala as classes fortes e elimina o
oversampling de 7×/11× do pool.

Dois resultados negativos registrados no caminho (útil pra não repetir):
balancear o **train** por classe (não só o val) viciou o modelo pró-defeito
num vídeo majoritariamente intacto; e treinar o fine-tune final só nas
capturas, sem nenhum replay do estágio anterior, apaga o que já tinha sido
aprendido. Histórico completo, incluindo bugs de medição no caminho (off-by-one
de classe, tracker depreciado) em `model/COMPARATIVO_YOLO11S_VS_RTDETR.md` §11.

**Validado no aparelho:** engine TensorRT FP16 construída no Orin Nano roda a
**53,2 qps** (18,7 ms de compute) — cabe em tempo real com folga, sem precisar
de INT8, do RF-DETR Nano ou de DeepStream. App ao vivo funcionando
(`jetson/vigil_jetson.py`), forte em multi-grão e fraco em grão solto, o que é
consequência direta do dataset (91% das caixas vêm de cena densa) e está
alinhado com o uso real.

### ⚠️ Avaliação de acurácia está SUSPENSA até haver padrão de captura

Decisão do dono, e é a leitura certa: o setup atual do Jetson é **outro domínio**
em relação ao que treinou o modelo (câmera, luz, fundo e distância diferentes das
capturas do celular). Medir "ele inspeciona bem?" agora produz número que não se
transfere — o mesmo domain shift que é o gargalo estrutural do projeto desde a
era EfficientNet (29% → 64% → 91,7%).

Separação prática:

| mensurável agora (independe de domínio) | só depois da padronização |
|---|---|
| fps, latência, uso de memória | acurácia, recall por classe |
| estabilidade de caixa e de tracking | confusão entre classes |
| integração (engine, câmera, app) | calibragem de `RATIOS` e `conf` |

**Isso reordena o backlog.** Coletar os ~120 grãos de `immature` e `spotted`
**não deve acontecer antes** da padronização: dado capturado no rig atual nasce
num domínio que será descartado.

### ✅ O rig padrão foi definido (v1) — ver `jetson/PADRAO_CAPTURA.md`

| Item | Escolha |
|---|---|
| Câmera | IMX219 — 8 MP, lente 120°, foco ajustável, CSI |
| Resolução de trabalho | 3280×2464 @ 21 fps (sensor cheio, para o ROI 1:1) |
| Iluminação | Ring light 6500 K |
| Fundo | Cartolina preta fosca, câmara fechada |
| Exposição / WB | **TRAVADOS** via `nvarguscamerasrc` (`aelock`, `awblock`, `wbmode=0`) |
| Enquadramento | **Recorte quadrado central** (`--quadrado`) |

Três decisões técnicas com motivo, que não devem ser desfeitas sem pensar:

1. **AE/AWB travados** — em automático a câmera compensa sozinha entre sessões e
   recria o domain shift que o rig existe pra eliminar. Nada falha; o dado só
   fica inconsistente. É o modo mais silencioso de invalidar um dataset.
2. **Recorte quadrado central** — a lente de 120° distorce as bordas (barril), e
   o modelo come uma entrada quadrada. O recorte descarta a região distorcida *e*
   elimina o letterbox, que gastava 25% da entrada em barra preta (medido).
3. **ROI 1:1 no sensor cheio, não downscale** — a 120° e 15 cm o quadro inteiro
   reduzido até a entrada espreme o grão para ~16 px, contra os 60-150 px do
   treino. Recortando `--roi 704` no centro dos 3280×2464, **sem reescalar**, o
   grão chega com ~55 px. Nunca usar os modos 1080p: eles **recortam o sensor** e
   mudam o enquadramento entre sessões.

**Consequência a assumir:** este rig é um **domínio novo**. Os modelos atuais
(FT4 e anteriores) vão degradar nele — esperado, não é defeito. Agora o caminho
é: capturar dataset neste rig → re-treinar → recalibrar `RATIOS` e `conf`. Essa
passa a ser a rodada de fine-tuning que realmente conta.

Continua valendo, para quando a padronização existir: gravar também um **vídeo de
lote propositalmente ruim** (defeito conhecido), porque o vídeo de teste atual é
majoritariamente `intact` e não distingue "modelo bom" de "modelo viciado em
dizer intacto".

### 5.4 Subida de resolução: 512 → 704 (RF-DETR **Large**)

Com 53,2 qps sobrando no Jetson, a pergunta natural foi treinar em resolução
maior. A análise que decidiu o desenho:

**No ROI 1:1, a entrada do modelo não controla o detalhe do grão — controla a
área.** O grão em pixels é fixado pela óptica (7,9 px/mm × 7 mm ≈ 55 px), e o
recorte não reescala nada. Subir de 512 para 704 não deixa o grão mais nítido;
leva a janela útil de 65 mm para 89 mm, **~1,9× mais grãos por quadro**. Como o
objetivo declarado é **vazão**, é a alavanca certa.

Custo estimado: ~28 fps FP16 no Orin Nano (de 53,2) — ainda acima dos 21 fps que
a câmera entrega no modo cheio. Ou seja, **a área sai de graça**. Medir com
`jetson/bench_trt.sh` para confirmar.

**Como chegar a 704 — e como NÃO chegar.** Não se passa `resolution=` custom: o
caminho limpo é **trocar de variante**, porque o checkpoint pré-treinado de cada
variante foi treinado *na resolução nativa dela* — o positional embedding casa
exato, sem interpolação. N/S/M/L têm praticamente o mesmo tamanho (~30-34M
params, todas Apache 2.0) e o que muda entre elas é a resolução nativa: Nano 384,
Small 512, Medium 576, **Large 704**.

A "Large" é literalmente *a Small a 704 px* — conferido no `rfdetr/config.py` da
versão 1.9.2: `RFDETRLargeConfig` usa o mesmo encoder `dinov2_windowed_small` e o
mesmo `patch_size=16` / `num_windows=2` da Small, mudando só `resolution` (512 →
704) e `dec_layers` (3 → 4). Não confundir com a **`RFDETRLargeDeprecated`**, que
é outra classe (560 px, patch 14) — instanciar a errada era o risco do
roboflow/rf-detr#960. Na 1.9.2 esse issue já está corrigido (há
`interpolate_position_embeddings` e `_sync_pe_with_resolution`), mas a instalação
está pinada em `rfdetr[train,loggers]>=1.9` justamente para não depender disso.

Mudanças que isso exigiu no `model/treino_rfdetr_small_completo.ipynb`:

- `VARIANTE = 'large'`, `RES` derivado da variante, e `SIZE = RES` — o canvas do
  dataset sintético passa a ser igual à entrada, sem reescala intermediária;
- **a resolução entrou nos caminhos dos checkpoints** (`_P = …_{VARIANTE}_{SIZE}`).
  Cada estágio tem guarda "pula se o arquivo existe": sem o sufixo, rodar em
  resolução nova reaproveitaria em silêncio um `.pth` treinado em 512 e pularia o
  retreino inteiro — o experimento sairia inválido sem nenhum erro;
- a contagem de grãos por cena sintética passou a escalar com a **área**
  (antes fixa em 6-25), senão a cena de 704 fica esparsa justamente no eixo que
  se quer treinar.

**O que a resolução NÃO resolve.** O gargalo de acurácia continua sendo domain
shift + rótulo ruim: o `immature` do FT1 está sistematicamente errado (as fotos
não são de grãos imaturos) e `spotted` tem poucos exemplos. Com o rig padronizado
e 10 mil+ grãos físicos disponíveis, a ordem é **calibrar o rig → capturar →
treinar a 704**. Trocar só a resolução entrega mais área com o mesmo erro de
rótulo.

---

## 6. Dataset e estratégia de dados

- **Base:** "SoyaBeans Classifications v2" (Roboflow Universe,
  hansaka-sudusinghe), MIT, 12.528 imagens 400×400, splits train/valid/test
  prontos. Usado tanto para classificação quanto (via pseudo-rótulo Otsu) como
  estágio base de detecção.
- **Fine-tuning no domínio real:** fotos/vídeos tirados pelo próprio dono,
  imitando o setup do dataset (fundo preto fosco, luz difusa de cima). Sem
  isso, a acurácia cai para ~3-8% (domain shift severo).
- **Cenas sintéticas multi-grão:** grãos recortados por segmentação de
  saturação, colados sobre fundos preto→cinza gerados, 6-25 grãos por cena,
  com sobreposição/encosto variável (10-55%) e 40% das cenas com blur
  artificial — usadas para ensinar o detector a lidar com quadros densos e
  grãos encostados, coisa que fotos de 1-grão-por-imagem não ensinam.
- **Vídeo real:** `teste_soja.mp4` é o vídeo de validação "juiz de verdade"
  usado em todos os comparativos de detecção — mesmo vídeo, mesmo pipeline,
  pra comparação justa entre modelos.
- **Protocolo de captura de vídeo** (`model/PROTOCOLO_ANOTACAO_VIDEO.md`):
  fundo preto fosco, luz difusa de cima, distância igual à de uso real,
  10-40 grãos por quadro, misturando classes e variando movimento (simula
  esteira). Meta: ~150-400 frames anotados → 2.000-5.000+ caixas. Anotação
  no Roboflow (Object Detection, não Classification), split 70/20/10 sem
  vazar frames do mesmo vídeo entre train/val.
- **Active learning** (`model/aprendizado_ativo.ipynb`): humano revisa
  `revisao.csv` gerado pelo app (Steam Deck ou navegador), corrige a classe
  onde o modelo errou, e o notebook propaga a correção + re-treina.

---

## 7. As 4 formas de rodar inferência hoje

### 7.1 Site público — modo foto (produção)
Foto → `POST /inspect` no backend FastAPI (HF Spaces) → YOLO11s-cls classifica
→ resultado salvo no Supabase → usuário pode conversar com IA (Groq/Llama
3.3 70B via `/api/explain`, rota Next.js) sobre a classe.

### 7.2 Site público — modo "ao vivo" (`frontend/src/app/ao-vivo/`)
Inferência **inteiramente no navegador**, via **ONNX Runtime Web**
(`onnxruntime-web`, `frontend/src/lib/onnx.ts`) — sem servidor. O fluxo:
segmenta o grão na webcam via Otsu (mesma lógica do treino,
`frontend/src/lib/segment.ts`), recorta o quadrado central em 224×224,
classifica localmente, aplica a regra de Premium (`intact` + confiança
≥ 0,80). Existe pra dar uma demo instantânea sem depender de nenhum backend.

### 7.3 Steam Deck — app local (`deck/vigil_deck.py`)
100% offline, sem internet nem servidor. Carrega o `.pt` direto (padrão:
`soja_yolo11n_base12k_v2.pt`), abre a câmera (USB, índice numérico, ou
DroidCam via Wi-Fi/URL), detecta+rastreia com `model.track()` (ByteTrack),
aplica voto exigente por classe e veredito travado (3-5s observando cada
grão antes de fechar a classe), desenha caixas suavizadas (EMA) e HUD com
contagem intacto/defeito. Suporta `--device intel:gpu` para OpenVINO no iGPU
AMD. Pode salvar recortes + `revisao.csv` para alimentar o active learning
(`--save-dir`).

### 7.4 Câmera do celular + GPU do Colab (demo dos modelos pesados)
Para testar YOLO11x/RT-DETR (pesados demais pra CPU) com a câmera do celular:
1. `model/demo_servidor_colab.ipynb` roda no Colab, carrega o modelo (cadeia
   de fallback: `soja_yolo11x_multi_v3.pt` → `soja_yolo11x_v3_box10.pt` →
   `soja_yolo11x_v3.pt`), sobe um FastAPI com o mesmo contrato `/inspect` do
   backend de produção, e expõe via túnel Cloudflare (link público tipo
   `https://xxxx.trycloudflare.com`).
2. `model/testar_gpu_colab.py` roda no PC local, lê a câmera do celular
   (DroidCam/Wi-Fi ou webcam), manda cada frame em base64 JPEG pro túnel, e
   desenha as caixas devolvidas. Mostra fps local e latência por chamada.
   Performance observada: ~2 fps (bottleneck é o round-trip de rede, não o
   modelo — ajustável com `--quality` menor pra reduzir o payload).
3. O mesmo túnel pode ser plugado direto no **site** via
   `?api=https://xxxx.trycloudflare.com` na URL da Vercel — troca o backend
   de visão sem redeploy (persistido em localStorage,
   `frontend/src/lib/api.ts`).

Existe ainda `model/testar_video.py`, um script standalone que desenha
**todas** as caixas cruas do modelo (sem voto/suavização) — serve
especificamente para julgar a qualidade de bounding box de um `.pt`, seja em
vídeo gravado (gera `.mp4` anotado) ou câmera ao vivo.

---

## 8. Stack tecnológico (consolidado, estado atual)

| Camada | Tecnologia | Detalhe |
|---|---|---|
| Treino/experimentos | Ultralytics/PyTorch | Google Colab (GPU T4 grátis para experimentos leves; A100/RTX PRO 6000 Blackwell para os treinos de detecção pesados) |
| Modelo — classificação | YOLO11s-cls | transfer learning, produção |
| Modelo — detecção | YOLO11n/s/m/x + RT-DETR-l | comparados; YOLO11x é o campeão de vídeo |
| Segmentação clássica | OpenCV (grayscale → Otsu → findContours → bbox) | usada no modo foto, no modo ao-vivo do browser, e como pseudo-rotulador do estágio base de detecção |
| Rastreamento (vídeo) | ByteTrack (via `model.track()` do Ultralytics) | dá o `id` persistente por grão pro voto/veredito travado |
| Backend de visão (produção) | FastAPI + Uvicorn, Docker | Hugging Face Spaces |
| Backend de visão (demo pesada) | FastAPI no Colab + túnel Cloudflare | mesmo contrato `/inspect` |
| Inferência client-side | ONNX Runtime Web (`onnxruntime-web`) | modo ao-vivo do site, sem servidor |
| LLM (análise conversacional) | Groq · Llama 3.3 70B | API compatível OpenAI, chamada só sob demanda (usuário pergunta) |
| Frontend | Next.js 14.2.3 + React 18 + TypeScript + Tailwind CSS | Vercel |
| Banco de dados | Supabase (Postgres gerenciado) | tabelas `inspecoes`, `lotes` |
| App local | OpenCV + Ultralytics, Python puro | Steam Deck (venv no home, torch CPU) |
| Dataset | Roboflow "SoyaBeans Classifications v2" | MIT, 12.528 imagens 400×400 |

Pin relevante do backend de produção: `ultralytics==8.4.80` (crítico — a
8.3.0 dava erro 500 no `/predict` por incompatibilidade de versão com o
checkpoint salvo); torch sem pin fixo (o 8.4.80 já resolve o
`weights_only=True` do torch≥2.6 automaticamente).

---

## 9. Bugs e correções (histórico consolidado)

| Sintoma | Causa raiz | Correção |
|---|---|---|
| Space não subia (`UnpicklingError`) | torch≥2.6 usa `weights_only=True`; ultralytics 8.2.0 não allowlista os globals do checkpoint | fixar `torch==2.2.2`/`torchvision==0.17.2` (versão antiga) |
| `Can't get attribute 'C3k2'` | `.pt` é YOLO11 (bloco C3k2), inexistente no ultralytics 8.2.0 | subir ultralytics para 8.3.0, depois **8.4.80** (ver abaixo) |
| `/predict` dava 500 mesmo carregando | ultralytics 8.3.0 incompatível com o formato do checkpoint salvo em 8.4.80 | fixar `ultralytics==8.4.80` — mesma versão que treinou/salvou o `.pt` |
| `ValueError: 'Broken soybeans'` | modelo usa rótulos longos ("Broken soybeans"); código comparava com os curtos ("broken") | normalizar nome de classe (`_normalize_class`) |
| Upload dava erro mesmo com HTTP 200 | CORS com wildcard `"*"` + `allow_credentials=True` (inválido pelo spec) | `allow_credentials=False` |
| "Não consegui ler o arquivo" | a mesma foto era lida duas vezes; 2ª leitura falhava no Safari/iOS | ler o arquivo uma única vez e reusar |
| Quota do `sessionStorage` estourava | data URL de foto de celular tem vários MB | reduzir a imagem antes de guardar/enviar |
| Site travava na abertura | `sessionStorage` lança exceção no Safari/modo privado | blindar com try/catch |
| Tela cheia do chat quebrada no celular | faltava `min-h-0`; barra de digitação empurrada pra fora | `min-h-0` + `100dvh` + área segura |
| Gravação no banco derrubava `/inspect` | Supabase pausado/fora quebrava a resposta inteira | gravação virou não-fatal — gera id local e segue se o banco falhar |
| Caixas frouxas / falso positivo no tapete | cenas sintéticas de treino tinham máscara com halo e fundo pouco realista | erosão de máscara (remove halo), blend de borda nítida, fundo escuro com gradiente |
| RT-DETR colapsava no fine-tune (mAP 0) | warmup padrão (bias-lr 0,1) e/ou AMP causavam NaN/colapso num modelo já convergido | receita obrigatória: `amp=False, warmup_epochs=0` |
| Pseudo-rótulo Otsu errado | letterbox aplicado **antes** de calcular a caixa | calcular a caixa antes do letterbox |
| YOLO11m com desempenho pior que 11s | treinado do zero sem ajustar receita (épocas/LR) pro tamanho maior | descartado por ora; 11s continua padrão local |
| Comparação YOLO vs RT-DETR enviesada | YOLOs não passavam pelo estágio base de 12,5k que o RT-DETR tinha | re-treinar 11l/11x pelo mesmo caminho de 2 estágios (`tira_teima_capacidade.ipynb`) |
| Colab não alcança a rede Wi-Fi local | são máquinas diferentes; Colab não tem rota pro IP do DroidCam | inverter a conexão: Colab expõe API via túnel Cloudflare, cliente local faz POST pra lá |

---

## 10. Estrutura do repositório (mapa de arquivos)

```
CLAUDE.md                          contexto oficial atual (foco: demo de classificação)
CONTEXTO_PROJETO.md                este arquivo
HANDOFF.md                         handoff de deploy (Era 2 — classificação/HF Spaces)
ROADMAP.md, docs/ESPECIFICACAO.md, docs/README.md, README.md
                                    documentos de eras anteriores (desatualizados — ver aviso no topo)
docs/deploy.md                     notas de deploy (legado, referência ao plano Render)

gerar_documentacao.py              PDF de documentação (Era 2, YOLO11s-cls, modo 1 grão)
gerar_relatorio.py                 PDF de relatório técnico (Era 1, EfficientNet)
gerar_versao.py                    PDF de registro de versão v0/v1 (Era 1, EfficientNet)

frontend/                          Next.js 14 (Vercel)
  src/app/page.tsx                 tela inicial (hero)
  src/app/resultado/[id]/page.tsx  tela de resultado (modo foto)
  src/app/ao-vivo/page.tsx         modo ao vivo — ONNX no navegador (Era 3)
  src/app/sobre/page.tsx           aba "Sobre"
  src/app/api/explain/route.ts     rota LLM (Groq/Llama 3.3, streaming)
  src/components/                  InspectionLogo, IntroSplash, InspectHero,
                                    ResultVerdict, DefectTable, BoundingBoxCanvas,
                                    ExplainPanel
  src/lib/api.ts                   cliente HTTP (/inspect, /api/explain, backend dinâmico via ?api=)
  src/lib/segment.ts               segmentação Otsu no navegador (modo ao-vivo)
  src/lib/onnx.ts                  inferência ONNX Runtime Web (modo ao-vivo)
  src/lib/supabase.ts              cliente Supabase

backend/                           FastAPI (produção, HF Spaces)
  main.py                          rotas: /health, /inspect, /inspecoes, /lotes
  inference.py                     modo 1 grão + modo multi-grão (OpenCV + YOLO11s-cls)
  database.py                      cliente Supabase
  Dockerfile                       uid 1000, porta 7860
  soja_yolo11s_finetuned.pt        modelo embutido (exceção no .gitignore)
  requirements.txt                 pins críticos: ultralytics==8.4.80

deck/                              app local Steam Deck (Era 3)
  vigil_deck.py                    inferência ao vivo 100% offline (voto + veredito travado)
  README.md                        guia completo de setup e uso
  requirements.txt

model/                             notebooks de treino/experimentos + scripts utilitários
  soja_classes.json                ordem das classes
  train_yolo.ipynb / finetune_yolo.ipynb        Era 2 — classificação (produção)
  train.ipynb / finetune.ipynb / finetune_eff_fair.ipynb   Era 1 — EfficientNet (histórico)
  treino_deteccao_video.ipynb                   Era 3 — 1º comparativo YOLO-detect vs RT-DETR
  treino_rtdetr_deteccao.ipynb                  treino RT-DETR staged (base → fine-tune → vídeo)
  melhoria_rtdetr_v3.ipynb                      RT-DETR v3 self-contained (patch NMS, veredito travado)
  finetune_lr_discriminativo.ipynb              experimento LR discriminativo (resultado negativo, arquivado)
  tira_teima_capacidade.ipynb                   comparativo decisivo de capacidade (11s/11l/11x vs RT-DETR-l)
  melhorar_boxes.ipynb                          plano de 4 estágios pra qualidade de caixa
  treino_yolo11n.ipynb / treino_yolo11s_e_11n.ipynb / teste_11n_max1280.ipynb
                                                 experimentos de tamanho/resolução (11n)
  treino_campeao_videos.ipynb / treino_campeao_videos_1280.ipynb / treino_campeao_640.ipynb
                                                 treinos do modelo campeão de vídeo em diferentes resoluções
  treino_11s_completo_640.ipynb / treino_11m_completo_640.ipynb / finetune_capturas_640.ipynb
                                                 pipelines completos por variante (11s, 11m) + fine-tune de capturas
  treino_definitivo_vigil.ipynb / .py           pipeline consolidado mais recente
  autotreino_video_v4.ipynb                     auto-treino: vídeos por pasta=classe, professor = YOLO11x
  aprendizado_ativo.ipynb                       active learning (Fase 1-3: correção humana → re-treino)
  comparativo_yolo_vs_rtdetr.ipynb / teste_justo_tamanho.ipynb
                                                 notebooks de comparação/validação
  validacao_por_classe.ipynb                    valida múltiplos detectores por classe (sem treinar)
  voto_exigente.ipynb                           desenvolvimento da regra de voto por classe
  build_holdout.py / compare_experiments.py     utilitários de avaliação
  testar_video.py                               script: caixas cruas de um .pt em vídeo/câmera (sem voto)
  testar_gpu_colab.py                           script: câmera local + inferência no Colab via túnel
  demo_servidor_colab.ipynb                     servidor FastAPI no Colab (GPU) + túnel Cloudflare
  magnus/train.py, sinnet/train.py              scripts do roadmap antigo "Sinnet/Magnus" (não usados hoje)
  COMPARATIVO_YOLO11S_VS_RTDETR.md              relatório completo do comparativo de detecção (§5 acima)
  PROTOCOLO_ANOTACAO_VIDEO.md                   protocolo de captura+anotação pra detecção em vídeo (§6 acima)

supabase/migrations/               schema inicial (inspecoes, lotes)
```

---

## 11. Decisões fechadas (o "não relitigar" atualizado)

Herdadas do `CLAUDE.md` (ainda válidas para o modo foto):
- ✅ Gradio → substituído por FastAPI (backend) + Next.js (frontend) na Era 2
- ✅ EfficientNet-B0 → substituído por YOLO11s-cls (experimento controlado, §5.1)
- ✅ OpenCV clássico pra recorte no modo foto (Otsu)
- ✅ Deploy: Vercel (frontend) + Hugging Face Spaces Docker (backend de visão) + Supabase (banco)

Novas, da Era 3 (detecção/vídeo):
- ✅ Classificação (1 grão) não escala pra cenário de vários grãos por quadro
  → trocado por detecção nesse caso de uso específico
- ✅ RT-DETR vs YOLO comparados empiricamente, sem apostar em hype — YOLO11x
  venceu depois de corrigir a assimetria de estágio de treino (§5.2)
- ✅ Cenas sintéticas multi-grão + motion blur são necessárias pra treinar
  detecção robusta a partir de fotos de 1-grão-por-imagem
- ✅ Voto ponderado por classe + veredito travado por rastreamento (ByteTrack)
  é a forma de matar alucinação de defeito em vídeo — não dá pra confiar na
  detecção de um frame isolado
- ✅ Modelos pesados (YOLO11x, RT-DETR) rodam via GPU do Colab + túnel
  Cloudflare quando precisa de demo com câmera real, não em CPU/edge
- ✅ Steam Deck usa modelo leve (11n) local, sem depender de servidor
- ✅ Destilação (professor pesado → aluno leve) é o caminho planejado pra
  levar a qualidade do YOLO11x pro edge

---

## 12. Estado atual — o que está pronto vs pendente

**Pronto e funcionando:**
- Site público com modo foto (produção) + modo ao-vivo no navegador (ONNX)
- Backend FastAPI no HF Spaces, banco Supabase, chat com LLM
- App local no Steam Deck com voto exigente, veredito travado, captura pra
  treino futuro
- Servidor de demo no Colab (GPU) pros modelos pesados, com túnel Cloudflare
- Modelo campeão de vídeo definido (YOLO11x_v3) após comparação extensa
  contra RT-DETR e outras variantes YOLO

**Pendente / próximos passos identificados no próprio histórico:**
- Auto-treino v4: vídeos organizados por pasta=classe, rotulados
  automaticamente pelo YOLO11x, pra treinar o modelo leve por destilação
- Re-testar o comparativo YOLO vs RT-DETR depois do v4 (testar se o gap era
  mesmo de dado, não só de arquitetura)
- Validação por composição: vídeos mistos com contagem conhecida de cada
  classe, pra ter uma métrica de acurácia com fonte (hoje as comparações de
  vídeo são avaliação visual, não benchmark rotulado — ver limitações)
- Reduzir a latência do fluxo Colab (hoje ~2 fps — round-trip de rede é o
  gargalo, não o modelo; `--quality` mais baixo já ajuda)
- Coleta de mais fotos reais pra validação robusta do modo foto (a validação
  atual de 91,7% é sobre só 12 fotos)
- **Rodada do rig padronizado, nesta ordem** (§5.4): montar e calibrar o rig
  (`jetson/calibrar_rig.py`, anotar px/mm na ficha do `PADRAO_CAPTURA.md` §6) →
  capturar ~120 grãos únicos de `immature` e `spotted` no rig, conferindo o
  rótulo grão a grão → treinar o RF-DETR Large a 704 (estágio 1 → FT1 → FT3 →
  FT4) → gerar engine no Jetson e medir fps → recalibrar `RATIOS` e `conf`

---

## 13. Limitações e observações honestas (importante manter na documentação final)

- **Validação do modo foto é pequena:** 91,7% equivale a 11/12 fotos — é uma
  tendência forte, não uma métrica estatística sólida.
- **Validações de vídeo são avaliação visual do dono**, não benchmark
  rotulado — os números (~65-75%, ~80-85%, ~95%) têm barra de erro
  reconhecida no próprio `COMPARATIVO_YOLO11S_VS_RTDETR.md`. A validação por
  composição (§12) é o próximo passo pra resolver isso.
- **Domain shift é o gargalo estrutural do projeto inteiro**, não só do
  modelo de classificação: qualquer modelo treinado majoritariamente em fundo
  preto de laboratório sofre queda de acurácia forte (3-8%) em fundo/luz
  reais até passar por fine-tuning no domínio de uso.
- **`skin-damaged` × `broken`** é a confusão mais comum no modo foto;
  **`spotted`** é a classe onde os detectores de vídeo mais erram.
- **RT-DETR-l e YOLO11x são modelos de GPU** — não rodam com fps útil em
  CPU/edge; o caminho pro Steam Deck/produção sem servidor é destilação pra
  um modelo leve (11n/11s/11m).
- **HF Spaces free hiberna** após inatividade — primeira chamada tem cold
  start; recomendado "aquecer" antes de demo ao vivo.
- **Segurança pendente (herdada do HANDOFF.md):** tabelas `modelos`,
  `datasets`, `melhorias` no Supabase estavam com RLS desabilitado; revisar
  antes de qualquer uso público mais amplo. Rotacionar `GROQ_API_KEY` e
  `service_role` que passaram por chat/sessões.
- **Escopo continua sendo demo acadêmica** (FATEC) — esteira, sensor NIR,
  soprador pneumático e hardware de borda (Jetson) são evolução futura, fora
  do que existe implementado hoje.
