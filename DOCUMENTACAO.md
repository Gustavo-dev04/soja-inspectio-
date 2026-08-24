# Vígil.ia — Documentação do projeto

**Inspeção automatizada de grãos de soja por visão computacional**
Projeto acadêmico — FATEC · Documento de estado e planejamento

---

## 1. Resumo executivo

O Vígil.ia classifica grãos de soja em **5 classes de qualidade** a partir de
imagem, e converte isso na decisão que o mercado realmente toma:
**premium × não-premium**.

O projeto passou por três eras técnicas (classificação de 1 grão → detecção
multi-grão em vídeo → inspeção em fluxo contínuo em hardware de borda) e hoje
tem **duas frentes que coexistem**:

| Frente | O que faz | Estado |
|---|---|---|
| **Web / modo foto** | usuário fotografa um grão, sistema classifica | **em produção**, no ar |
| **Câmara de inspeção / Jetson** | grãos passando por uma câmara com esteira, contagem e laudo | **em construção** — é o MVP |

O objetivo dos próximos 6 meses é o segundo: um **MVP com viabilidade
comercial**, comparável em valor a uma classificadora de soja pequena
(R$ 5.000–7.500).

---

## 2. Objetivo final

### 2.1 O que o MVP entrega

Uma **câmara de inspeção fechada** por onde passa um fluxo de soja. A câmera
enxerga cada grão, classifica, e ao fim do lote emite um **laudo**: quantos
grãos passaram, o percentual premium, a distribuição por tipo de defeito e a
massa estimada.

**Não separa fisicamente os grãos nesta fase.** A ejeção (solenoide / ar
comprimido) fica como fase 2 — mas o pipeline já produz o que ela precisaria:
posição e instante de cada grão.

### 2.2 Metas numéricas

| Parâmetro | Meta | Situação |
|---|---|---|
| Vazão | **0,5 t/dia em 14–16 h** = ~33 kg/h = ~58 grãos/s | dimensionado: **41 kg/h** de teto (folga 1,23×) |
| Operação | 14–16 h contínuas | ensaio térmico pendente |
| Câmara | 100 mm de faixa × 150 mm de curso | definida |
| Hardware | NVIDIA Jetson Orin Nano + câmera IMX219 | **validado no aparelho** |
| Classes | 5, com decisão binária premium × não-premium | modelo treinado, **acurácia a revalidar no rig** |

### 2.3 Critério de sucesso

O que decide se o MVP presta **não é mAP**, é:

1. **Recall de defeito** — quantos grãos ruins passam como premium (o erro
   caro).
2. **Perda de rendimento** — quantos grãos bons são descartados à toa.
3. **Vazão sustentada** em jornada real.

Os dois primeiros exigem um conjunto de validação anotado à mão no próprio
rig; hoje ele **não existe** (ver §7).

### 2.4 Posição honesta de mercado

0,5 t/dia em 15 h são **~33 kg/h**. Uma classificadora mecânica de
R$ 5.000–7.500 é especificada em **t/h** — uma a duas ordens de grandeza acima.
**O projeto não compete em vazão**, e afirmar o contrário cai no primeiro
questionamento.

O diferencial é outro: máquina mecânica separa por **tamanho, densidade e cor**
— ela não sabe *qual* é o defeito. O Vígil.ia entrega **taxonomia de defeito
por grão, com laudo auditável**, informação que aquela faixa de preço não
produz.

A escala é linear e previsível: `grãos/s = largura_mm × velocidade_mm/s ÷ 96 mm²`.
Chegar a 500 kg/h pediria ~15× mais largura imageada ou mais câmeras —
dimensionável com `jetson/calcular_vazao.py`.

---

## 3. As 5 classes

| Índice | Rótulo (dataset) | Rótulo (UI) | Observação |
|---|---|---|---|
| 0 | `broken` | Quebrado | |
| 1 | `immature` | Imaturo | classe mais fraca do modelo atual (recall 0,04) |
| 2 | `intact` | Intacto | **única classe premium** |
| 3 | `skin-damaged` | Casca danificada | fronteira ambígua com `broken` |
| 4 | `spotted` | Manchado | segunda classe mais fraca (recall 0,30) |

**Regra de veredito.** Um grão não é julgado por um frame isolado. O sistema
rastreia cada grão ao longo de várias observações, acumula votos ponderados por
confiança, e só aceita um defeito se ele reunir **75–85% dos votos** (constante
`RATIOS`). Caso contrário o grão recebe benefício da dúvida e vira `intact`.
Isso existe para matar alucinação de defeito — o falso positivo era o modo de
falha dominante em vídeo.

> ⚠️ Esses limiares são valores **escolhidos na mão**, sem curva por trás.
> Calibrá-los a partir de dados é item pendente (§7).

---

## 4. O que já existe e funciona

### 4.1 Site público (produção)

- **Modo foto:** foto → `POST /inspect` → YOLO11s-cls classifica → resultado no
  Supabase → usuário pode conversar com um LLM (Groq / Llama 3.3 70B) sobre o
  resultado.
- **Modo ao vivo no navegador:** inferência **inteiramente client-side** via
  ONNX Runtime Web — segmentação Otsu na webcam, recorte, classificação local.
  Sem servidor.
- Frontend Next.js 14 na Vercel; backend FastAPI em Docker no Hugging Face
  Spaces; banco Supabase.

### 4.2 Apps de inspeção multi-grão

| App | Onde roda | Característica |
|---|---|---|
| `deck/vigil_deck.py` | Steam Deck, 100% offline | voto exigente, veredito travado, captura para active learning |
| `jetson/vigil_jetson.py` | Jetson Orin Nano | RF-DETR via TensorRT, **modo esteira**, laudo |
| `model/demo_servidor_colab.ipynb` | Colab (GPU) + túnel Cloudflare | serve os modelos pesados para a câmera do celular |

### 4.3 Jetson Orin Nano — validado no aparelho

- Engine TensorRT **FP16 construída no próprio Jetson**:
  **53,2 qps** no modelo de 512 px, **98,2 qps** no de 384 px — ambos medidos,
  em modo 15 W.
- **Consumo medido** (`jetson/bench_energia.py`, ao ar livre, modo 15 W):

  | | 384 px | 512 px |
  |---|---|---|
  | Velocidade | 98,2 qps | 59,3 qps |
  | Ocioso | 3,87 W | 3,99 W |
  | Sob carga | 9,39 W | **10,62 W** (pico 11,6) |
  | Custo do modelo | +5,51 W | +6,63 W |
  | Junção máx | 52,0°C | 53,0°C — contra ~95°C do throttling |
  | RAM sob carga | 4,0 GB | 4,1 GB de 7,6 |
  | Jornada de 15 h | 0,14 kWh | 0,16 kWh ≈ R$ 4,30/mês |

  **Nem consumo nem velocidade separam os dois modelos na prática.** A
  diferença de potência é 1,2 W (R$ 0,50/mês), e a câmera trava em 21 fps, de
  modo que ambos sobram com folga sobre as 14 varreduras/s do dimensionamento.
  A escolha é por **qualidade**, e aí os 512 px ganham: ~33% mais pixel linear
  no grão, que é o que compra recall de defeito.

  Com o ring light (5 W) e o motor da esteira, **o rig inteiro fica em
  ~20-25 W** — menos que uma lâmpada, e roda de bateria. A fonte do dev kit é
  de 65 W, folga larga. Falta repetir dentro da câmara fechada por 14-16 h,
  que é onde a temperatura estabiliza num patamar mais alto.
- App de inferência ao vivo funcionando, com câmera CSI e travamento de
  exposição/balanço de branco.
- Comportamento observado: **forte em multi-grão, fraco em grão solto** — o que
  é consequência direta do dataset (91% das caixas vêm de cena densa) e está
  alinhado com o uso real.

### 4.4 Ferramentas de dimensionamento do rig

Três scripts que substituem chute por conta, todos em `jetson/`:

| Script | Responde |
|---|---|
| `calcular_optica.py` | com essa lente, nessa distância, o grão chega com quantos pixels? |
| `calcular_vazao.py` | essa configuração de câmara entrega a vazão alvo? qual a velocidade máxima da esteira? |
| `calibrar_rig.py` | quantos px/mm o rig entrega *de verdade* (com régua) e a exposição está boa? |
| `testar_esteira.py` | roda no PC, sem Jetson nem câmera: valida rastreamento, recortes e contabilidade |

---

## 5. Arquitetura

```
   ┌──────────────────────────┐
   │  Frontend Next.js 14      │  Vercel
   └──────┬──────────┬─────────┘
   modo foto      modo ao-vivo (ONNX no navegador,
   POST /inspect   sem servidor)
          ▼
   ┌──────────────────┐      ┌──────────────────┐
   │ FastAPI + YOLO11s │─────▶│ Supabase Postgres │
   │ HF Spaces (Docker)│      └──────────────────┘
   └──────────────────┘

  ═══════════ frente de inspeção em fluxo (o MVP) ═══════════

   câmara fechada          Jetson Orin Nano
   ┌───────────────┐       ┌────────────────────────┐
   │ ring light    │       │ RF-DETR → TensorRT FP16 │
   │ fundo preto   │──CSI─▶│ 2 recortes 1:1          │
   │ esteira       │       │ rastreamento + voto     │
   │ 100 × 150 mm  │       │ laudo (JSON periódico)  │
   └───────────────┘       └────────────────────────┘
```

---

## 6. O rig de inspeção — os números que decidem tudo

Documento normativo completo: **`jetson/PADRAO_CAPTURA.md`**.

### 6.1 Geometria escolhida

| Parâmetro | Valor | Por quê |
|---|---|---|
| Câmera | IMX219, 3280×2464, lente M12 de 120° | já comprada |
| Distância de trabalho | **9,3 cm** | é a distância em que 2 recortes de 704 px cobrem exatamente os 100 mm da faixa |
| Escala | **12,7 px/mm** | consequência da distância |
| Grão no modelo | **89 px** | dentro da faixa de treino (60–150 px) |
| Recortes | **2 × 704 px**, sobreposição de 140 px | cobre a faixa em escala 1:1, sem reescalar |
| Iluminação | ring light 6500–7000 K, difusa | fidelidade de cor (cor é sinal de defeito) |
| Fundo | cartolina preta fosca | torna a segmentação Otsu trivial |
| Exposição / WB | **travados** | em automático, a câmera recria domain shift entre sessões |

**Por que 2 recortes e não 1:** com um recorte só, a faixa de 100 mm só cabe a
15 cm de distância — e aí o grão cai para 55 px. Com dois recortes dá para
aproximar até 9,3 cm, cobrir a mesma faixa e **dobrar o detalhe linear**. Como
o critério de sucesso é recall de defeito, e defeito é textura (mancha, casca),
pixel em cima do defeito é exatamente o que compra recall.

### 6.2 O limite real do sistema: rastreamento, não fps

Existem dois tetos de velocidade, e o menor manda:

| Limite | Do que depende | Valor |
|---|---|---|
| Varreduras | o veredito só trava após 8 observações do mesmo grão | 97 mm/s |
| **Rastreamento** | o grão precisa ser reconhecido de uma varredura para a outra | **69 mm/s** |

Essa conta revelou um problema que **não daria erro nenhum na tela**: o tracker
original casava detecções por IoU puro, com o pressuposto (escrito no próprio
código) de que "o grão quase não se move entre quadros". Duas caixas de 7 mm
deslocadas de 4 mm têm IoU 0,27, abaixo do mínimo de 0,30 — **limite de
53 mm/s, contra os 56 mm/s que a meta exige**. O grão trocaria de ID no meio da
travessia, zerando os votos e inflando a contagem.

Corrigido com **compensação de movimento** (a caixa é prevista pela velocidade
antes de casar). O novo limite de 69 mm/s é **físico**: acima de meio
espaçamento entre grãos, o vizinho da frente fica mais perto do que o grão
andou, e nenhum algoritmo desempata isso sem outra fonte de informação.

**Resultado: 69 mm/s → 41 kg/h, contra a meta de 33 kg/h.**

### 6.3 Se o movimento não for esteira

Todo o dimensionamento assume **esteira** (velocidade constante). Calha
vibratória exigiria revisar o rastreamento (o grão gira entre varreduras);
**queda livre invalidaria o modelo de votação inteiro** — o grão apareceria em
1–2 quadros, e o sistema viraria classificação por quadro único com strobe.
Detalhamento em `PADRAO_CAPTURA.md` §7b.

---

## 7. O que ainda falta fazer

A ordem importa: cada item depende do anterior.

### Fase 1 — Montagem e calibração *(bloqueia todo o resto)*

- [ ] Montar a câmara fechada com fundo preto fosco e ring light difuso
- [ ] Fixar a câmera a ~9,3 cm e **travar o foco fisicamente**
- [ ] **Calibrar com régua** (`calibrar_rig.py`) — confirmar os 12,7 px/mm e
      ajustar a exposição (sem pixel estourado, fundo escuro mas não cravado)
- [ ] Definir o acionamento da esteira e **medir a velocidade real** com
      cronômetro; precisa ficar abaixo de 69 mm/s
- [ ] Preencher a ficha do rig (`PADRAO_CAPTURA.md` §6)

### Fase 2 — Dataset no domínio padronizado *(o item de maior impacto)*

O gargalo de acurácia nunca foi arquitetura — é **domain shift e rótulo ruim**.
O `immature` do dataset atual está sistematicamente errado (as fotos não são de
grãos imaturos) e `spotted` tem poucos exemplos. Com o rig padronizado e
10 mil+ grãos físicos disponíveis, isso deixa de ser limitação.

- [ ] **Bandeja de classe única → rótulo de graça.** Passar um lote de UMA
      classe por vez: a caixa sai do Otsu (fundo controlado, >98% de acerto
      histórico) e a classe vem da pasta. Dezenas de milhares de caixas
      corretas, **zero anotação manual**.
- [ ] **Bandeja mista → validação anotada à mão.** Algumas centenas de grãos,
      5 classes balanceadas. É o **único** conjunto que mede recall de verdade.
- [ ] Gravar um **lote propositalmente ruim**, com defeito conhecido — sem ele
      não dá para distinguir modelo bom de modelo viciado em dizer "intacto".

### Fase 3 — Treino no novo domínio

A estratégia se inverteu: em vez de adaptar o modelo ao domínio, **a câmara
reproduz o domínio do dataset**. Isso permite uma linha de base limpa antes de
qualquer captura própria.

- [x] **Linha de base — só o 12,5 k** (`model/treino_base12k_jetson.ipynb`).
      Um treino só, COCO → 12,5 k, com cenas multi-grão renderizadas a partir
      dos recortes do próprio dataset, **na escala do rig** (grão a ~89 px,
      fundo da câmara, densidade da esteira). Notebook pronto e testado; falta
      rodar.
- [ ] **Rodar o modelo base na câmara com soja real.** É o teste da aposta: se
      inspecionar bem, o domínio foi mesmo reproduzido. Se não, a diferença que
      sobrar diz exatamente o que o dataset próprio precisa cobrir
- [ ] Fine-tune a partir do `.pth` base (não do COCO), com o dataset da Fase 2
- [ ] Exportar ONNX e gerar a engine TensorRT **no próprio Jetson**

### Fase 4 — Medição que vale

- [ ] Matriz de confusão 5×5 **e** a binária premium × não-premium, no conjunto
      anotado
- [ ] **Curva de operação**: para cada limiar, recall de defeito × perda de
      rendimento. O ponto na curva é decisão de negócio, não de engenharia
- [ ] **Calibrar os `RATIOS`** a partir dessa curva, substituindo os valores na
      mão de hoje
- [ ] Medir fps real a 704 px no Jetson (**os ~28 fps são estimativa**, não
      medição — o valor medido de 53,2 qps é a 512 px)

### Fase 5 — Operação contínua

- [ ] **Ensaio térmico de jornada longa** — os 53,2 qps foram medidos em
      rajada curta, no modo 15 W. Quinze horas dentro de uma câmara fechada é
      outro regime; conferir throttling
- [ ] Watchdog com reinício automático (o laudo parcial periódico já existe)
- [ ] Documentar o modo `nvpmodel` usado

### Fase 6 — Otimização *(só se pagar)*

- [ ] Avaliar **INT8**. ⚠️ **Não assumir que é mais rápido:** há relato de
      regressão de 2,7× com INT8 num ViT-S no Orin Nano — e o backbone do
      RF-DETR Large é exatamente um ViT-S. A/B obrigatório contra FP16, medindo
      também acurácia no conjunto anotado. FP16 já entrega a meta com folga:
      **INT8 é margem, não requisito.**

### Fora do MVP (registrado, não priorizado)

- Ejeção física (fase 2 do produto)
- Sensor espectral (AS7265x) com fusão por *cross-attention* — só faz sentido
  se o RGB puro empacar depois da Fase 4; hoje seria complexidade prematura
- Destilação para modelo leve rodar em CPU/edge sem Jetson
- Revisar RLS das tabelas Supabase e rotacionar chaves antes de uso público
  mais amplo

---

## 8. Histórico técnico — como chegamos aqui

### 8.1 Classificação: EfficientNet → YOLO11s-cls

Experimento controlado, mesmas 57 fotos, mesmo recorte, mesmo split:

| Modelo | Acurácia (domínio real) | Gap treino-val |
|---|---|---|
| EfficientNet-B0 (receita antiga) | 64,0% | 25,0% (overfitting) |
| EfficientNet-B0 (receita justa) | 75,0% | 7,2% |
| **YOLO11s-cls** | **91,7%** | **3,9%** |

A receita explica ~11 pp; a **arquitetura** explica outros ~17 pp. Em produção
desde então.

### 8.2 Detecção: a comparação mais extensa do projeto

A hipótese inicial era que RT-DETR (transformer) venceria a família YOLO. Em
vez de apostar no hype, os dois foram treinados nas mesmas condições:

| Modelo | Params | Veredito no vídeo *(avaliação visual)* |
|---|---|---|
| YOLO11s (sem estágio base) | 9,4 M | ~65–75% |
| YOLO11l | 25,3 M | ≈ RT-DETR |
| RT-DETR-l | 32,0 M | ~80–85% |
| **YOLO11x** | **56,9 M** | **~95% — campeão** |

**Conclusão:** o gap não era arquitetura, era **capacidade + caminho de dados**.
A primeira comparação estava enviesada — os YOLOs não tinham passado pelo
estágio base de 12,5 k imagens que o RT-DETR teve. Corrigida a assimetria, a CNN
com atenção venceu.

### 8.3 Edge: RF-DETR para o Jetson

YOLO11x e RT-DETR-l são modelos de GPU de servidor, inviáveis num Orin Nano.
O RF-DETR (backbone DINOv2, Apache 2.0) foi testado pelo mesmo caminho.

Sequência de fine-tunings, todos julgados no mesmo vídeo:

| Estágio | Resultado | Lição |
|---|---|---|
| Base 12,5 k | mAP 0,985 | tarefa fácil, satura rápido |
| FT1 | mAP 0,826 | perfil saudável, fraco em achar defeito |
| FT2 (só capturas) | mAP 0,375 | **catastrophic forgetting** — perdeu até para o FT1 sozinho |
| FT3 (30% replay) | recuperou | experience replay resolve o esquecimento |
| **FT4 (fonte por classe)** | **campeão** | funcionou como **filtro de qualidade de rótulo** |

**Descoberta importante do FT4:** o ganho não veio de imagem melhor, veio de
**rótulo melhor** — as fotos de `immature` do FT1 estavam sistematicamente
erradas. Foi o que motivou a decisão de recapturar em vez de remendar.

**Escolha da variante.** Para subir a resolução de 512 para 704 px não se passa
`resolution=` custom: troca-se de **variante**. As quatro (Nano 384 / Small 512
/ Medium 576 / Large 704) têm praticamente o mesmo tamanho (~30–34 M params) —
o que muda é a resolução nativa. A `RFDETRLarge` é literalmente *a Small a
704 px* (mesmo encoder `dinov2_windowed_small`, mesmo patch 16). Assim o
positional embedding pré-treinado casa exato, sem interpolação.

### 8.4 Resultados negativos registrados

Valem tanto quanto os positivos, porque impedem repetir o erro:

- **Balancear o *train* por classe** (e não só o *val*) viciou o modelo
  pró-defeito num vídeo majoritariamente intacto. O val deve ser balanceado; o
  train deve espelhar a distribuição de uso.
- **Fine-tune final só com dado novo, sem replay**, apaga o que o estágio
  anterior aprendeu.
- **LR discriminativo** não ajudou — o gargalo era dado, não otimização.
- **YOLO11m** ficou pior que o 11s por ter sido treinado sem ajustar a receita
  para o tamanho maior.

---

## 9. Dados

| Fonte | Volume | Uso |
|---|---|---|
| Roboflow "SoyaBeans Classifications v2" (MIT) | 12.528 imagens 400×400 | estágio base, via pseudo-rótulo Otsu |
| Fotos reais do domínio | ~600 (com oversampling) | fine-tuning |
| Cenas sintéticas multi-grão | ~1.200 | ensina quadros densos e grãos encostados |
| `teste_soja.mp4` | ~809 frames | "juiz de verdade" de todos os comparativos |
| **Grãos físicos disponíveis** | **10 mil+** | **ainda não capturados — Fase 2** |

**Pseudo-rótulo Otsu** é o que torna o dataset barato: com fundo preto
controlado, a segmentação clássica acerta a caixa em >98% dos casos, e a classe
vem da organização em pastas. Nenhuma caixa foi desenhada à mão até aqui — e
justamente por isso a validação anotada da Fase 2 é indispensável.

---

## 10. Limitações honestas

Manter esta seção na documentação final. Ela é o que separa um relatório
defensável de um que cai no primeiro questionamento.

- **A acurácia atual não vale para o rig.** Todos os números medidos vêm de
  outro domínio (fotos de celular, fundo diferente). Medir "ele inspeciona
  bem?" hoje produz número que não se transfere. Até a Fase 4, só o que
  independe de domínio é mensurável: fps, latência, estabilidade de caixa e de
  rastreamento.
- **Os números de vídeo (~65–75%, ~80–85%, ~95%) são avaliação visual**, não
  benchmark rotulado. Têm barra de erro reconhecida.
- **A validação do modo foto é pequena:** 91,7% equivale a 11 de 12 fotos — é
  tendência forte, não métrica estatisticamente sólida.
- **Domain shift é o gargalo estrutural do projeto inteiro.** Ele reapareceu em
  todas as três eras (29% → 64% → 91,7%). Todo modelo treinado em fundo
  controlado cai para 3–8% em fundo/luz reais até passar por fine-tuning no
  domínio de uso. É por isso que o padrão de captura é documento normativo — e
  por que a estratégia atual é **inverter o problema**: em vez de adaptar o
  modelo ao domínio novo, construir a câmara para reproduzir o domínio do
  dataset. Se der certo, o modelo base transfere sem fine-tune; se não der, a
  diferença que sobrar é medível e diz o que capturar.
- **`immature` (recall 0,04) e `spotted` (recall 0,30)** são as classes fracas
  do modelo atual. A causa identificada é rótulo errado + poucos exemplos, não
  arquitetura.
- **O vídeo de teste é quase todo `intact`** — ele não distingue "modelo bom" de
  "modelo viciado em dizer intacto". Daí a necessidade do lote propositalmente
  ruim.
- **Os ~28 fps a 704 px são estimativa**, escalada do valor medido a 512 px.
- **HF Spaces free hiberna** após inatividade — aquecer antes de demo ao vivo.
- **Pendência de segurança:** revisar RLS no Supabase e rotacionar chaves antes
  de uso público mais amplo.

---

## 11. Mapa do repositório

```
DOCUMENTACAO.md                este documento
CONTEXTO_PROJETO.md            contexto bruto e completo (fonte deste documento)
CLAUDE.md                      instruções de projeto para assistente de código

jetson/                        ── a frente do MVP ──
  PADRAO_CAPTURA.md            NORMATIVO: define o rig; dado fora dele não entra no dataset
  vigil_jetson.py              app de inferência ao vivo (TensorRT, modo esteira, laudo)
  calcular_optica.py           lente + distância → pixels por grão
  calcular_vazao.py            câmara + recortes + fps → velocidade máx e kg/h
  calibrar_rig.py              medição com régua: px/mm e exposição
  testar_esteira.py            testes que rodam no PC, sem Jetson nem câmera
  bench_trt.sh                 benchmark da engine no aparelho
  README.md                    guia de ONNX → engine → app

model/                         notebooks de treino (30+, uma por experimento)
  treino_base12k_jetson.ipynb           LINHA DE BASE: só o dataset 12,5k, pro Jetson
  testar_base12k*.py                    testes do notebook acima, rodam sem GPU
  treino_rfdetr_small_completo.ipynb    pipeline multi-estágio (histórico do FT1→FT4)
  tira_teima_capacidade.ipynb           comparativo decisivo YOLO vs RT-DETR
  COMPARATIVO_YOLO11S_VS_RTDETR.md      relatório completo do comparativo
  PROTOCOLO_ANOTACAO_VIDEO.md           protocolo de captura e anotação

backend/                       FastAPI + YOLO11s-cls (produção, HF Spaces)
frontend/                      Next.js 14 (produção, Vercel)
deck/                          app local Steam Deck, 100% offline
```

> ⚠️ `ROADMAP.md`, `docs/ESPECIFICACAO.md`, `docs/README.md` e `README.md`
> descrevem versões **anteriores e diferentes** do projeto (outros rótulos,
> outros modelos, Gradio). Tratar como histórico, não como verdade atual.

---

## 12. Glossário

| Termo | Significado |
|---|---|
| **Domain shift** | o modelo aprende num domínio (câmera/luz/fundo) e é usado em outro; acurácia despenca |
| **Pseudo-rótulo Otsu** | gerar a caixa automaticamente por limiarização, aproveitando o fundo controlado |
| **Catastrophic forgetting** | ao treinar só com dado novo, o modelo apaga o que sabia |
| **Experience replay** | misturar uma fração do dado antigo no treino novo para evitar o esquecimento |
| **Recorte 1:1 (ROI)** | recortar do sensor no tamanho da entrada do modelo, sem reescalar |
| **Varredura** | uma passada completa do modelo por todos os recortes de um quadro |
| **Veredito travado** | a classe final de um grão, fechada após votos suficientes e não mais alterada |
| **Premium** | grão `intact`; qualquer defeito é não-premium |
