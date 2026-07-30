# YOLO11s vs RT-DETR — Comparativo documentado (Vígil.ia)

> Registro consolidado dos treinos, bugs, correções e validações que levaram à
> escolha do RT-DETR-l como modelo de **detecção em vídeo** do projeto, com o
> YOLO11s-det mantido como candidato de **execução local** (a fortalecer via
> destilação). Todos os números vêm dos logs reais das rodadas; onde a fonte é
> avaliação visual ou estimativa, isso está marcado.

---

## 1. Contexto — por que saímos da classificação

O Vígil.ia começou com **classificação** (1 grão por imagem): YOLO11s-cls venceu
o EfficientNet-B0 em experimento controlado (91,7% vs 75% no domínio real) e é o
modelo do modo foto até hoje. O cenário **industrial**, porém, é vídeo com
**vários grãos por quadro** — classificação + recorte Otsu não escala pra isso
(o Otsu exige fundo controlado e 1 grão). A solução é **detecção**: o modelo
acha *e* classifica cada grão num passo só.

A hipótese do projeto (do dono) era que o **RT-DETR** (detector transformer da
Baidu, família DETR) se sairia melhor em vídeo que a família YOLO. Seguindo o
método do projeto — *o experimento decide, não o hype* — os dois foram treinados
e comparados nas mesmas condições.

## 2. Os dois modelos

| | **RT-DETR-l** | **YOLO11s-det** |
|---|---|---|
| Família | DETR (transformer end-to-end) | CNN one-stage |
| Parâmetros | 32,0 M | ~9,4 M |
| Custo | 103,5 GFLOPs @640 | ~21,5 GFLOPs @640 |
| Peso em disco | ~66 MB (.pt) | ~19 MB (.pt) |
| NMS | **Não roda NMS** no ultralytics (só filtro de confiança — verificado no fonte 8.4.80); precisou de patch de NMS agnóstico de classe | NMS nativo (`agnostic_nms=True` disponível) |
| Particularidades de treino | Mosaic treina em tela 2×imgsz (1280) por design; **AMP pode dar NaN** (aviso do próprio ultralytics); **warmup padrão (bias-lr 0,1) colapsa** um modelo já convergido em dataset pequeno → receita obrigatória no fine-tune: `amp=False, warmup_epochs=0` | Treino padrão estável (warmup e AMP default funcionam) |

Ambos treinados com **ultralytics 8.4.80**, imgsz 640, AdamW, mesmas
augmentations, mesma GPU (RTX PRO 6000 Blackwell, 97 GB).

## 3. Dados usados (idênticos para os dois no comparativo)

- **Base (só RT-DETR):** 12.528 imagens do dataset Roboflow (classificação),
  convertidas para detecção via **pseudo-rótulo Otsu** (1 grão/imagem, fundo
  preto → caixa automática). Zero anotação manual.
- **Dataset v3 (fine-tune, usado nos DOIS modelos):** ~600 fotos reais
  balanceadas por oversampling (a coleta tinha 122 intact vs ~21–32 por defeito)
  ×2 com **motion blur sintético**, + ~600 **cenas multi-grão sintéticas**
  (compositor: grãos recortados por segmentação de saturação colados em fundos
  preto→cinza, 6–25 por cena, com sobreposição/encosto), 40% das cenas borradas.
  Validação: 40 fotos reais.

## 4. Histórico de treinos (linha do tempo)

| Rodada | Modelo | Resultado | Observações |
|---|---|---|---|
| base_12k | RT-DETR-l ← COCO | **mAP50 0,982 · mAP50-95 0,980** (val 12,5k) | 50 épocas, ~4 h; salto no close_mosaic (ép. 41: 0,955→0,978); revelou vício de "caixa = quadro inteiro" fora do dataset |
| ft_real (1ª) | RT-DETR ← base | **mAP 0 (falha)** | Bug do pseudo-rótulo: letterbox antes do Otsu → caixa = foto inteira em fundo cinza |
| ft_real2 (2ª) | RT-DETR ← base | mAP 0 por colapso → corrigido → **mAP50 ~0,79** | Colapso por warmup (cls_loss→3e-5); receita `amp=False, warmup_epochs=0` resolveu |
| ft_v3 | RT-DETR ← base, dataset v3 | **Campeão atual** | Curou o "tudo vira broken" no borrão; com NMS + veredito travado = vídeo estável |
| yolo11s_v3 | YOLO11s ← COCO, dataset v3 | Treinou normal | Comparativo justo: mesmos dados/época/imgsz |
| ft_v3_disc | RT-DETR ← ft_v3, LR discriminativo | **Pior que ft_v3** (resultado negativo) | Dataset já saturado — gargalo era dado, não otimização. Registrado e descartado |

### mAP por classe — RT-DETR base_12k (val 12,5k, pseudo-rótulo)

| classe | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| broken | 0,923 | 0,923 | 0,961 | 0,956 |
| immature | 0,973 | 0,978 | 0,992 | 0,992 |
| **intact** | 0,966 | 0,983 | **0,993** | 0,993 |
| skin-damaged | 0,965 | 0,948 | 0,981 | 0,981 |
| spotted | 0,924 | 0,955 | 0,982 | 0,980 |

`intact` (a classe do veredito Premium) é a mais forte; `broken` e `spotted` as
mais fracas — padrão que se repetiu no vídeo.

## 5. Validação em vídeo (o juiz de verdade)

Protocolo: mesmo vídeo real (`teste_soja.mp4`, ~809 frames, vários grãos, fundo
escuro), mesmo pipeline de inferência: **NMS agnóstico de classe** (iou 0,6,
conf 0,35) + **rastreamento ByteTrack** + **veredito travado por grão** (voto
majoritário ponderado por confiança; classe trava com ≥8 frames e ≥60% de
consenso).

### Achados por rodada (RT-DETR)

1. **ft_real2:** detecção multi-grão funcionou (até ~20 grãos/quadro, contagens
   estáveis por dezenas de frames, 14,4 ms/frame ≈ 70 fps na GPU). Fraqueza:
   em **movimento médio**, o borrão apagava a textura e todo defeito virava
   "broken" — mas o `intact` continuava sendo reconhecido. Em movimento lento,
   tudo correto. *(avaliação visual do dono)*
2. **ft_v3 (blur+balanceamento+multi-grão):** o erro do "tudo broken" sumiu.
   Restou troca de classe entre frames (flicker) → resolvido pelo veredito
   travado, e caixa dupla com 2 classes no mesmo grão → resolvido pelo patch
   NMS. Resultado descrito pelo dono: **"infinitamente melhor — sweet spot"**.

### Cara a cara — ft_v3 vs yolo11s_v3 no mesmo vídeo *(avaliação visual do dono)*

| Critério | RT-DETR ft_v3 | YOLO11s v3 |
|---|---|---|
| `spotted` | sentiu dificuldade | — |
| Alucinação (caixas/classes falsas) | menos | **mais** |
| **Acurácia premium vs não-premium (estimada)** | **~80–85%** | ~65–75% |

Diferença de ~10–15 pontos percentuais na métrica que importa pro negócio
(premium/não-premium), a favor do RT-DETR, **com os mesmos dados de treino**.

## 6. Velocidade e custo de execução

| Cenário | RT-DETR-l | YOLO11s | Fonte |
|---|---|---|---|
| GPU (RTX PRO 6000), vídeo | 14,4 ms ≈ **70 fps** | mais rápido ainda | medido (log) |
| GPU, val | 1,6 ms/img | — | medido (log) |
| CPU i3-1315U, PyTorch | ~0,5–1 fps | — | estimativa |
| CPU i3-1315U, ONNX | ~1–2 fps | ~5–8 fps | estimativa |
| CPU i3-1315U, OpenVINO INT8 / iGPU | ~3–5 fps | **~10–15 fps** | estimativa |

O RT-DETR-l é um modelo de GPU; o YOLO11s é viável em CPU/edge. Nenhuma
otimização de runtime muda a ordem de grandeza dessa diferença (~5× em FLOPs).

## 7. Leitura técnica — por que o transformer venceu em qualidade

O padrão **se repete** no histórico do projeto:

- Classificação: EfficientNet-B0 (CNN pura) 75% vs YOLO11s-cls (CNN + bloco de
  atenção C2PSA) 91,7% — atribuído em parte à atenção espacial.
- Detecção: YOLO11s-det (CNN) ~65–75% vs RT-DETR-l (transformer end-to-end,
  encoder AIFI + decoder com queries) ~80–85% no premium.

Com **dado escasso** (~600 fotos reais + sintético), mecanismos de atenção
generalizam melhor e alucinam menos: o transformer compara o grão com o contexto
global do quadro, enquanto a CNN pequena depende mais de padrões locais — que o
borrão e a variação de luz corrompem primeiro. O custo dessa robustez é
computação (5× mais FLOPs) e treino mais delicado (colapso de warmup, NaN com
AMP, convergência lenta nas primeiras épocas — tudo documentado na seção 2).

Nota honesta: 65–75% vs 80–85% são estimativas visuais num vídeo, não um
benchmark rotulado. A conclusão direcional é firme (o dono viu os dois vídeos
lado a lado no mesmo material); os números exatos têm barra de erro.

## 8. Decisão e papéis atuais

| Papel | Modelo | Status |
|---|---|---|
| **Modo foto (produção)** | YOLO11s-cls (`soja_yolo11s_finetuned.pt`) | no ar (HF Space) — intocado |
| **Vídeo / industrial (demo)** | **RT-DETR-l ft_v3** + NMS + veredito travado | pronto; demo via GPU do Colab (`demo_servidor_colab.ipynb` + `?api=` no site) |
| **Local / edge (futuro)** | YOLO11s-det | aguardando **destilação**: o RT-DETR anota os vídeos brutos (classe = pasta, caixa = professor) e o YOLO reaprende com dado do domínio real de vídeo — objetivo: fechar o gap de qualidade mantendo os ~10–15 fps no i3 |

## 9. Próximos passos registrados

1. Pastas de vídeo: `treino/<classe>/` (vídeos de classe única, pares luz
   normal + flash em TODAS as classes) e `validacao/` (mistos, composição real
   anotada) → auto-treino v4 com classe-da-pasta (elimina o eco do
   self-training) e **validação por composição** (contagem conhecida vs veredito
   travado = acurácia em número, sem anotar caixa).
2. Re-testar o comparativo YOLO vs RT-DETR **depois** do v4 — a tese da
   destilação é que o gap é de dado, não só de arquitetura; o v4 é o teste dela.
3. Resultado negativo arquivado: LR discriminativo em dataset saturado não
   melhora (ft_v3_disc < ft_v3). Re-testável junto com dado novo, não sozinho.


---

## 10. ADENDO — Tira-teima da curva de capacidade (resultado que muda a decisão)

Após a seção 8, rodamos o experimento decisivo (`tira_teima_capacidade.ipynb`):
YOLO11l e YOLO11x treinados pelo **mesmo caminho de 2 estágios do RT-DETR**
(COCO → base 12,5k com pseudo-rótulo Otsu → fine-tune v3) — corrigindo a
assimetria do comparativo original, em que os YOLOs não viam as 12,5k imagens.
Avaliação: mesmo vídeo, mesmo pipeline (NMS + veredito travado), estimativa
visual de acurácia premium pelo dono.

| Modelo | Params | GFLOPs | Veredito no vídeo *(visual)* |
|---|---|---|---|
| YOLO11s (sem estágio base) | 9,4 M | 21,5 | ainda atrás (~65–75%) |
| YOLO11l | 25,3 M | 86,9 | ≈ RT-DETR; alucina mais, porém **melhor no spotted** |
| RT-DETR-l | 32,0 M | 103,5 | ~80–85% (campeão anterior) |
| **YOLO11x** | **56,9 M** | **194,9** | **~95% — novo campeão** |

### Conclusões

1. **O gap era capacidade + caminho de dados, não arquitetura.** Em capacidade
   pareada (11l vs RT-DETR-l), empate com perfis de erro distintos; com ~2× a
   capacidade, a CNN (com atenção C2PSA) supera o transformer.
2. A tese da seção 7 ("transformer generaliza melhor com dado escasso") fica
   **restrita ao regime de capacidade igual/menor** — e parte da vantagem
   original do RT-DETR sobre o 11s era o estágio base que o 11s não teve.
3. Bônus de engenharia do 11x: NMS nativo (`agnostic_nms=True`, sem patch),
   treino estável (sem o colapso de warmup nem o risco de NaN com AMP do
   RT-DETR), pipeline mais simples.
4. Ressalva honesta de sempre: ~95% é estimativa visual no vídeo de teste, não
   benchmark rotulado; a validação por composição (vídeos mistos com contagem
   conhecida) vai dar o número com fonte.

### Papéis atualizados

| Papel | Modelo | Status |
|---|---|---|
| Modo foto (produção) | YOLO11s-cls | no ar — intocado |
| **Vídeo / demo (GPU)** | **YOLO11x_v3** (`soja_yolo11x_v3.pt`) | novo campeão; demo via Colab atualizada |
| Reserva / segunda opinião | RT-DETR-l ft_v3 | arquivado no Drive; útil como comparador |
| Local / edge (futuro) | YOLO11s ou 11m via destilação | professor da destilação agora é o **11x** |

O auto-treino v4 (vídeos por classe) passa a usar o **YOLO11x como professor**
das caixas (classe continua vindo da pasta) — adaptação junto com a leitura da
pasta de vídeos.

---

## 11. ADENDO — RF-DETR Small, mira no Jetson Orin Nano

Motivação: os campeões de vídeo (RT-DETR-l, YOLO11x) são modelos de servidor —
inviáveis em tempo real num Jetson Orin Nano (8GB, ~67 TOPS INT8). O RF-DETR
(família da Roboflow, mesmo espírito DETR do RT-DETR mas com backbone DINOv2 e
NAS de arquitetura) publica variantes Nano/Small/Medium com latência de poucos
ms na T4 via TensorRT — candidato natural pra edge. Testado o **Small** (32,1M
params, patch 16, resolução 512/672 multi-scale) pelo mesmo caminho de estágios:
COCO → base 12,5k (pseudo-rótulo Otsu) → fine-tune no domínio real.

### Histórico de treinos (`model/treino_rfdetr_small_completo.ipynb`)

| Rodada | Dado de treino | Resultado no `teste_soja.mp4` |
|---|---|---|
| estágio 1 (base 12,5k) | 1 grão/imagem, pseudo-rótulo Otsu | mAP50-95 0,985 no val — saturou rápido (tarefa fácil, mesmo padrão do RT-DETR base_12k) |
| **FT1** (fotos reais + vídeo de defeito + cenas sintéticas) | val misto (single+cena), balanceado | mAP50-95 0,826 (v2); no vídeo, forte em intacto (~84%), fraco em achar defeito |
| FT2 (capturas do `vigil_deck`, 100% cena sintética) | val misto, balanceado | mAP50-95 0,375 (v2), classes colapsadas (skin-damaged 0,13, immature 0,20) — **catastrophic forgetting**: perdeu no vídeo até pro FT1 sozinho |
| **FT3** (capturas + 30% replay do FT1, val misto das 2 fontes) | experience replay, técnica já usada na era EfficientNet (`gerar_relatorio.py`, §7-8) | **melhor dos dois mundos no vídeo** — resolveu o esquecimento do FT2 sem perder o que o FT1 tinha, caixas boas *(avaliação visual do dono)* |
| **FT4** (fonte de recorte escolhida **por classe** nas cenas) | `broken`/`skin-damaged` da foto real, `spotted`/`immature` da captura, `intact` meio a meio | **campeão — o "sweet spot"**: parou de confundir `immature` com `intact` e manteve o resto das classes bem *(avaliação visual do dono)* |

### O que o FT4 realmente consertou (importante, e não é o que parecia)

A hipótese ao montar o FT4 era de **qualidade de imagem**: cada fonte teria fotos
melhores de classes diferentes. O que aconteceu na prática foi outra coisa — as
fotos de `immature` do FT1 estavam **mal rotuladas** (erro de anotação do dono),
e ensinavam o modelo que "imaturo se parece com intacto". Trocar a fonte dessa
classe não deu imagens melhores ao modelo: **removeu o contra-exemplo errado do
treino**.

Ou seja, o `FONTE_POR_CLASSE` funcionou como **filtro de qualidade de rótulo**,
não de imagem. Consequências práticas:

- É um **contorno**, não uma correção. O dado ruim continua lá em
  `Soja total/.../Immature soybeans` e volta a atrapalhar em qualquer receita
  que use aquela pasta (inclusive o FT1 e o FT3).
- Pode haver contaminação parecida **em outras classes** que ainda não apareceu
  porque não foi isolada por nenhum experimento. Vale rodar o campeão sobre as
  fotos do FT1 e revisar aquelas em que o modelo discorda com confiança alta da
  pasta — é exatamente o fluxo do `model/aprendizado_ativo.ipynb` (correção
  humana → re-treino).
- Corrigir o rótulo na origem provavelmente **melhora ainda mais** que contornar,
  porque devolve ao treino as fotos de `immature` do FT1 (mais dado), em vez de
  descartá-las.

### Ressalva que só o val mostra: "parou de confundir" ≠ "aprendeu a detectar"

O val do FT4 (mAP 0,631 no melhor checkpoint) revela um detalhe invisível no vídeo:

| classe | AP 50-95 | precisão | **recall** | fonte do recorte |
|---|---|---|---|---|
| broken | 0,859 | 0,865 | 0,768 | foto real |
| intact | 0,810 | 0,660 | 0,912 | 50/50 |
| skin-damaged | 0,674 | 0,669 | 0,703 | foto real |
| spotted | 0,299 | 0,360 | 0,596 | captura |
| **immature** | **0,147** | 0,246 | **0,041** | captura |

**Recall 0,041 em `immature`**: o modelo praticamente parou de prever essa classe.
A confusão imaturo↔intacto sumiu do vídeo porque ele quase não chuta mais
"imaturo" — não porque passou a distinguir os dois.

⚠️ **Confirmado com o dono: o `teste_soja.mp4` não tem nenhum grão imaturo.**
Logo, "não confundir" e "não detectar" produzem exatamente a mesma imagem nesse
vídeo, e **o vídeo não valida a classe `immature` de forma alguma** — o val é o
único sinal que existe pra ela, e ele diz que a classe está fraca. Isso não
invalida o FT4 como melhor escolha *para o dado de hoje* (decisão consciente do
dono), mas fica registrado: **se um lote com grão imaturo entrar na linha, este
modelo vai deixar passar**. Reavaliar antes de qualquer uso com lote diferente
do que foi filmado.

Causa provável: **falta de variedade**, não de receita. Contagem de recortes
únicos disponíveis por fonte:

```
foto real: broken 122, immature 122, intact 122, skin-damaged 122, spotted 122
captura:   broken  99, immature  56, intact  22, skin-damaged  42, spotted  35
```

O pool pede 400/classe: `immature` sai de 56 únicos (7× duplicação) e `spotted`
de 35 (11×). As duas classes com menos recortes únicos são exatamente as duas
com AP baixo; as fortes vêm das fotos reais, com 122 cada. O gargalo agora é
**quantidade de grão real distinto**, não escolha de fonte nem hiperparâmetro.

Dois caminhos, na ordem de custo-benefício:
1. **Corrigir os rótulos de `immature` no FT1** (`aprendizado_ativo.ipynb`) —
   devolve 122 recortes bons e mata o problema na raiz. Mais barato que capturar.
2. **Capturar mais grãos de `immature` e `spotted`** com o `vigil_deck` — alvo
   de pelo menos ~100 únicos por classe, pra igualar o que as fotos reais já têm.

### Dois resultados negativos registrados no caminho

1. **Balancear o pool de recorte do *train* por classe** (não só o val) criou
   viés pró-defeito: as cenas de treino ficaram ~20% por classe enquanto o
   vídeo real é majoritariamente `intact`, e isso piorou o resultado (FT1_v2
   ainda foi melhor que FT2_v2, mas o balanceamento de train foi retirado da
   receita de produção — só o *val* deve ser balanceado, o *train* deve
   espelhar o prior real).
2. **FT2 sozinho (só capturas) perde pro FT1 sozinho.** Cena sintética feita de
   recorte de capturas já recortadas (recorte de recorte, fundo gerado) é dado
   mais artificial que fotos reais + frames de vídeo real — treinar só nele
   apaga o que o estágio anterior aprendeu.

### Bugs de medição encontrados no caminho (não do modelo)

Vale registrar porque quase levaram a descartar o RF-DETR por engano:

- **Off-by-one no mapeamento de classe** na primeira rodada de avaliação em
  vídeo: `category_id` do COCO (1-indexado) usado direto contra `class_id` do
  rfdetr (0-indexado) — deslocava todo rótulo em 1 e tornava `spotted`
  inalcançável. Corrigido detectando a base pelos ids realmente vistos no vídeo.
- **`sv.ByteTrack` depreciado** (supervision ≥ 0.28) devolve `tracker_id` vazio
  em silêncio — zerava todos os votos mesmo com o modelo detectando 10
  caixas/frame. Corrigido com fallback pro `ByteTrackTracker` do pacote
  `trackers`.
- **`model.inference(dtype=torch.float16)` não ajudava** (38 ms vs 36 ms em
  eager) e uma vez chegou a zerar detecções por causa do trace não generalizar
  pro shape do vídeo — desligado por padrão (`USE_FP16=False`); o número de
  latência que importa é o do TensorRT no próprio Jetson, não o eager do Colab.

### Papel atual

| Papel | Modelo | Status |
|---|---|---|
| **Candidato a edge (Jetson Orin Nano)** | **RF-DETR Small — FT4 (fonte por classe)** | campeão da família RF-DETR; export ONNX pronto (`soja_rfdetr_small_CAMPEAO.onnx`) |
| Segunda opinião | RF-DETR Small — FT3 (replay) | bom, mas ainda confundia `immature` × `intact` |
| Pendente | Engine TensorRT no Jetson físico | `.engine` precisa ser gerado no próprio aparelho; fps do `trtexec` decide se entra em produção |
| Comparação pendente | RF-DETR Small (FT3) vs YOLO11n/s destilado do 11x | mesmo vídeo, mesmo pipeline — quem for melhor em qualidade E rodar em tempo real no Orin Nano vence |

Ressalva de sempre: "melhor no vídeo" aqui é avaliação visual de um dono que já
viu muitos desses comparativos — não benchmark rotulado, e o vídeo de teste é
majoritariamente `intact`. Falta um vídeo de lote **propositalmente ruim**
(defeito conhecido) pra medir recall de defeito de verdade antes de confiar
nesse candidato em produção.
