# Pesquisa — backbones vendáveis e o qps de cada um no Jetson

**Pergunta:** quais backbones de detecção podem entrar num produto **vendido**,
e quantos qps cada um entrega no **Jetson Orin Nano 8 GB** do projeto?

Levantamento de 2026-09. Licenças conferidas **no arquivo `LICENSE` de cada
repositório**, não em blog ou tabela de terceiro. Números de velocidade: dois
**medidos por nós no aparelho**, o resto projetado por um método que está
descrito e é conferível.

> Reproduzir a tabela: `python3 jetson/projetar_qps.py --vazao`

---

## 1. Resumo — o que decidir

1. **RF-DETR-L continua sendo a escolha.** É Apache 2.0, é o melhor Apache em
   transferência de domínio (RF100-VL 62,2), e a projeção dá **~28 qps** no
   aparelho a 15 W — que vira **42 kg/h** no rig, contra a meta de 33 kg/h.
   Não há motivo para trocar.
2. **DEIMv2 está fora.** Apesar de ser o melhor da lista em papel, a licença é
   **não-comercial**. Detalhe em §3.
3. **O maior ganho de velocidade disponível não é trocar de modelo, é ligar o
   MAXN_SUPER** — até 1,6× de graça, só mexendo em `nvpmodel`. Ainda não medido.
4. **Se um dia faltar vazão, a saída é D-FINE**, não baixar o RF-DETR. Mas o
   qps dele no Orin **não dá para prever** com o que temos — a faixa é de 51 a
   99 qps para o `S`, e isso é largo demais para decidir. Precisa medir.
5. **Velocidade não é o gargalo do projeto.** Todo candidato Apache bate a meta
   de vazão. O que separa um do outro é acurácia em domínio próprio e quantos
   pixels o grão tem quando chega — ver a armadilha em §6.

---

## 2. A referência de latência

Todas as linhas abaixo vêm do **mesmo arnês de medição** — NVIDIA T4, TensorRT,
FP16, batch 1 — publicado pela Roboflow no `README` do `rf-detr`. Isso importa
mais do que parece: latência medida por fornecedores diferentes, em máquinas e
protocolos diferentes, **não é comparável**, e somar essas linhas numa tabela só
é honesto quando saíram do mesmo lugar.

A coluna que interessa ao projeto é **RF100-VL**, não COCO. RF100-VL mede
*fine-tuning em dataset pequeno e customizado* — exatamente o nosso caso. COCO
mede capacidade em 80 classes genéricas, que não é o que vamos pedir do modelo.

| Modelo | Res | Params | COCO AP | **RF100-VL AP** | ms (T4) | Licença |
|---|---|---|---|---|---|---|
| RF-DETR-N | 384 | 30,5 M | 48,4 | 57,7 | 2,3 | Apache 2.0 |
| RF-DETR-S | 512 | 32,1 M | 53,0 | 60,2 | 3,5 | Apache 2.0 |
| RF-DETR-M | 576 | 33,7 M | 54,7 | 61,2 | 4,4 | Apache 2.0 |
| **RF-DETR-L** | **704** | **33,9 M** | **56,5** | **62,2** | **6,8** | **Apache 2.0** |
| RF-DETR-XL / 2XL | 700 / 880 | 126 M | 58,6 / 60,1 | 62,9 / 63,2 | 11,5 / 17,2 | ❌ PML 1.0 |
| LW-DETR-T … L | 640 | 12–47 M | 42,9–56,1 | 57,1–61,5 | 1,9–6,9 | Apache 2.0 |
| D-FINE-N | 640 | 3,8 M | 42,7 | 58,2 | 2,1 | Apache 2.0 |
| D-FINE-S | 640 | 10,2 M | 50,6 | 60,3 | 3,5 | Apache 2.0 |
| D-FINE-L | 640 | 31,0 M | 57,2 | 61,6 | 7,5 | Apache 2.0 |
| D-FINE-X | 640 | 62,0 M | 59,3 | 62,2 | 11,5 | Apache 2.0 |
| YOLO11-S / X | 640 | 9,4 / 56,9 M | 44,4 / 50,9 | 56,2 / 56,2 | 3,2 / 10,5 | ❌ AGPL-3.0 |
| YOLO26-S / L | 640 | 9,4 / 25,3 M | 47,7 / 54,1 | 57,0 / 59,3 | 2,6 / 5,7 | ❌ AGPL-3.0 |

**Viés declarado:** a tabela é publicada pela Roboflow, que é dona do RF-DETR.
Pesa a favor da credibilidade que os concorrentes aparecem bem — D-FINE-S
empata com RF-DETR-S em RF100-VL na mesma latência, e D-FINE-N ganha do
RF-DETR-N. Ainda assim, para uma decisão de compra valeria reproduzir. Para
decidir arquitetura de um projeto acadêmico, serve.

**Observação que muda o discurso comercial:** os YOLO (AGPL) vão bem em COCO e
mal em RF100-VL. Em transferência para dataset próprio — o nosso caso — o
YOLO11-X (56,9 M params) fica em 56,2, **abaixo do D-FINE-N com 3,8 M**. A
família que não podemos usar por licença também não é a que queremos.

---

## 3. Licenças — conferidas arquivo por arquivo

| Projeto | Arquivo lido | Veredito |
|---|---|---|
| **RF-DETR** (`roboflow/rf-detr`) | `LICENSE` | ✅ Apache 2.0 — pacote `rfdetr` e modelos N/S/M/L |
| RF-DETR **XL / 2XL** | `README` | ❌ **PML 1.0** — pacote `rfdetr_plus`, separado |
| **D-FINE** (`Peterande/D-FINE`) | `LICENSE` | ✅ Apache 2.0 |
| **DEIM** v1 (`ShihuaHuang95/DEIM`) | `LICENSE` | ✅ Apache 2.0 |
| **DEIMv2** (`Intellindust-AI-Lab/DEIMv2`) | `LICENSE.md` | ❌ **não-comercial** |
| **LW-DETR** (`Atten4Vis/LW-DETR`) | `LICENSE` | ✅ Apache 2.0 |
| **YOLOX** (`Megvii-BaseDetection/YOLOX`) | `LICENSE` | ✅ Apache 2.0 |
| **Ultralytics** (YOLOv5/v8/v11/26) | — | ❌ AGPL-3.0 |

### Três armadilhas que o levantamento encontrou

**1. O nome da família não carrega a licença.** `RF-DETR` é Apache 2.0 **até o
Large**; XL e 2XL são PML 1.0, num pacote diferente (`rfdetr_plus`). Escolher
"RF-DETR" não é escolher Apache — escolher a *variante* é.

**2. DEIMv2 não é o DEIM.** O DEIM v1 é Apache 2.0. O **DEIMv2 tem licença
própria**, que diz com todas as letras:

> *"No rights are granted for Commercial Use."* — e a definição de Commercial
> Use inclui explicitamente *"incorporating DEIMv2 into commercial hardware or
> software"*, que é precisamente o que o Vígil.ia seria.

O README exibe métricas excelentes (DEIMv2-S: 50,9 AP com 9,7 M params) e não
destaca isso. A licença restringe também os **pesos**. E o DEIMv2 depende de
DINOv3, que tem licença própria de terceiro por cima. Está fora.

**3. Mesma arquitetura, implementação diferente, licença diferente.** RT-DETR no
repositório original (`lyuwenyu`) é Apache 2.0; **o mesmo RT-DETR rodado pelo
Ultralytics é AGPL**, porque o código é deles. Qualquer `from ultralytics
import …` é AGPL, não importa o modelo carregado.

> Levantamento técnico de licenças, não parecer jurídico.

---

## 4. Os dois números medidos no aparelho

Base de tudo que vem depois. Jetson Orin Nano 8 GB, JetPack, TensorRT FP16,
modo **15 W**, medidos com `jetson/bench_energia.py` rodando o próprio
`vigil_jetson.py`:

| | RF-DETR-N (384) | RF-DETR-S (512) |
|---|---|---|
| **qps** | **98,2** | **59,3** |
| latência | 10,18 ms | 16,86 ms |
| ocioso | 3,87 W | 3,99 W |
| sob carga | 9,39 W | 10,62 W |
| custo do modelo | +5,51 W | +6,63 W |
| tj máx | 52,0 °C | 53,0 °C |

Teste contínuo de 30 min no Small: 10,48 W, tj 56,4 °C, **59,3 qps sem queda** —
sem throttling, 39 °C de margem até o limite.

---

## 5. Como o T4 vira Orin — e por que não é uma constante

Dividindo o que medimos pela referência T4:

| | ms T4 | ms Orin (medido) | razão |
|---|---|---|---|
| RF-DETR-N | 2,3 | 10,18 | 4,43× |
| RF-DETR-S | 3,5 | 16,86 | 4,82× |

A razão **não é fixa** — sobe com o tamanho do modelo. Ajustando a reta pelos
dois pontos:

```
ms_orin = 5,567 × ms_t4 − 2,620
```

O coeficiente linear negativo não é erro de conta: o T4 paga um custo fixo por
chamada que o Orin não paga na mesma proporção, então modelo pequeno "perde
menos" na troca de máquina. A **inclinação de 5,57×** é o que vale para a parte
que é cálculo puro.

**Três verificações independentes, para o RF-DETR-L (704):**

| Método | Resultado |
|---|---|
| reta acima (6,8 ms T4) | 35,2 ms → **28,4 qps** |
| razão direta a 4,8× | 32,6 ms → **30,6 qps** |
| escala por área a partir do **nosso** Small (704/512)² | 31,9 ms → **31,4 qps** |

Três caminhos, faixa de 28–31 qps. O terceiro nem usa o T4 — parte só da nossa
medição, e é legítimo porque o RF-DETR-L é literalmente *a Small a 704 px*
(mesmo encoder `dinov2_windowed_small`, mesmo patch 16; muda a resolução e uma
camada de decoder). **Adotado 28 qps para planejamento**, o mais conservador.

### O limite do método: ele não atravessa arquiteturas

Um ponto de terceiro, num YOLOv8n (CNN pequena) no mesmo Orin Nano: 4,43 ms a
640 em MAXN_SUPER → ~7,1 ms convertido para 15 W, contra 2,5 ms de um modelo
equivalente no T4. Razão: **2,9×** — bem longe dos 4,4–4,8× do RF-DETR, **na
mesma latência de T4**.

A explicação é arquitetural: uma CNN pequena não satura o T4, então a vantagem
de cálculo do T4 nem aparece e a troca de máquina custa pouco. Um ViT de 30 M
satura, e aí a diferença toda se realiza.

**Consequência prática:** aplicar o fator do RF-DETR ao D-FINE **subestimaria**
a velocidade dele, talvez em 1,7×. Por isso a tabela abaixo mostra o D-FINE como
**faixa**, não como número. A faixa é larga porque a nossa ignorância é larga —
fingir precisão ali seria inventar.

---

## 6. A projeção

```
python3 jetson/projetar_qps.py            # qps
python3 jetson/projetar_qps.py --vazao    # qps → kg/h no rig
```

**qps no Orin Nano 8 GB, 15 W, FP16** (só os Apache 2.0):

| Modelo | Res | RF100-VL | ms T4 | qps no Orin | |
|---|---|---|---|---|---|
| RF-DETR-N | 384 | 57,7 | 2,3 | **98,2** | medido |
| RF-DETR-S | 512 | 60,2 | 3,5 | **59,3** | medido |
| RF-DETR-M | 576 | 61,2 | 4,4 | 45,7 | ±10% |
| **RF-DETR-L** | **704** | **62,2** | **6,8** | **28,4** | **±10%** |
| LW-DETR-T | 640 | 57,1 | 1,9 | 125,7 | ±10% |
| LW-DETR-S | 640 | 57,4 | 2,6 | 84,4 | ±10% |
| LW-DETR-M | 640 | 59,8 | 4,4 | 45,7 | ±10% |
| LW-DETR-L | 640 | 61,5 | 6,9 | 27,9 | ±10% |
| D-FINE-N | 640 | 58,2 | 2,1 | 85–164 | a medir |
| D-FINE-S | 640 | 60,3 | 3,5 | 51–99 | a medir |
| D-FINE-M | 640 | 60,6 | 5,4 | 33–64 | a medir |
| D-FINE-L | 640 | 61,6 | 7,5 | 24–46 | a medir |

### E o que isso vira de vazão no rig

Faixa de 100 mm, 2 recortes com 20% de sobreposição, rastreamento com
compensação de movimento, meta de 33 kg/h:

| Modelo | Entrada | **Grão** | Varred/s | Vel. máx | kg/h | Margem |
|---|---|---|---|---|---|---|
| RF-DETR-N | 384 | ⚠ 48 px | 49,1 | 241 mm/s | 144 | 4,33× |
| RF-DETR-S | 512 | 65 px | 29,6 | 145 mm/s | 87 | 2,62× |
| RF-DETR-M | 576 | 73 px | 22,9 | 112 mm/s | 67 | 2,02× |
| **RF-DETR-L** | **704** | **89 px** | **14,2** | **70 mm/s** | **42** | **1,25×** |
| LW-DETR-S | 640 | 81 px | 42,2 | 207 mm/s | 124 | 3,72× |
| LW-DETR-L | 640 | 81 px | 14,0 | 68 mm/s | 41 | 1,23× |

**A coluna "Grão" é a armadilha desta tabela.** Cobrir os mesmos 100 mm com uma
entrada menor obriga a **afastar a câmera**, e o grão chega ao modelo com menos
pixel. O Nano parece um ganho de 4,33× em vazão; na verdade entrega o grão com
48 px, abaixo do piso de ~60 px em que a textura do defeito (mancha, casca
danificada) ainda sobrevive. **Vazão alta com grão pequeno não é ganho, é
troca** — e é uma troca ruim, porque o projeto se vende por *taxonomia de
defeito*, não por tonelagem.

Todo candidato Apache bate a meta de vazão. Nenhum bate o RF-DETR-L em pixels
por grão na mesma faixa. É por isso que a decisão não muda.

---

## 7. O ganho que ainda não foi colhido: MAXN_SUPER

Todas as nossas medições foram feitas em **15 W**. O Orin Nano 8 GB aceita o
modo **MAXN_SUPER**, que sobe o clock da GPU de **635 para 1020 MHz** (1,61×) —
disponível por atualização de software nos devkits existentes, sem trocar
hardware.

Teto teórico, aplicando o clock (o `projetar_qps.py --modo super` calcula):

| | 15 W (medido) | MAXN_SUPER (teto) |
|---|---|---|
| RF-DETR-S | 59,3 qps | ≤ 95 qps |
| RF-DETR-L | ~28 qps | ≤ 46 qps |

É **teto**, não previsão: a parte do modelo limitada por memória não acompanha o
clock, então o ganho real fica abaixo. Custo: a potência sobe ~1,7× — o modelo
passaria de +6,6 W para ~+11 W, e o total do aparelho de ~10,5 W para ~18 W.

**O que falta para colher:**

```bash
sudo nvpmodel -q                  # que modos este aparelho oferece?
sudo nvpmodel -m 2                # MAXN_SUPER (conferir o número em -q)
sudo jetson_clocks
python3 bench_energia.py --engine <engine> --minutos 30
```

Tem que ser medido **dentro da câmara fechada**, não na bancada. A 15 W sobrava
39 °C de margem térmica; a 25 W, num volume fechado com ring light dentro, essa
margem encolhe e o throttling aparece exatamente no teste longo — que é o
regime real de 14–16 h/dia.

---

## 8. O que fazer com isto

**Não muda nada agora.** RF-DETR-L, Apache 2.0, ~28 qps, 42 kg/h, 1,25× de
margem sobre a meta. A decisão de arquitetura está certa e confirmada por um
caminho independente do que a originou.

**Fica registrado como plano B, na ordem:**

1. **MAXN_SUPER** — até 1,6×, sem trocar de modelo nem retreinar. Primeira coisa
   a tentar se faltar vazão. Medir junto o térmico na câmara fechada.
2. **D-FINE-S** — se precisar de muito mais vazão (faixa mais larga, mais kg/h).
   Apache 2.0, RF100-VL 60,3 (contra 62,2 do RF-DETR-L), e com backbone CNN, que
   **quantiza para INT8 de forma muito mais previsível que um ViT** — o alerta
   de regressão INT8 registrado em `DOCUMENTACAO.md` §7 é específico de ViT-S.
   Exige refazer a geometria para entrada de 640 e retreinar do zero.
3. **DEIM v1 (Apache) como receita de treino sobre D-FINE** — ganho de AP sem
   custo de inferência, se o caminho 2 for tomado.

**Descartados e por quê:** DEIMv2 (não-comercial), RF-DETR-XL/2XL (PML 1.0),
toda a família Ultralytics (AGPL-3.0), YOLOX e RTMDet (Apache, mas dominados em
RF100-VL pelos DETR de custo igual), EfficientDet (geração anterior, ordens de
grandeza mais lento no Jetson).

---

## 9. Fontes

- `roboflow/rf-detr` — tabela de benchmark (T4/TensorRT/FP16/bs1, arnês SAB) e licenças por variante: https://github.com/roboflow/rf-detr
- `Peterande/D-FINE` — model zoo e `LICENSE`: https://github.com/Peterande/D-FINE
- `ShihuaHuang95/DEIM` — `LICENSE` Apache 2.0: https://github.com/ShihuaHuang95/DEIM
- `Intellindust-AI-Lab/DEIMv2` — `LICENSE.md` não-comercial: https://github.com/Intellindust-AI-Lab/DEIMv2
- `Atten4Vis/LW-DETR` e `Megvii-BaseDetection/YOLOX` — `LICENSE` Apache 2.0
- Ponto de CNN no Orin Nano (YOLOv8n 640, MAXN_SUPER, 4,43 ms): https://github.com/hokwangchoi/jetson-orin-nano-benchmarks
- Modo Super do Orin Nano (635 → 1020 MHz, JetPack 6.2): https://premioinc.com/blogs/blog/what-is-super-mode-on-nvidia-jetson-orin-nano-and-nx-in-jetpack-6-2-sdk-release
- Medições próprias: `jetson/bench_energia.py` neste repositório
