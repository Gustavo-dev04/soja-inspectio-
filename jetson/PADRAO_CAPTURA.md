# Padrão de captura — Vígil.ia

> Este documento define o **rig de inspeção**. Ele existe porque *domain shift* é o
> gargalo estrutural do projeto: toda vez que câmera, luz, fundo ou distância
> mudaram, o modelo perdeu acurácia e foi preciso recapturar tudo.
>
> **Regra:** dado capturado fora deste padrão não entra no dataset. Se o padrão
> mudar, ele vira uma **versão nova** (v2, v3…) e o dataset antigo não se mistura
> com o novo.

---

## 0. A câmara e a meta de vazão

O rig deixou de ser bancada e virou **câmara de inspeção com movimento**, com uma
meta numérica que dita todo o resto:

| Item | Valor |
|---|---|
| Câmara | **100 mm de faixa** (transversal ao movimento) × **150 mm de curso** |
| Meta de vazão | **0,5 t/dia em 14-16 h** = **~33 kg/h** = **~58 grãos/s** |
| Movimento | esteira (velocidade constante) — ver §7 se for calha ou queda livre |
| Entrega | **classificar + laudo**. Sem separação física nesta fase |
| Decisão de produto | **premium × não-premium** (as 5 classes continuam sendo o meio) |

A conta que amarra tudo, e que `calcular_vazao.py` resolve:

```
grãos/s = largura_faixa_mm × velocidade_mm/s ÷ 96 mm²
          (96 mm² = (7 mm × 1,4)², o espaço de um grão em monocamada)
```

Grão de soja ≈ **0,16 g** (peso de mil grãos 120-200 g), então 0,5 t/dia são
**3,1 milhões de grãos**. Numa faixa de 100 mm, a esteira precisa andar a
**~56 mm/s**. Isso não é rápido — o limite do sistema não é a esteira, é o
rastreamento (§0b).

### 0b. O limite real: rastreamento, não fps

Existem **dois** tetos de velocidade, e o menor manda:

| limite | do que depende | valor no rig |
|---|---|---|
| **Varreduras** | o veredito só trava depois de 8 observações do mesmo grão | 97 mm/s |
| **Rastreamento** | o grão precisa ser reconhecido de uma varredura para a outra | **69 mm/s** |

O de rastreamento é o que aperta, e ele tem dois regimes:

- **IoU cru** (o que o app fazia): precisa de *sobreposição* entre a caixa de uma
  varredura e a da seguinte. Duas caixas de 7 mm deslocadas de 4 mm já têm IoU
  0,27, abaixo do mínimo de 0,30 → **limite de 3,77 mm por varredura = 53 mm/s**,
  ou seja, **abaixo dos 56 mm/s que a meta exige**. O grão trocaria de ID no meio
  da travessia, zerando os votos e inflando a contagem, sem erro nenhum na tela.
- **Com compensação de movimento** (`--esteira`): a caixa é prevista pela
  velocidade antes de casar, então sobreposição deixa de importar. Sobra só a
  exigência de **não-ambiguidade**: o grão não pode andar mais que metade do
  espaçamento até o vizinho, senão o vizinho da frente fica mais perto do que o
  grão andou. **Limite de 4,9 mm por varredura = 69 mm/s.**

O segundo limite é físico (ambiguidade de abertura), não falta de código: nenhum
algoritmo desempata isso sem outra fonte de informação.

**Resultado: 69 mm/s → 41 kg/h, contra a meta de 33 kg/h — folga de 1,23×.**

## 1. Hardware do rig

| Item | Especificação |
|---|---|
| Câmera | **IMX219** — 8 MP, 3280×2464, mount **M12/CS destacável** (lente trocável) |
| Cabo | **CSI 22-pin** nativo, pitch 0,5 mm — o Orin Nano usa 22-pin (o Jetson Nano antigo usava 15-pin; é a origem da confusão em anúncios genéricos) |
| Lente | M12, foco manual ajustável, **FOV ~120°** — ver §1b/§1c |
| Distância de trabalho | **9,3 cm** (define os 12,7 px/mm — ver §1c) |
| Iluminação | Ring light LED, 144 LEDs, **6500-7000 K**, ~12000 lux, 5 W, **USB DC 5 V**, diâmetro interno ajustável 30-61 mm |
| Fundo | **Cartolina preta fosca** (não reflexiva) |
| Câmara | Fechada — sem luz ambiente entrando |

Decisões já fechadas e o porquê (não reabrir sem motivo novo):

- **RGB com filtro IR-cut, não NoIR.** A NoIR foi cogitada e descartada: sem o
  corte de IR a cor fica distorcida (tom rosado/roxo), e **cor é sinal relevante**
  para classificar dano no grão. NIR "de verdade" fica a cargo do sensor
  espectral dedicado (AS7265x), não de uma câmera sem filtro.
- **Ring light USB 5 V**, e não AC ou 12 V: casa com a tensão da Jetson e permite
  liga/desliga por MOSFET de nível lógico, sem isolamento de alta tensão.
- **6500-7000 K** deliberado: branco neutro/frio para fidelidade de cor. LED mais
  quente amarelaria a leitura do grão.

## 1b. ⚠️ A lente de 120° provavelmente é larga demais — conferir antes do flange

Com a distância fechada em 15 cm, dá para calcular o campo de visão
(`python3 calcular_optica.py`). O resultado pede atenção:

| | 120° @ 15 cm |
|---|---|
| Campo de visão | **415 × 312 mm** |
| Grão de 7 mm na captura (1640×1232) | 28 px |
| **Grão na entrada do modelo (704×704)** | **16 px** |
| Grãos que caberiam no quadro | ~1000 |

O modelo é treinado com grãos de **60-150 px** na entrada (o canvas das cenas
sintéticas é igual à entrada do modelo — ver `SIZE = RES` no notebook). A 16 px,
o grão chega **~4× menor** do que ele aprendeu — e a textura que separa `spotted`
de `skin-damaged` não sobrevive a essa escala.

**O mount M12 destacável resolve** — trocando só a lente, mantendo os 15 cm:

| FOV diagonal | focal | campo útil | grão no modelo | grãos/quadro |
|---|---|---|---|---|
| 120° (atual) | 1,3 mm | 312 mm | 16 px | ~1000 |
| 60° | 4,0 mm | 104 mm | 47 px | 112 |
| **45°** | **5,5 mm** | **75 mm** | **66 px** | **58** |
| **~30°** | **~8,6 mm** | **48 mm** | **102 px** | **24** |
| 20° | 13 mm | 32 mm | 155 px | 10 |

**30-45° (lente M12 de 5,5-8,6 mm) é a faixa bem casada** com o alvo de 60-150 px.

> O raciocínio original — 120° em vez de 160° para controlar distorção — está
> certo na direção. Só que, nesta distância, o eixo que mais pesa não é
> distorção, é **densidade de pixel por grão**.

## 1c. ✅ Como fazer a 120° render sem trocar lente: **ROI no sensor cheio**

Como o hardware já está comprado, a solução é por software — e é melhor do que
parece. **O problema nunca foi o sensor, foi o *downscale*.**

Reduzir o quadro inteiro de 1640 px para a entrada do modelo é o que espreme o
grão até 16 px. Mas o IMX219 tem 3280×2464, e a distância da câmera decide quanto
disso vira px/mm:

```
câmera a 9,3 cm  ->  campo 258 mm em 3280 px  ->  12,7 px/mm
2 recortes de 704 px, com 140 px de sobreposição
                 ->  1268 px = os 100 mm da faixa, em escala 1:1
grão de 7 mm     ->  ~89 px   (DENTRO do alvo 60-150)
```

```bash
python3 vigil_jetson.py --camera csi --roi 704 --tiles 2 --esteira
```

| | binado + downscale | 1 recorte a 15 cm | **2 recortes a 9,3 cm** |
|---|---|---|---|
| Captura | 1640×1232 @30 fps | 3280×2464 @21 fps | 3280×2464 @21 fps |
| Faixa coberta | 312 mm | 89 mm | **100 mm (a câmara inteira)** |
| Grão no modelo | 16 px | 55 px | **89 px** |
| Reescala | 1232 → 704 | nenhuma (1:1) | **nenhuma (1:1)** |
| Inferências/varredura | 1 | 1 | 2 |

**Por que recortar em vez de reduzir:** no ROI 1:1 o grão em pixels é fixado pela
**óptica** — a entrada do modelo não muda o detalhe do grão, muda a **área**.

**Por que 2 recortes e não 1:** com 1 recorte a faixa de 100 mm só cabe a 15 cm
de distância, e aí o grão cai para 55 px. Com 2 recortes dá para aproximar até
9,3 cm, cobrir a mesma faixa e **dobrar o detalhe linear**. Como o critério de
sucesso é **recall de defeito**, e defeito é textura (mancha, casca), pixel em
cima do defeito é exatamente o que compra recall. O custo são 2 inferências por
varredura, que baixam a taxa de 28 para 14 varreduras/s — e a conta de §0b mostra
que ainda sobra folga.

**Sobreposição de 140 px (11 mm, ~1,6 grãos) é obrigatória**, senão o grão em
cima da costura sai cortado nos dois recortes e nenhum dos pedaços vira detecção
boa. O app junta os recortes em coordenadas globais e aplica NMS na faixa de
sobreposição, para o grão da costura contar **uma vez só**.

> Trocar a lente para 30-45° continua sendo a solução *mais limpa* — usaria o
> sensor inteiro e dispensaria os recortes. Mas com o ROI em tiles o rig atual
> **já entrega a meta**, sem comprar nada.

### Calibre com a régua antes de fechar o valor do ROI

Os 7,9 px/mm vêm da conta, que assume FOV **diagonal**; se os 120° forem
horizontais o campo é maior, e spec de M12 barata é aproximada. A medição decide:

```bash
python3 calibrar_rig.py            # régua no fundo, na distância de trabalho
```

Ele mede px/mm (clicando em dois pontos da régua), analisa a exposição
(estouro, fundo cravado) e **diz qual `--roi` usar**.

## 2. Configuração da câmera — o item mais crítico

**Exposição e balanço de branco TÊM que estar travados.** Em automático, a câmera
compensa sozinha entre sessões: a mesma soja fotografada hoje e amanhã vira
imagem diferente, e isso recria exatamente o domain shift que o rig deveria
eliminar. É a forma mais silenciosa de invalidar um dataset inteiro — nada
falha, o dado só fica inconsistente.

Os valores ficam em `CSI_TRAVAS`, no `vigil_jetson.py`:

```python
CSI_TRAVAS = {
    'wbmode': 0,                                 # balanço de branco manual
    'awblock': 'true',                           # trava o AWB
    'aelock': 'true',                            # trava a exposição
    'exposuretimerange': '"13000000 13000000"',  # ns — valor FIXO, não faixa
    'gainrange': '"1 1"',                        # ganho analógico fixo
    'ispdigitalgainrange': '"1 1"',              # ganho digital fixo
}
```

### Como calibrar a exposição (uma vez, e anotar)

1. Rode a calibração com o automático ligado, só para achar o ponto:
   `python3 calibrar_rig.py --sem-trava`  (tecla **e** mostra o histograma)
2. Ajuste o ring light e o `exposuretimerange` até que:
   - o **fundo** fique bem escuro, mas **não** cravado em 0
   - o **grão** fique bem exposto, **sem pixel estourado em 255**
3. Fixe o valor em `CSI_TRAVAS` e **anote neste documento** (§6).

> Pixel estourado é informação **perdida** — nenhum modelo recupera textura de
> uma região saturada. Se o ring light marcar um anel de brilho no grão (soja é
> levemente lustrosa), use um **difusor**.

### Resolução

Para o rig, use **3280×2464 @ 21 fps** — o modo cheio, que é o que torna o ROI
1:1 possível (§1c). O `--roi 704 --tiles 2` recorta a faixa da câmara sem reescalar.

O modo binado **1640×1232 @ 30 fps** mantém o mesmo FOV e serve para enquadrar e
calibrar, mas não para capturar dataset: o downscale até a entrada do modelo é
justamente o que espreme o grão até 16 px.

Os modos 1080p **recortam** o sensor, ou seja, mudam o enquadramento e quebrariam
a padronização entre sessões — não usar.

## 3. Distorção da lente de 120°

Grande angular distorce as bordas (barril). Um grão no canto tem forma diferente
do mesmo grão no centro, e o modelo aprende isso como se fosse variação real.

**Solução adotada: recorte quadrado central** (`--quadrado`). Resolve duas coisas
de uma vez:

- descarta as bordas, onde a distorção é pior;
- elimina o *letterbox* — o modelo come 704×704 quadrado, então uma imagem 4:3
  gastaria ~25% da entrada em barra preta.

```bash
python3 vigil_jetson.py --camera csi --roi 704     # modo do rig (ver §1c)
```

## 4. Geometria — o que precisa ficar fixo

| Parâmetro | Como fixar |
|---|---|
| **Distância câmera-fundo** | Suporte rígido. **Meça e anote** (§6). |
| **Foco** | Ajuste uma vez e **trave fisicamente** (fita/trava). Um esbarrão no anel de foco muda o domínio. |
| **Posição do ring light** | Fixa em relação à câmera e ao fundo. |
| **Enquadramento** | Marque o fundo para os grãos caírem sempre na mesma área útil. |

## 5. Protocolo de captura

- Grãos **espalhados sem se tocar** — encostados atrapalham a caixa.
- Densidade de monocamada: ~64 grãos por varredura nos dois recortes.
- Velocidade da esteira: **56 mm/s** para a meta; **69 mm/s** é o teto (§0b).

### Como rotular sem desenhar caixa nenhuma

Este é o truque que torna o dataset abundante barato, e ele só funciona porque a
câmara é controlada:

- **Bandeja de classe única → rótulo de graça.** Passe um lote de UMA classe por
  vez. A caixa sai do Otsu (fundo preto controlado, >98% de acerto histórico) e a
  classe vem da pasta. Dezenas de milhares de caixas corretas, zero anotação
  manual. É o mesmo mecanismo do `autotreino_video_v4.ipynb`.
- **Bandeja mista → validação anotada à mão.** Algumas centenas de grãos, as 5
  classes balanceadas. É o **único** conjunto que mede recall de verdade — sem
  ele não dá para distinguir modelo bom de modelo viciado em dizer "intacto".
- Grave também um **lote propositalmente ruim**, com defeito conhecido.
- Meta por classe: **~120 grãos únicos no mínimo**. `immature` e `spotted` são as
  duas classes fracas do modelo atual e as que mais ganham com o rig.

```bash
# sessão de captura, com laudo periódico
python3 vigil_jetson.py --camera csi --roi 704 --tiles 2 --esteira \
                        --out sessao.mp4 --laudo sessao.json
```

## 6. Ficha do rig — PREENCHER e manter atualizada

> Sem estes valores anotados, o rig não é reproduzível depois de desmontado.

| Parâmetro | Valor | Data |
|---|---|---|
| **px/mm medido** (régua) | `_______` | |
| **`--roi` em uso** | `_______` | |
| Janela útil resultante | `_______ × _______ mm` | |
| `exposuretimerange` | `_______` | |
| `gainrange` | `_______` | |
| Distância câmera → fundo | `_______ cm` | |
| Altura/posição do ring light | `_______` | |
| Difusor no ring light? | ( ) sim ( ) não | |
| Área útil marcada no fundo | `_______ × _______ cm` | |
| Versão do padrão | **v1** | |

## 7. O que muda no modelo depois disso

Este rig é um **domínio novo** — diferente das fotos de celular que treinaram os
modelos atuais. Portanto:

1. Os modelos de hoje (FT4 e anteriores) **vão degradar** neste rig. Isso é
   esperado, não é defeito.
2. É preciso **capturar dataset novo** aqui e **re-treinar**. Essa passa a ser a
   rodada de fine-tuning que realmente conta.
3. Só **depois** disso os números de acurácia significam alguma coisa — e aí os
   limiares de voto (`RATIOS`) e de confiança precisam ser recalibrados neste
   domínio.

Até lá, meça apenas o que independe de domínio: fps, latência, estabilidade de
caixa e de rastreamento.

### 7b. Se o movimento não for esteira

Todo o dimensionamento acima assume **esteira**: velocidade constante, grão
parado em relação à correia. É o caso mais fácil e o mais barato. O que muda nos
outros dois:

| | esteira | calha vibratória | queda livre |
|---|---|---|---|
| Velocidade | constante, escolhida | irregular, quica | ~1,7 m/s após 15 cm de queda |
| Exposição | longa, sem borrão | curta (o grão salta) | **strobe obrigatório** |
| Pose entre varreduras | estável | **gira** — o IoU e a compensação sofrem | gira e acelera |
| Varreduras por grão | 11 na configuração atual | menos, e irregular | **1-2**: a regra de 8 não vale |
| Veredito | por voto acumulado | por voto, com mais ruído | teria de ser **por quadro único** |

Ou seja: calha vibratória exige revisar os parâmetros do rastreamento (a rotação
entre varreduras derruba o IoU mesmo com compensação); **queda livre invalida o
modelo de votação inteiro** e viraria outro projeto — classificação por quadro
único, com strobe e sincronismo, sem tracking. Se a decisão mudar para queda
livre, este documento vira v2 e o dataset não se mistura.

## 8. Pendências de montagem física

- [ ] **Calibrar com a régua** (`calibrar_rig.py`) — confirma os 12,7 px/mm e a
      exposição; depois rode `calcular_vazao.py --dist <medido>` para conferir
      que a vazão fecha com a distância real, não com a da conta
- [ ] Definir o acionamento da esteira e **medir a velocidade real** (cronômetro
      num trecho marcado); ela precisa ficar abaixo de 69 mm/s
- [ ] Definir/imprimir a **flange 3D** acoplando ring light + lente + parede da
      câmara, mantendo alinhamento no eixo óptico e a distância fixa de 9,3 cm
- [ ] Conferir a **orientação do cabo CSI 22-pin** na instalação — os contatos
      podem precisar de inversão de lado dependendo do cabo; é erro comum
- [ ] Travar o foco fisicamente depois de ajustado
- [ ] Preencher a ficha do rig (§6)

## 9. Futuro: canal NIR (não fechado)

Planejado, fora do escopo do rig v1:

- Sensor **AS7265x** (SparkFun), 410-940 nm em 18 canais
- LED NIR cobrindo **~660-940 nm**. *(Um LED de 1000 nm chegou a ser sugerido e
  foi descartado: fica na borda extrema da faixa útil — os canais NIR relevantes
  do AS72651/AS72652 ficam entre 610-860 nm.)*
- **RGB e NIR não devem acender juntos**: pulsar via GPIO/MOSFET, um de cada vez,
  para evitar contaminação cruzada entre imagem e leitura espectral. A câmera RGB
  tem IR-cut de fábrica e não deveria ser afetada, mas pulsar segue recomendado.

Quando o NIR entrar, ele **muda o rig** — logo, vira padrão **v2** e exige
recaptura. Não misture dataset v1 com v2.
