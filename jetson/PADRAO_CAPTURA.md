# Padrão de captura — Vígil.ia

> Este documento define o **rig de inspeção**. Ele existe porque *domain shift* é o
> gargalo estrutural do projeto: toda vez que câmera, luz, fundo ou distância
> mudaram, o modelo perdeu acurácia e foi preciso recapturar tudo.
>
> **Regra:** dado capturado fora deste padrão não entra no dataset. Se o padrão
> mudar, ele vira uma **versão nova** (v2, v3…) e o dataset antigo não se mistura
> com o novo.

---

## 1. Hardware do rig

| Item | Especificação |
|---|---|
| Câmera | **IMX219** — 8 MP, 3280×2464, mount **M12/CS destacável** (lente trocável) |
| Cabo | **CSI 22-pin** nativo, pitch 0,5 mm — o Orin Nano usa 22-pin (o Jetson Nano antigo usava 15-pin; é a origem da confusão em anúncios genéricos) |
| Lente | M12, foco manual ajustável, **FOV ~120°** — ver §1b, provavelmente precisa trocar |
| Distância de trabalho | **15 cm** (faixa avaliada: 10-30 cm) |
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
| **Grão na entrada do modelo (512×512)** | **11 px** |
| Grãos que caberiam no quadro | ~1000 |

O modelo foi treinado com grãos de **48-120 px** na entrada (60-150 px num canvas
de 640, nas cenas sintéticas). A 11 px, o grão chega **~4× menor** do que ele
aprendeu — e a textura que separa `spotted` de `skin-damaged` não sobrevive a
essa escala.

**O mount M12 destacável resolve** — trocando só a lente, mantendo os 15 cm:

| FOV diagonal | focal | campo útil | grão no modelo | grãos/quadro |
|---|---|---|---|---|
| 120° (atual) | 1,3 mm | 312 mm | 11 px | ~1000 |
| 60° | 4,0 mm | 104 mm | 34 px | 112 |
| 45° | 5,5 mm | 75 mm | 48 px | 58 |
| **~30°** | **~8,6 mm** | **48 mm** | **74 px** | **24** |
| 20° | 13 mm | 32 mm | 113 px | 10 |

**~30° (lente M12 de ~8 mm) é a mais bem casada:** 74 px por grão fica no meio do
alvo, e ~24 grãos por quadro bate com a faixa das cenas de treino (6-25).

> O raciocínio original — 120° em vez de 160° para controlar distorção — está
> certo na direção. Só que, nesta distância, o eixo que mais pesa não é
> distorção, é **densidade de pixel por grão**.

## 1c. ✅ Como fazer a 120° render sem trocar lente: **ROI no sensor cheio**

Como o hardware já está comprado, a solução é por software — e é melhor do que
parece. **O problema nunca foi o sensor, foi o *downscale*.**

Reduzir o quadro inteiro de 1640 px para os 512 px do modelo é o que espreme o
grão até 11 px. Mas o IMX219 tem 3280×2464:

```
campo 415 mm em 3280 px            ->  7,9 px/mm
recorte de 512×512 no centro       ->  janela de 65 × 65 mm
grão de 7 mm                       ->  ~55 px   (DENTRO do alvo 48-120)
```

Recortando 512×512 direto do centro do sensor cheio, **em escala 1:1 e sem
reescalar nada**, o grão chega ao modelo do tamanho certo. E o centro é
justamente onde a lente de 120° distorce menos — o recorte resolve escala e
distorção de uma vez.

```bash
python3 vigil_jetson.py --camera csi --roi 512
```

| | binado + downscale | **ROI no sensor cheio** |
|---|---|---|
| Captura | 1640×1232 @30 fps | 3280×2464 @21 fps |
| Janela útil | 312 mm | **65 mm** |
| Grão no modelo | 11 px | **~55 px** |
| Reescala | 1232 → 512 (perde detalhe) | **nenhuma (1:1)** |
| Grãos por quadro | ~1000 (inútil) | **~40** |

**O custo:** a janela de inspeção passa a ser 65 × 65 mm, e usa-se ~2% dos
pixels do sensor. Para bancada é ótimo (~40 grãos por quadro bate com as cenas
de treino de 6-25); para vazão, move-se a bandeja.

> Trocar a lente para ~30° continua sendo a solução *mais limpa* — usaria o
> sensor inteiro para a mesma janela. Mas com o ROI o rig atual **já funciona**,
> sem comprar nada.

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

Use **1640×1232 @ 30 fps** — modo binado do IMX219, que mantém o **FOV completo**.
Os modos 1080p **recortam** o sensor, ou seja, mudam o enquadramento e quebrariam
a padronização entre sessões.

## 3. Distorção da lente de 120°

Grande angular distorce as bordas (barril). Um grão no canto tem forma diferente
do mesmo grão no centro, e o modelo aprende isso como se fosse variação real.

**Solução adotada: recorte quadrado central** (`--quadrado`). Resolve duas coisas
de uma vez:

- descarta as bordas, onde a distorção é pior;
- elimina o *letterbox* — o modelo come 512×512 quadrado, então uma imagem 4:3
  gastaria ~25% da entrada em barra preta.

```bash
python3 vigil_jetson.py --camera csi --roi 512     # modo do rig (ver §1c)
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
- **10 a 40 grãos** por quadro (o modelo foi treinado com cenas de 6-25).
- Misture as classes no mesmo quadro; grave também lotes quase só `intact` e
  lotes com muito defeito.
- Se for simular esteira, mova a bandeja **devagar** (borrão real ajuda o treino).
- Meta por classe: **~120 grãos únicos**. Hoje `immature` tem 56 e `spotted` 35 —
  são as duas classes fracas do modelo atual.
- Grave também um **lote propositalmente ruim**, com defeito conhecido: sem ele
  não dá para medir *recall* de defeito, só a taxa de acerto em lote bom.

```bash
# coleta (salva recorte + revisao.csv no formato do aprendizado_ativo.ipynb)
python3 vigil_jetson.py --camera csi --roi 512 --out sessao.mp4
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

## 8. Pendências de montagem física

- [ ] **Calibrar com a régua** (`calibrar_rig.py`) — define o `--roi` e a exposição
- [ ] Definir/imprimir a **flange 3D** acoplando ring light + lente + parede da
      câmara, mantendo alinhamento no eixo óptico e a distância fixa de 15 cm
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
