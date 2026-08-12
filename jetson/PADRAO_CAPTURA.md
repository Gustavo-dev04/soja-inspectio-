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
| Câmera | **IMX219** — 8 MP, 3280×2464, lente 120°, foco ajustável, conector CSI |
| Placa | Jetson Orin Nano (conector CSI da própria placa) |
| Iluminação | **Ring light, 6500 K** |
| Fundo | **Cartolina preta fosca** (não reflexiva) |
| Câmara | Fechada — sem luz ambiente entrando |

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

1. Rode com o automático ligado só para achar o ponto:
   `python3 vigil_jetson.py --camera csi --csi-sem-trava`
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
python3 vigil_jetson.py --camera csi --quadrado
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
python3 vigil_jetson.py --camera csi --quadrado --out sessao.mp4
```

## 6. Ficha do rig — PREENCHER e manter atualizada

> Sem estes valores anotados, o rig não é reproduzível depois de desmontado.

| Parâmetro | Valor | Data |
|---|---|---|
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
