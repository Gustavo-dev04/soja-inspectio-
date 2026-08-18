# Vígil.ia no Jetson Orin Nano — RF-DETR Small via TensorRT

Leva o modelo campeão da família RF-DETR (**FT4**, fonte de recorte por classe)
pro Jetson e responde a pergunta que motivou o experimento inteiro:
**cabe em tempo real num Orin Nano?**

---

## 0. Cartão SD novo? Rode isto primeiro

```bash
chmod +x setup_jetson.sh && ./setup_jetson.sh
```

Faz o setup inteiro de uma imagem recém-gravada: confere JetPack/CUDA/TensorRT
e espaço em disco, instala o que falta (`pip3`, OpenCV **do apt** — o do pip não
traz GStreamer e a CSI não abre —, `cuda-python`), **reconstrói as engines** de
todo `.onnx` que estiver na pasta e verifica cada uma. É idempotente: rodar de
novo não estraga nada.

⚠️ **Trocar o cartão SD troca a versão do TensorRT, e engine antiga não serve
mais.** Guarde os `.onnx`, não os `.engine` — o script reconstrói em minutos.

## 0a. Apresentação: um comando (ou um clique)

```bash
./demo.sh              # acha a câmera sozinho e abre em tela cheia
./demo.sh nano         # usa a engine do nano em vez do small
./demo.sh --video x.mp4 # roda um arquivo gravado
./demo.sh --atalho     # cria o ícone clicável na área de trabalho
```

Feito para o pior cenário de demo: o IP do celular mudou de manhã, o Wi-Fi do
auditório é outro, e você tem trinta segundos com a plateia olhando. Ele procura
a câmera em quatro caminhos — `VIGIL_CAMERA`, o último IP que funcionou, o
gateway (quando o celular é o roteador, ele **é** o gateway) e uma varredura da
rede local — guarda o que deu certo, e **se nada responder cai num `.mp4`
gravado** em vez de travar na sua frente.

Usa `--conf 0.10` de propósito: a demo com celular na mão é fora do domínio de
treino, e a confiança padrão de 0.35 esconderia detecções válidas.

## 0b. Antes de tudo: o `.pth` não é o arquivo que roda aqui

| arquivo | onde vive | serve pra |
|---|---|---|
| `.pth` | Colab / Drive | treinar e avaliar (precisa de `rfdetr` + torch) |
| `.onnx` | ponte | formato neutro, gerado **no Colab** |
| `.engine` | **só neste Jetson** | o que realmente roda rápido |

Instalar `rfdetr` + torch no Jetson é possível mas dolorido (torch pro Jetson é
build especial da Nvidia). O caminho certo é exportar o ONNX no Colab, onde tudo
já está instalado, e trazer só ele.

⚠️ **A `.engine` é atada a este aparelho + esta versão de TensorRT.** Não dá pra
gerar no PC e copiar; e se atualizar o JetPack, reconstrua.

## 1. Gerar o ONNX (no Colab, não aqui)

No `model/treino_rfdetr_small_completo.ipynb`, rode a célula **"Export ONNX"**:

```python
!pip -q install onnx onnxruntime
```

Ela pega o campeão (`FT4 → FT3 → FT1`, o primeiro que existir), valida o grafo e
salva `soja_rfdetr_small_CAMPEAO.onnx` no Drive. **Anote o shape de entrada que
ela imprime** — se for fixo (ex. `[1, 3, 512, 512]`), o pré-processamento no
Jetson precisa fazer letterbox pra exatamente esse tamanho.

## 2. Copiar pro Jetson

```bash
# do seu PC, com o Jetson na mesma rede
scp soja_rfdetr_small_CAMPEAO.onnx usuario@IP_DO_JETSON:~/vigilia/
scp bench_trt.sh                   usuario@IP_DO_JETSON:~/vigilia/
```

(ou baixe direto do Drive pelo navegador do Jetson, se tiver desktop)

## 3. Construir a engine e medir

```bash
cd ~/vigilia
chmod +x bench_trt.sh
./bench_trt.sh soja_rfdetr_small_CAMPEAO.onnx fp16
```

O script:
1. Põe o Jetson no modo de **máxima potência** (`nvpmodel` + `jetson_clocks`) —
   sem isso o número sai artificialmente baixo, e é erro comum de benchmark
2. Constrói a engine FP16 (leva minutos: o TensorRT testa kernels pra escolher
   o mais rápido)
3. Mede 100 iterações e imprime throughput + latência

### ✅ Resultado medido (Orin Nano, JetPack L4T r39.2 / CUDA 13.2, FP16)

```
Throughput      : 53,2 qps
GPU Compute Time: mean 18,68 ms  |  median 18,13  |  p99 22,57
engine          : 59 MB
```

**O RF-DETR Small cabe em tempo real no Orin Nano, com folga.** Não precisa de
INT8, nem cair pro Nano, nem DeepStream.

Nota de método: a expectativa prévia era ~35-40 qps, extrapolando dos 151 fps do
**AGX Orin** pela razão de TOPS (~275 contra ~67). O real veio **acima** —
escalar por TOPS subestima, fica o registro pra próximas contas.

### Como ler o `Throughput`

É o que o **modelo** aguenta sozinho. O app real fica abaixo, porque decodificar
vídeo, rastrear e desenhar custam à parte.

| throughput do modelo | leitura |
|---|---|
| **> 40 qps** | folga confortável — app real deve passar de 20 fps ← **estamos aqui (53)** |
| **20-40 qps** | viável; app em ~10-20 fps, suficiente pro veredito travado |
| **10-20 qps** | apertado — tentar INT8 ou cair pro RF-DETR **Nano** (384px) |
| **< 10 qps** | Small não serve; testar Nano ou voltar pro YOLO |

## 4. Se o FP16 não bastar (não foi o caso — só referência)

Em ordem de custo:

1. **RF-DETR Nano** em vez do Small — 384px em vez de 512, ~2,3 ms na T4 contra
   3,5 ms do Small. Precisa re-treinar (mesmo notebook, trocando a classe), mas
   o pipeline de dados já está pronto e validado.
2. **INT8** — `./bench_trt.sh ... int8` dá o teto de velocidade, **mas sem
   calibração a acurácia cai**. Se o ganho compensar, o caminho certo é
   Quantization-Aware Training (TAO Toolkit da Nvidia), que treina o modelo já
   sabendo que vai ser quantizado e perde bem menos.
   ⚠️ **Não assuma que INT8 é mais rápido aqui.** Há relato no fórum da NVIDIA de
   INT8 causando regressão de 2,7× num ViT-S no Orin Nano — e o backbone do
   RF-DETR Large é exatamente um ViT-S (`dinov2_windowed_small`). A quantização
   insere nós de reformat/dequant que podem dominar em transformer. Meça o A/B
   contra o FP16 antes de trocar, e compare acurácia no conjunto anotado.
3. **DeepStream** — só vale quando o gargalo for o pipeline de vídeo (decodificar,
   copiar CPU↔GPU) ou quando houver **várias câmeras**. Pra uma câmera só, é
   complexidade sem retorno. A Roboflow publicou parser pronto pro RF-DETR, então
   quando fizer sentido, a parte difícil já existe.

## 4a. Modo rig: esteira, recortes e laudo

Com a câmara de inspeção (100 mm de faixa, esteira, ver `PADRAO_CAPTURA.md`), o
app roda assim:

```bash
python3 vigil_jetson.py --camera csi --roi 704 --tiles 2 --esteira \
                        --laudo laudo.json
```

- `--tiles 2` divide a faixa em dois recortes 1:1 lado a lado (140 px de
  sobreposição), cobrindo os 100 mm sem reescalar. Grão a ~89 px.
- `--esteira` liga a **compensação de movimento** no rastreamento e trava o
  veredito por contagem de varreduras. Sem ela, o grão troca de ID no meio da
  travessia acima de 53 mm/s — abaixo da velocidade que a meta de vazão exige.
- `--laudo` grava contagem, % premium, massa estimada e kg/h, atualizando a cada
  5 min (`--laudo-seg`) para uma jornada de 14-16 h não perder tudo se cair.

Dimensione antes, e confira depois com a régua:

```bash
python3 calcular_vazao.py --compensado      # distância, velocidade máx, kg/h
python3 calibrar_rig.py                     # px/mm real e exposição
python3 testar_esteira.py                   # roda no PC, sem Jetson nem câmera
```

`testar_esteira.py` exercita justamente o que não depende de GPU — rastreamento
com movimento, junção dos recortes e contabilidade do laudo — e falha se algum
dos limites documentados regredir.

## 4b. Dependências do app ao vivo

O JetPack já traz TensorRT, CUDA e (normalmente) OpenCV. Falta só a ponte de
memória CUDA pro Python:

```bash
sudo apt install -y python3-pip python3-dev
pip3 install cuda-python          # wheel pronto, não compila
```

Se aparecer `externally-managed-environment` (Ubuntu novo), acrescente
`--break-system-packages`. Plano B é o `pycuda`, mas ele **compila** e precisa do
nvcc no PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
pip3 install pycuda --break-system-packages
```

⚠️ Se usar venv, crie com `--system-site-packages` — o `tensorrt` vem do JetPack
(apt), não do pip, e um venv isolado não enxerga ele.

## 5. Depois da medição

Com a engine validada, o app de inferência ao vivo é o `vigil_jetson.py` — o
equivalente do `deck/vigil_deck.py`, mas consumindo a `.engine` via TensorRT em
vez do `.pt` via ultralytics, mantendo a mesma regra de **voto exigente por
classe** e **veredito travado por grão**. Para o rig com esteira, ver §4a.

---

## Comportamento esperado: multi-grão bom, grão solto ruim

**É de propósito, não é defeito.** O FT4 foi treinado assim (números do build):

```
fotos soltas : 1892 imagens × 1 caixa  =  1.892 caixas  (9%)
cenas        : 1200 imagens × 6-25     = 18.477 caixas  (91%)
```

**91% do treino é cena densa.** Junte a isso o fato de a família DETR aprender um
prior de *quantos objetos existem no quadro* (foi o mesmo mecanismo que quebrou o
FT2, ver `model/COMPARATIVO_YOLO11S_VS_RTDETR.md` §11) e o resultado é o
esperado: um grão sozinho, preenchendo o quadro, está fora da distribuição de
treino — nas cenas o grão ocupa 60-150 px num canvas de 640.

Isso está alinhado com o uso real: inspeção de lote é multi-grão por definição.
**Se algum dia precisar de estação de grão único** (conferência final, por
exemplo), não "conserte" este modelo — ou treine um com mistura diferente, ou use
o classificador do modo foto (YOLO11s-cls), que é feito pra isso.

## ⚠️ Antes de julgar acurácia: padronize a captura

O modelo foi treinado com fotos do **celular**. Se o rig do Jetson tem outra
câmera, outra luz, outro fundo ou outra distância, você está medindo **noutro
domínio** — e o número não se transfere. Domain shift é o gargalo estrutural
deste projeto desde a era EfficientNet (29% → 64% → 91,7% só com fine-tune no
domínio certo).

Enquanto não houver padrão de captura, meça só o que independe de domínio: fps,
latência, estabilidade de caixa e de tracking, integração. **Acurácia, recall por
classe e calibragem de `RATIOS`/`conf` ficam para depois.**

Pelo mesmo motivo, **não colete os grãos de `immature`/`spotted` ainda** — dado
capturado antes da padronização nasce num domínio que vai ser descartado.

O padrão precisa fixar: distância, fundo, enquadramento, câmera e **iluminação
travada** — sem auto-exposição nem auto-white-balance mudando entre sessões, que
é o jeito mais fácil de introduzir domain shift sem perceber.

## Limitação conhecida do modelo atual

O FT4 tem `broken` (AP 0,86), `intact` (0,81) e `skin-damaged` (0,67) sólidos,
mas **`immature` está praticamente inoperante** (recall 0,041) e `spotted` é
fraco (AP 0,30) — falta grão único desses dois tipos no dataset (56 e 35, contra
122 das classes fortes). O vídeo de teste não tem grão imaturo, então isso não
aparece nele. **Um lote com grão imaturo passaria batido por este modelo.**
Detalhes em `model/COMPARATIVO_YOLO11S_VS_RTDETR.md` §11.
