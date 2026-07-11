# Vígil.ia no Steam Deck — inspeção de soja ao vivo, 100% local

Roda o modelo campeão **direto no Steam Deck**, sem internet, sem site, sem servidor.
Abre a câmera, detecta e classifica cada grão em tempo real com voto exigente por classe.

Depois, o **celular via USB** vira só "outra câmera" — mesmo script, `--camera <n>`.

---

## 1. Pré-requisitos (uma vez)

No Deck, entre no **Modo Desktop** (botão STEAM → Ligar/desligar → Alternar para área de
trabalho) e abra o **Konsole** (terminal).

O sistema do Deck é somente-leitura, mas a sua pasta pessoal (`~`) é gravável — por isso
usamos um ambiente virtual no home:

```bash
# 1) ambiente isolado no home (sobrevive a updates do SteamOS)
python3 -m venv ~/vigilia
source ~/vigilia/bin/activate

# 2) TORCH CPU primeiro (o Deck é AMD; a build CUDA é ~2 GB e não roda aqui)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3) o resto
pip install ultralytics==8.4.80 opencv-python
```

> Se o `pip` reclamar de permissão, confirme que o venv está ativo (o prompt mostra
> `(vigilia)`). Nunca use `sudo pip` no SteamOS.

## 2. Pegar o modelo e o script

```bash
mkdir -p ~/vigilia-app && cd ~/vigilia-app
# baixe estes 2 arquivos para cá:
#   - vigil_deck.py            (este repositório, pasta deck/)
#   - soja_yolo11n_base12k_v2.pt   (seu campeão, do Google Drive)
```

O `.pt` você baixa do Drive pelo navegador (Firefox no Modo Desktop). Deixe o `.pt` na
**mesma pasta** do `vigil_deck.py` (ou passe o caminho com `--model`).

## 3. Rodar

```bash
source ~/vigilia/bin/activate
cd ~/vigilia-app
python vigil_deck.py                     # câmera padrão (0)
```

Janela abre com a câmera; cada grão ganha caixa + rótulo; o HUD conta intactos/defeitos.
Teclas: **q** sai (salva) · **espaço** salva agora (só com `--save-dir`) · **c** zera a contagem · **p** pausa.

Ajustes úteis:

```bash
python vigil_deck.py --imgsz 480         # mais FPS (menos resolução)
python vigil_deck.py --conf 0.4          # menos detecção fraca
python vigil_deck.py --hold 5            # observa 5s cada grão antes de fechar a classe
python vigil_deck.py --model /caminho/outro.pt
```

> **Segure a câmera parada sobre os grãos.** O grão só fecha a classe depois de
> `--hold` segundos observando (padrão **3s**) — filmar rápido demais fecha veredito
> errado. As caixas são suavizadas (não tremem) e detecções piscantes de 1-2 frames
> são ignoradas, pra caixa não ficar bagunçada. Errar a *classe* tudo bem; o que
> evitamos é caixa mal posicionada.

## 4. Câmera do celular

> O Steam Deck **não tem câmera embutida** — o celular (ou uma webcam USB) é a câmera.

### 4a. Via USB (Android 14+ com modo Webcam)

1. Plugue o cabo (tem que ser cabo de **dados**, não só carga) e, no celular,
   toque na notificação USB → **Preferências USB** → escolha **"Webcam"** (modo UVC nativo).
   Nem todo Android 14 tem essa opção — fabricantes de entrada (ex. XOS/Infinix)
   às vezes não incluem; nesse caso use a via Wi-Fi (4b).
2. No Deck, com o cabo conectado:
   ```bash
   python vigil_deck.py --list-cameras    # mostra os índices disponíveis
   python vigil_deck.py --camera 2        # o índice do celular (geralmente o maior)
   python vigil_deck.py --camera 2 --rotate 90   # se o celular estiver em pé
   ```

### 4b. Via Wi-Fi (qualquer Android — DroidCam)

Não instala nada a mais no Deck: o celular transmite o vídeo pela rede e o
script abre a URL direto.

1. Instale o app **DroidCam** (Dev47Apps) no celular, pela Play Store.
2. Celular e Deck na **mesma rede Wi-Fi**. Abra o app — ele mostra algo como
   `WiFi IP: 192.168.0.15` e `Port: 4747`.
3. No Deck:
   ```bash
   python vigil_deck.py --camera http://192.168.0.15:4747/video   # use o IP do SEU app
   ```
   Se a imagem vier deitada, acrescente `--rotate 90`.

A câmera do celular costuma ter lente melhor que webcams baratas — bom pra qualidade de imagem.

## 5. Capturar grãos p/ treino futuro

O app pode salvar, de cada grão com veredito fechado, o recorte mais nítido visto
durante o rastreio — pra depois revisar e reaproveitar num re-treino:

```bash
python vigil_deck.py --save-dir capturas --camera http://192.168.0.15:4747/video
```

Cria `capturas/sessao_AAAAMMDD_HHMMSS/` com:
- `graos/0001_intact.jpg`, `graos/0002_broken.jpg`, … — um recorte por grão
- `revisao.csv` — colunas `id, classe_prevista, confianca, n_frames, classe_corrigida`

Aperte **espaço** a qualquer momento pra gravar o que já foi coletado (o terminal
mostra `💾 N grão(s) salvo(s)`); ao sair com **q** ele também grava. Sem `--save-dir`,
o espaço só avisa que não há onde salvar. **c** zera a contagem (não confundir com salvar).

Mesmo formato do `model/aprendizado_ativo.ipynb` (Fase 2): abra o CSV, preencha
`classe_corrigida` só onde o modelo errou (vazio = confirmado certo,
`descartar` = não é grão), e os recortes + o CSV corrigido alimentam a Fase 3
do notebook (propaga a correção e re-treina). Só grãos com veredito **travado**
são salvos — os que ainda diziam "analisando..." ficam de fora.

---

## Notas

- **Desempenho:** o 11n é minúsculo (2,6 M params); no CPU do Deck deve dar ~10–20 FPS a
  640. Se quiser mais fluidez, `--imgsz 480`.
- **Modo de Jogo (Game Mode):** a janela do OpenCV precisa de desktop — rode no **Modo
  Desktop**. (Dá pra criar um atalho depois.)
- **Voto por classe:** os limiares que matam a alucinação de defeito estão no topo do
  `vigil_deck.py` (`RATIOS`). Sem internet, é tudo local.
- **Iluminação:** o modelo foi treinado com fundo escuro e luz de cima; evite ponto de
  luz estourado (reflexo) no quadro — atrapalha tanto quanto no treino.
