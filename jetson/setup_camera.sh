#!/usr/bin/env bash
# Vígil.ia — põe a câmera CSI (IMX219) para funcionar no Orin Nano.
#
#     ./setup_camera.sh            # diagnostica e testa
#     ./setup_camera.sh --foto     # salva uma foto de teste
#     ./setup_camera.sh --ver      # abre a visualização ao vivo
#
# Diagnostica de baixo para cima — cabo, kernel, driver, GStreamer, OpenCV —
# porque quase todo problema de CSI é elétrico ou de device tree, e olhar o
# código nessas horas é procurar no lugar errado.
set -uo pipefail

VERDE=$'\033[32m'; VERM=$'\033[31m'; AMAR=$'\033[33m'; FIM=$'\033[0m'
ok()   { echo "${VERDE}OK${FIM}   $*"; }
aviso(){ echo "${AMAR}AVISO${FIM} $*"; }
erro() { echo "${VERM}ERRO${FIM} $*"; }
titulo(){ echo; echo "=============================================================="; \
          echo " $*"; echo "=============================================================="; }

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
ACAO="${1:-diag}"
SENSOR=0

titulo "1. O kernel enxerga a câmera?"

if [ "$(uname -m)" != "aarch64" ]; then
    erro "isto roda no Jetson (aarch64), e aqui é $(uname -m)"; exit 1
fi

# /dev/videoN é o nó que o driver cria. Sem ele, nada adiante funciona — e a
# causa é quase sempre física: cabo invertido, mal encaixado, ou a placa ligada
# na hora de plugar (o que pode danificar o sensor).
if ls /dev/video* >/dev/null 2>&1; then
    ok "nós de vídeo: $(ls /dev/video* | tr '\n' ' ')"
else
    erro "nenhum /dev/video* — o driver não subiu."
    echo
    # Distinção que economiza muito tempo: o LED do módulo indica que o trilho
    # de 3,3 V chegou nele. Com LED aceso, o cabo está encaixado e na orientação
    # certa — o problema deixa de ser físico e passa a ser device tree, que no
    # Orin Nano Dev Kit sai de fábrica SEM câmera nenhuma habilitada.
    echo "  >>> O LED do módulo da câmera está ACESO?"
    echo
    echo "      SE ESTÁ ACESO: o cabo está bom (o módulo tem energia). O caso"
    echo "      quase certo é o DEVICE TREE — o Orin Nano Dev Kit vem sem"
    echo "      nenhuma câmera habilitada. Habilite o IMX219:"
    echo
    echo "          sudo /opt/nvidia/jetson-io/jetson-io.py"
    echo "          Configure Jetson 24pin CSI Connector"
    echo "            -> Configure for compatible hardware"
    echo "              -> Camera IMX219 Dual"
    echo "                -> Save pin changes -> Save and reboot"
    echo
    echo "      Confirme antes que o kernel realmente não tentou carregar:"
    echo "          sudo dmesg | grep -i -E 'imx219|nvcsi|vi5'"
    echo "          grep -i overlays /boot/extlinux/extlinux.conf"
    echo
    echo "      SE ESTÁ APAGADO: é físico. Confira NESTA ordem:"
    echo "   1. A placa estava DESLIGADA quando você plugou o cabo?"
    echo "      Plugar com a Jetson ligada pode queimar o sensor."
    echo "      Desligue, replugue, ligue de novo."
    echo "   2. Orientação do cabo: os CONTATOS METÁLICOS do flat ficam"
    echo "      virados para o lado do dissipador (para dentro da placa)."
    echo "      Do lado da câmera, os contatos ficam para BAIXO (lado da PCB)."
    echo "   3. Trava do conector puxada para trás antes de inserir, e"
    echo "      empurrada de volta depois. O flat tem que entrar reto e até o fim."
    echo "   4. Conector certo: o Orin Nano usa CSI de 22 pinos (passo 0,5 mm)."
    echo "      Cabo de 15 pinos é do Jetson Nano ANTIGO e não serve."
    echo "   5. Tente o outro conector CAM (CAM0 / CAM1)."
    echo
    echo "  Depois de replugar, veja o que o kernel diz:"
    echo "    sudo dmesg | grep -i -E 'imx219|tegra-cam|camera'"
    exit 1
fi

echo
echo "--- o que o kernel registrou sobre o sensor ---"
if sudo dmesg 2>/dev/null | grep -i -E 'imx219' | tail -5 | grep -q .; then
    sudo dmesg | grep -i -E 'imx219' | tail -5 | sed 's/^/    /'
    ok "IMX219 reconhecido pelo driver"
else
    aviso "o kernel não menciona 'imx219'."
    sudo dmesg 2>/dev/null | grep -i -E 'tegra-camrtc|camera|sensor' | tail -5 | sed 's/^/    /'
    echo
    echo "  Se o /dev/video existe mas o sensor não aparece, o DEVICE TREE pode"
    echo "  estar apontando para outra câmera (o Orin Nano vem configurável)."
    echo "  Selecione o IMX219 com:"
    echo "    sudo /opt/nvidia/jetson-io/jetson-io.py"
    echo "    -> Configure Jetson 24pin CSI Connector -> Camera IMX219"
    echo "    -> Save and reboot"
fi

titulo "2. Modos que o sensor oferece"

# nvarguscamerasrc lista os modos suportados ao subir. É a fonte da verdade
# sobre resolução e fps — spec de anúncio costuma estar errada.
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
    ok "GStreamer presente"
else
    erro "gst-inspect-1.0 ausente: sudo apt install -y gstreamer1.0-tools"
fi

MODOS="$(timeout 15 gst-launch-1.0 nvarguscamerasrc sensor-id=$SENSOR num-buffers=1 \
         ! fakesink 2>&1 | grep -E 'GST_ARGUS: [0-9]+ x [0-9]+' || true)"
if [ -n "$MODOS" ]; then
    echo "$MODOS" | sed 's/^/    /'
    ok "sensor respondeu"
else
    erro "nvarguscamerasrc não conseguiu abrir o sensor."
    echo "  O serviço da câmera está rodando?"
    echo "    sudo systemctl restart nvargus-daemon && sleep 2"
    echo "  Depois rode este script de novo."
fi

titulo "3. Capturando um quadro de verdade"

# O teste que importa: um quadro chegando no OpenCV, pelo MESMO pipeline que o
# app usa. Passar aqui e falhar no app seria contradição.
python3 - "$ACAO" <<'PY'
import sys
sys.path.insert(0, '.')      # o script já fez cd para a pasta dele
import cv2
from vigil_jetson import abrir_camera

acao = sys.argv[1]
# sem trava de exposição: aqui o objetivo é VER, não padronizar
cap = abrir_camera('csi', travar_csi=False)
ok, frame = None, None
for _ in range(20):
    ok, frame = cap.read()
    if ok and frame is not None:
        break
if not ok or frame is None:
    cap.release()
    sys.exit('nenhum quadro chegou ao OpenCV pelo pipeline da CSI')

h, w = frame.shape[:2]
print(f'    quadro: {w}x{h}')
cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print(f'    brilho médio: {cinza.mean():.0f}/255  '
      f'(min {cinza.min()}, max {cinza.max()})')
if cinza.mean() < 5:
    print('    >>> QUASE PRETO: tampa da lente? ring light desligado?')
elif cinza.max() < 40:
    print('    >>> muito escuro: aumente a luz ou a exposição')
elif (cinza >= 253).mean() > 0.05:
    print('    >>> ESTOURADO: reduza a luz — pixel saturado é textura perdida')
else:
    print('    >>> exposição plausível')

cv2.imwrite('camera_teste.jpg', frame)
print('    salvo: camera_teste.jpg')

if acao == '--ver':
    print('\n    janela aberta — q para sair')
    cv2.namedWindow('CSI (q sai)', cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow('CSI (q sai)', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cv2.destroyAllWindows()
cap.release()
PY
CODIGO=$?

titulo "4. Próximo passo"

if [ $CODIGO -ne 0 ]; then
    erro "a captura falhou — resolva os itens acima antes de seguir"
    exit 1
fi
ok "câmera CSI funcionando"

cat <<'FIM_AJUDA'

Ver a imagem ao vivo (para enquadrar e focar):
    ./setup_camera.sh --ver

Ajustar FOCO e EXPOSIÇÃO com a régua no fundo — é o que define px/mm:
    python3 calibrar_rig.py --sem-trava
    # tecla ESPAÇO mede com a régua · 'e' mostra a exposição · 's' salva

    O alvo do rig: 12,7 px/mm (câmera a ~9,3 cm). Se a régua der outro
    valor, ajuste a distância ANTES de fixar o suporte — depois disso a
    geometria congela.

Depois de calibrado, TRAVE a exposição em CSI_TRAVAS (vigil_jetson.py) e
anote os valores na ficha do PADRAO_CAPTURA.md §6. Em automático, a câmera
compensa sozinha entre sessões e recria o domain shift que o rig existe
para eliminar — nada falha, o dado só fica inconsistente.

Inspeção ao vivo pela CSI:
    python3 vigil_jetson.py --engine soja_rfdetr_small_CAMPEAO_fp16.engine \
        --camera csi --quadrado --conf 0.10

FIM_AJUDA
