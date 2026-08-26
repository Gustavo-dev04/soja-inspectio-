#!/usr/bin/env bash
# Vígil.ia — configura o Jetson do ZERO, num cartão SD recém-gravado.
#
#     chmod +x setup_jetson.sh && ./setup_jetson.sh
#
# Idempotente: pode rodar de novo sem estragar nada. Ele checa o ambiente,
# instala o que falta, constrói as engines TensorRT dos .onnx que estiver na
# pasta, e verifica se o app sobe.
#
# Por que existe: cada item aqui já quebrou pelo menos uma vez neste projeto —
# pip3 que não vem instalado, "externally-managed-environment", cuda-python que
# mudou de módulo entre a 12.x e a 13.x, venv que não enxerga o tensorrt do
# JetPack. O script resolve os quatro sem você ter que lembrar.
set -uo pipefail

VERDE=$'\033[32m'; VERM=$'\033[31m'; AMAR=$'\033[33m'; FIM=$'\033[0m'
ok()   { echo "${VERDE}OK${FIM}   $*"; }
aviso(){ echo "${AMAR}AVISO${FIM} $*"; }
erro() { echo "${VERM}ERRO${FIM} $*"; }
titulo(){ echo; echo "=================================================================="; \
          echo " $*"; echo "=================================================================="; }

FALHAS=0
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUI"

# ============================================================= 1. o aparelho
titulo "1. Onde estamos"

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    erro "isto é '$ARCH', não um Jetson (aarch64)."
    echo "     Copie a pasta jetson/ pro aparelho e rode lá:"
    echo "       scp -r jetson/ usuario@IP_DO_JETSON:~/vigilia"
    exit 1
fi
ok "arquitetura aarch64"

if [ -f /etc/nv_tegra_release ]; then
    ok "L4T: $(head -1 /etc/nv_tegra_release)"
else
    aviso "/etc/nv_tegra_release ausente — não parece um JetPack padrão"
fi
[ -f /proc/device-tree/model ] && ok "placa: $(tr -d '\0' < /proc/device-tree/model)"

# Espaço: o TensorRT precisa de folga pra construir a engine, e cartão cheio
# falha no meio da construção, depois de minutos.
LIVRE_KB="$(df --output=avail / | tail -1)"
LIVRE_GB=$((LIVRE_KB / 1024 / 1024))
if [ "$LIVRE_GB" -lt 5 ]; then
    erro "só ${LIVRE_GB} GB livres em / — a construção da engine pode falhar no meio"
    FALHAS=$((FALHAS+1))
else
    ok "espaço livre: ${LIVRE_GB} GB"
fi

# ======================================================== 2. o que o JetPack traz
titulo "2. JetPack — CUDA, TensorRT, trtexec"

if [ -x /usr/local/cuda/bin/nvcc ]; then
    ok "CUDA: $(/usr/local/cuda/bin/nvcc --version | grep -o 'release [0-9.]*' | head -1)"
else
    aviso "nvcc não encontrado em /usr/local/cuda/bin (só importa se for compilar pycuda)"
fi

TRTEXEC=""
for c in "$(command -v trtexec 2>/dev/null || true)" \
         /usr/src/tensorrt/bin/trtexec /usr/local/tensorrt/bin/trtexec \
         /opt/nvidia/tensorrt/bin/trtexec; do
    [ -n "$c" ] && [ -x "$c" ] && { TRTEXEC="$c"; break; }
done
[ -z "$TRTEXEC" ] && TRTEXEC="$(find /usr /opt -name trtexec -type f -executable 2>/dev/null | head -1 || true)"

if [ -n "$TRTEXEC" ]; then
    ok "trtexec: $TRTEXEC"
else
    erro "trtexec não encontrado — o TensorRT não está instalado."
    echo "     sudo apt update && sudo apt install -y nvidia-jetpack"
    echo "     (baixa alguns GB; TensorRT, CUDA e cuDNN vêm juntos)"
    FALHAS=$((FALHAS+1))
fi

# =========================================================== 3. pacotes Python
titulo "3. Pacotes Python"

# pip3 NÃO vem no JetPack por padrão — é o primeiro tropeço numa imagem nova.
if ! command -v pip3 >/dev/null 2>&1; then
    echo "pip3 ausente, instalando…"
    sudo apt update -qq && sudo apt install -y python3-pip python3-dev
fi
command -v pip3 >/dev/null 2>&1 && ok "pip3: $(pip3 --version | cut -d' ' -f2)" \
    || { erro "pip3 continua ausente"; FALHAS=$((FALHAS+1)); }

python3 -c "import cv2" 2>/dev/null || {
    echo "OpenCV ausente, instalando do apt (o do pip não traz o backend GStreamer,"
    echo "  e sem GStreamer a câmera CSI não abre)…"
    sudo apt install -y python3-opencv
}
if python3 -c "import cv2" 2>/dev/null; then
    ok "OpenCV: $(python3 -c 'import cv2; print(cv2.__version__)')"
    if python3 - <<'PY' 2>/dev/null
import re, sys, cv2
m = re.search(r'GStreamer:\s*(\S+)', cv2.getBuildInformation())
sys.exit(0 if m and m.group(1).upper().startswith('YES') else 1)
PY
    then
        ok "OpenCV com GStreamer (câmera CSI vai abrir)"
    else
        aviso "OpenCV SEM GStreamer — a câmera CSI não vai abrir."
        echo "      O OpenCV do pip não traz GStreamer; o do apt traz:"
        echo "        pip3 uninstall -y opencv-python opencv-python-headless"
        echo "        sudo apt install -y python3-opencv"
        echo "      (sem CSI ainda dá pra usar --camera 0 ou o celular por URL)"
    fi
else
    erro "OpenCV não importa"; FALHAS=$((FALHAS+1))
fi

# tensorrt vem do APT (JetPack), não do pip. Num venv sem --system-site-packages
# ele fica invisível, e o app morre no import.
if python3 -c "import tensorrt" 2>/dev/null; then
    ok "tensorrt (python): $(python3 -c 'import tensorrt; print(tensorrt.__version__)')"
else
    erro "'import tensorrt' falhou."
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "     Você está num venv ($VIRTUAL_ENV). O tensorrt vem do apt, não do pip:"
        echo "       deactivate && rm -rf '$VIRTUAL_ENV'"
        echo "       python3 -m venv --system-site-packages '$VIRTUAL_ENV'"
    else
        echo "     sudo apt install -y nvidia-jetpack"
    fi
    FALHAS=$((FALHAS+1))
fi

# Ponte de memória CUDA pro Python. cuda-python é wheel pronto; pycuda compila.
if ! python3 - <<'PY' 2>/dev/null
import importlib
for m in ('cuda.bindings.runtime', 'cuda.cudart', 'cuda.runtime'):
    try:
        importlib.import_module(m); raise SystemExit(0)
    except ImportError:
        pass
try:
    import pycuda.driver; raise SystemExit(0)
except ImportError:
    raise SystemExit(1)
PY
then
    echo "sem cuda-python nem pycuda, instalando cuda-python…"
    pip3 install cuda-python 2>/dev/null \
      || pip3 install cuda-python --break-system-packages   # Ubuntu 24.04+
fi

CUDA_MOD="$(python3 - <<'PY' 2>/dev/null
import importlib
for m in ('cuda.bindings.runtime', 'cuda.cudart', 'cuda.runtime'):
    try:
        importlib.import_module(m); print(m); break
    except ImportError:
        pass
else:
    try:
        import pycuda.driver; print('pycuda')
    except ImportError:
        pass
PY
)"
if [ -n "$CUDA_MOD" ]; then
    ok "runtime CUDA no Python: $CUDA_MOD"
else
    erro "nem cuda-python nem pycuda importam."
    echo "     pip3 install cuda-python --break-system-packages"
    echo "     plano B (COMPILA, precisa do nvcc no PATH):"
    echo "       export PATH=/usr/local/cuda/bin:\$PATH"
    echo "       pip3 install pycuda --break-system-packages"
    FALHAS=$((FALHAS+1))
fi

# ============================================================ 4. as engines
titulo "4. Engines TensorRT"

# A .engine é ATADA a este aparelho + esta versão de TensorRT. Trocar o cartão
# SD troca a versão do TensorRT: engines antigas não servem mais, e é por isso
# que elas são reconstruídas aqui em vez de copiadas.
shopt -s nullglob
ONNXS=(*.onnx)
shopt -u nullglob

if [ ${#ONNXS[@]} -eq 0 ]; then
    aviso "nenhum .onnx nesta pasta ($AQUI)."
    echo "      Copie os modelos exportados pelo Colab pra cá e rode de novo:"
    echo "        scp soja_rfdetr_*.onnx usuario@IP_DO_JETSON:~/vigilia/"
else
    echo "encontrados: ${ONNXS[*]}"
    [ -z "$TRTEXEC" ] && { erro "sem trtexec, não dá pra construir"; FALHAS=$((FALHAS+1)); }
    for onnx in "${ONNXS[@]}"; do
        engine="${onnx%.onnx}_fp16.engine"
        if [ -f "$engine" ]; then
            ok "$engine já existe ($(du -h "$engine" | cut -f1)) — pulando"
            continue
        fi
        [ -z "$TRTEXEC" ] && continue
        echo
        echo "--- construindo $engine (leva alguns minutos: o TensorRT testa"
        echo "    vários kernels pra escolher o mais rápido) ---"
        if "$TRTEXEC" --onnx="$onnx" --saveEngine="$engine" --fp16 \
               --memPoolSize=workspace:2048 > "build_${onnx%.onnx}.log" 2>&1; then
            ok "$engine ($(du -h "$engine" | cut -f1))"
        else
            erro "falhou. Últimas linhas de build_${onnx%.onnx}.log:"
            tail -15 "build_${onnx%.onnx}.log" | sed 's/^/       /'
            FALHAS=$((FALHAS+1))
        fi
    done
fi

# ========================================================= 5. verificação
titulo "5. Verificação"

shopt -s nullglob
ENGINES=(*.engine)
shopt -u nullglob

if [ ${#ENGINES[@]} -eq 0 ]; then
    aviso "nenhuma engine para verificar"
else
    for e in "${ENGINES[@]}"; do
        echo
        echo "--- $e ---"
        python3 inspect_engine.py "$e" 2>&1 | sed 's/^/    /' || {
            erro "inspect_engine.py falhou em $e"; FALHAS=$((FALHAS+1)); }
    done
fi

echo
if python3 -c "
import sys, types
sys.argv = ['x', '--help']
exec(open('vigil_jetson.py').read().split('def main()')[0])
print('imports do vigil_jetson.py OK')
" 2>/dev/null; then
    ok "vigil_jetson.py carrega (imports e classes)"
else
    aviso "não consegui validar o vigil_jetson.py isoladamente — teste rodando:"
    echo "      python3 vigil_jetson.py --help"
fi

# ============================================================== 6. resumo
titulo "6. Resumo"

if [ "$FALHAS" -eq 0 ]; then
    echo "${VERDE}Tudo pronto.${FIM}"
else
    echo "${VERM}$FALHAS problema(s) acima.${FIM} Resolva antes de seguir."
fi

echo
echo "Próximo passo — CONFIRA O MAPEAMENTO DE CLASSE antes de confiar no"
echo "resultado. O off-by-one (coluna extra do 'no-object' no fim ou na frente)"
echo "já apareceu 3 vezes neste projeto, e ele não dá erro: só faz tudo virar"
echo "a mesma classe."
echo
if [ ${#ENGINES[@]} -gt 0 ]; then
    E0="${ENGINES[0]}"
else
    E0="SUA_ENGINE.engine"
fi

# O --roi tem que casar com a ENTRADA da engine: recortar 704 px e entregar a um
# modelo de 384 faria o app reduzir o recorte, que é exatamente o que o ROI 1:1
# existe pra evitar. Em vez de reimplementar a leitura (o inspect_engine.py já
# cobre TensorRT 8.x e 10.x), lê a saída dele.
ENTRADA="$(python3 inspect_engine.py "$E0" 2>/dev/null \
           | sed -n 's/.*ENTRADA.*shape=(\([0-9, ]*\)).*/\1/p' \
           | head -1 | tr -d ' ' | awk -F, '{print $NF}')"
# A geometria do rig (9,3 cm, 12,7 px/mm, 2 recortes) foi dimensionada para
# entrada 704. Com outra entrada, a distância da câmera muda — e usar a errada
# entrega o grão numa escala diferente da que o modelo viu no treino.
AVISO_ROI=""
if [ -n "$ENTRADA" ] && [ "$ENTRADA" != "704" ]; then
    AVISO_ROI="
    ATENCAO: esta engine tem entrada ${ENTRADA}, nao 704. A geometria do rig
    (camera a 9,3 cm) foi dimensionada para 704. Recalcule a distancia:
        python3 calcular_vazao.py --compensado   # e edite ENTRADA_MODELO
    Sem isso o grao chega ao modelo numa escala diferente da do treino."
fi

cat <<FIM_AJUDA
    python3 vigil_jetson.py --engine $E0 --source video.mp4 --diag 60
    # a coluna com mais detecções tem que ser 'intact'. Se não for:
    #   --class-offset 0  -> coluna extra no FIM
    #   --class-offset 1  -> coluna extra na FRENTE

Depois, o rig completo (câmara de 100 mm, esteira — ver PADRAO_CAPTURA.md):

    python3 vigil_jetson.py --engine $E0 \\
        --camera csi --roi ${ENTRADA:-<entrada da engine>} --tiles 2 --esteira --laudo laudo.json
${AVISO_ROI}
Sem a câmera CSI ainda, dá pra testar tudo com o celular ou um arquivo:

    python3 vigil_jetson.py --engine $E0 --camera http://IP_DO_CELULAR:4747/video
    python3 vigil_jetson.py --engine $E0 --source teste_soja.mp4 --out saida.mp4

Medir a velocidade real do modelo:      ./bench_trt.sh <arquivo.onnx> fp16
Calibrar a óptica do rig (com régua):   python3 calibrar_rig.py
Dimensionar a vazão da câmara:          python3 calcular_vazao.py --compensado
FIM_AJUDA

exit $((FALHAS > 0 ? 1 : 0))
