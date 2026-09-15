#!/usr/bin/env bash
# Vígil.ia — constrói a engine TensorRT do RF-DETR Small e mede no Jetson.
#
# Uso:  ./bench_trt.sh [caminho.onnx] [fp16|int8]
# Ex.:  ./bench_trt.sh soja_rfdetr_small_CAMPEAO.onnx fp16
#
# O .engine é ATADO a este aparelho + esta versão de TensorRT — por isso ele é
# gerado aqui, e não no Colab. Guarde-o: a construção leva minutos (o TensorRT
# testa vários kernels pra escolher o mais rápido), a carga leva segundos.
set -euo pipefail

ONNX="${1:-soja_rfdetr_small_CAMPEAO.onnx}"
PREC="${2:-fp16}"
ENGINE="${ONNX%.onnx}_${PREC}.engine"

# ---- estamos mesmo no Jetson? (erro nº1: rodar o script no PC) -------------
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    echo "ERRO: este script roda DENTRO do Jetson, e aqui a arquitetura é '$ARCH'."
    echo "      (Jetson = aarch64. x86_64 = seu PC.)"
    echo
    echo "      Copie os arquivos pro Jetson e rode lá:"
    echo "        scp $ONNX bench_trt.sh usuario@IP_DO_JETSON:~/"
    echo "        ssh usuario@IP_DO_JETSON"
    echo "        chmod +x bench_trt.sh && ./bench_trt.sh $ONNX $PREC"
    exit 1
fi

[ -f "$ONNX" ] || { echo "ERRO: não achei '$ONNX'."; \
  echo "      Exporte no Colab (célula 'Export ONNX') e copie pra cá."; exit 1; }

# ---- acha o trtexec (o caminho muda entre versões de JetPack) --------------
TRTEXEC=""
for c in "$(command -v trtexec 2>/dev/null || true)" \
         /usr/src/tensorrt/bin/trtexec \
         /usr/local/tensorrt/bin/trtexec \
         /opt/nvidia/tensorrt/bin/trtexec; do
    [ -n "$c" ] && [ -x "$c" ] && { TRTEXEC="$c"; break; }
done
if [ -z "$TRTEXEC" ]; then
    echo "procurando trtexec no sistema…"
    TRTEXEC="$(find /usr /opt -name trtexec -type f -executable 2>/dev/null | head -1 || true)"
fi
if [ -z "$TRTEXEC" ]; then
    echo "ERRO: não achei o trtexec neste Jetson."
    echo
    echo "  TensorRT instalado?"
    dpkg -l 2>/dev/null | grep -i tensorrt | head -5 || echo "    (nenhum pacote tensorrt)"
    echo
    echo "  Instale o JetPack completo:"
    echo "    sudo apt update && sudo apt install nvidia-jetpack"
    echo "  (baixa alguns GB; TensorRT vem junto)"
    exit 1
fi
echo "trtexec: $TRTEXEC"

echo "=============================================="
echo " 1. Energia"
echo "=============================================="
MODO_ATUAL="$(sudo nvpmodel -q 2>/dev/null | grep -i 'power mode' | head -1 || echo '?')"
echo "modo atual: $MODO_ATUAL"

# Trocar pra MAXN às vezes EXIGE reboot, e o nvpmodel abre um prompt YES/no que
# travaria o script. Respondemos "no" e seguimos: medir num modo consistente
# vale mais que medir no modo mais rápido — comparar dois modelos em modos
# diferentes não compara nada.
if [ "${MAXN:-0}" = "1" ]; then
    echo "MAXN=1 -> tentando modo 0 (pode pedir reboot)"
    sudo nvpmodel -m 0 || true
else
    echo 'mantendo o modo atual (rode com MAXN=1 ./bench_trt.sh … pra tentar o modo 0)'
    echo '  → se trocar de modo, RE-MEÇA todos os modelos: número de 15W não'
    echo '    é comparável com número de MAXN.'
fi
sudo jetson_clocks 2>/dev/null || echo '(jetson_clocks falhou; seguindo assim)'

echo
echo "=============================================="
echo " 2. Construindo engine ($PREC) — leva alguns minutos"
echo "=============================================="
FLAGS=(--onnx="$ONNX" --saveEngine="$ENGINE" --memPoolSize=workspace:2048)
[ "$PREC" = "fp16" ] && FLAGS+=(--fp16)
[ "$PREC" = "int8" ] && FLAGS+=(--int8 --fp16)   # int8 sem calibração = queda de
                                                 # acurácia; use só p/ ver o teto
"$TRTEXEC" "${FLAGS[@]}" 2>&1 | tee build_"$PREC".log | \
  grep -E "Engine built|error|Error|ERROR|WARNING: .*fall" || true

[ -f "$ENGINE" ] || { echo "ERRO: engine não foi gerada. Veja build_$PREC.log"; exit 1; }

echo
echo "=============================================="
echo " 3. Medindo (100 iterações)"
echo "=============================================="
"$TRTEXEC" --loadEngine="$ENGINE" --iterations=100 --warmUp=500 --avgRuns=100 \
  2>&1 | tee bench_"$PREC".log | grep -E "Throughput|Latency: min|mean =|median =|GPU Compute" || true

echo
echo "=============================================="
echo " Resumo"
echo "=============================================="
echo "engine : $ENGINE ($(du -h "$ENGINE" | cut -f1))"
grep -E "^\[.*\] \[I\] Throughput" bench_"$PREC".log | tail -1 || true
grep -E "GPU Compute Time: .*mean" bench_"$PREC".log | tail -1 || true
echo
echo "Como ler: 'Throughput' em qps = quadros/segundo que o MODELO aguenta."
echo "O app real fica abaixo disso (decodificar vídeo, rastrear e desenhar custam"
echo "à parte). Pro veredito travado por grão, 15-30 fps já é confortável."
