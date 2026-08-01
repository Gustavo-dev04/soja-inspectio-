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
TRTEXEC=/usr/src/tensorrt/bin/trtexec
ENGINE="${ONNX%.onnx}_${PREC}.engine"

[ -f "$ONNX" ] || { echo "ERRO: não achei '$ONNX'."; \
  echo "      Exporte no Colab (célula 'Export ONNX') e copie pra cá."; exit 1; }
[ -x "$TRTEXEC" ] || { echo "ERRO: trtexec não está em $TRTEXEC."; \
  echo "      Confira se o JetPack está instalado: 'dpkg -l | grep tensorrt'"; exit 1; }

echo "=============================================="
echo " 1. Energia no máximo (senão o número sai menor)"
echo "=============================================="
echo "modos disponíveis:"
sudo nvpmodel -q --verbose 2>/dev/null | head -20 || nvpmodel -q || true
echo
# No Orin Nano o modo de maior potência costuma ser o 0 (MAXN). Se o seu for
# outro, rode antes:  sudo nvpmodel -m <N>
sudo nvpmodel -m 0 2>/dev/null || echo "(nvpmodel -m 0 falhou; seguindo no modo atual)"
sudo jetson_clocks 2>/dev/null || echo "(jetson_clocks falhou; seguindo assim)"
echo "modo ativo: $(sudo nvpmodel -q 2>/dev/null | tail -1 || echo '?')"

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
