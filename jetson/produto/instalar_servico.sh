#!/usr/bin/env bash
# Instala o vigild como serviço do sistema: sobe no boot e reinicia se cair.
#   sudo ./produto/instalar_servico.sh
#   sudo ./produto/instalar_servico.sh --remover
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "rode com sudo"; exit 1; }
AQUI="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
JETSON="$(dirname "$AQUI")"
USUARIO="${SUDO_USER:-$(stat -c %U "$JETSON")}"
DESTINO=/etc/systemd/system/vigild.service

if [ "${1:-}" = "--remover" ]; then
    systemctl disable --now vigild 2>/dev/null || true
    rm -f "$DESTINO"; systemctl daemon-reload
    echo "vigild removido."; exit 0
fi

[ -f "$AQUI/aparelho.json" ] || { echo "falta $AQUI/aparelho.json"; exit 1; }
ENGINE="$(python3 -c "import json,os,sys; c=json.load(open('$AQUI/aparelho.json')); e=c.get('engine',''); print(e if os.path.isabs(e) else os.path.normpath(os.path.join('$AQUI', e)))")"
if [ ! -f "$ENGINE" ]; then
    echo "AVISO: engine não encontrada: $ENGINE"
    echo "       ajuste \"engine\" em $AQUI/aparelho.json antes de usar."
fi
# a câmera CSI e a GPU exigem os grupos video/render
usermod -aG video,render "$USUARIO" 2>/dev/null || true

sed -e "s#__USUARIO__#$USUARIO#g" -e "s#__JETSON__#$JETSON#g" \
    "$AQUI/vigild.service" > "$DESTINO"
systemctl daemon-reload
systemctl enable --now vigild
sleep 2
systemctl --no-pager --lines=12 status vigild || true
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "Pronto. Abra no celular:  http://${IP:-IP_DO_JETSON}:$(python3 -c "import json; print(json.load(open('$AQUI/aparelho.json')).get('porta',8080))")"
echo "Logs ao vivo:            journalctl -u vigild -f"
