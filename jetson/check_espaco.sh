#!/usr/bin/env bash
# Vígil.ia — diagnóstico de armazenamento no Jetson.
#
# Uso:  ./check_espaco.sh
#
# Mostra quanto espaço sobra, o que está ocupando, e o que dá pra limpar sem
# quebrar nada. O JetPack completo pede ~12-15 GB durante a instalação (baixa
# os .deb e depois descompacta), então é comum apertar em cartão microSD.
set -uo pipefail

human () { numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1"; }

echo "════════════════════════════════════════════════════"
echo " 1. PARTIÇÕES"
echo "════════════════════════════════════════════════════"
df -h -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null || df -h

# ---- raiz: quanto sobra, e em que tipo de mídia ----
ROOT_DEV="$(findmnt -no SOURCE / 2>/dev/null || echo '?')"
case "$ROOT_DEV" in
    *nvme*)   MIDIA="NVMe (SSD) — rápido" ;;
    *mmcblk*) MIDIA="microSD / eMMC — mais lento, e cartão enche rápido" ;;
    *sd[a-z]*) MIDIA="USB / SATA" ;;
    *)        MIDIA="desconhecida" ;;
esac
AVAIL_KB="$(df --output=avail / 2>/dev/null | tail -1 | tr -d ' ')"
USED_PCT="$(df --output=pcent / 2>/dev/null | tail -1 | tr -d ' %')"
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))

echo
echo "════════════════════════════════════════════════════"
echo " 2. RAIZ (/)"
echo "════════════════════════════════════════════════════"
echo "  dispositivo : $ROOT_DEV"
echo "  mídia       : $MIDIA"
echo "  livre       : ${AVAIL_GB} GB"
echo "  usado       : ${USED_PCT}%"
echo
if   [ "$AVAIL_GB" -lt 5  ]; then
    echo "  🔴 CRÍTICO — o JetPack não cabe. Limpe antes (seção 4)."
elif [ "$AVAIL_GB" -lt 15 ]; then
    echo "  🟡 APERTADO — JetPack pede ~12-15 GB durante a instalação."
    echo "     Provavelmente dá, mas limpe o cache antes pra ter folga."
else
    echo "  🟢 OK — espaço suficiente pro JetPack e pras engines."
fi

echo
echo "════════════════════════════════════════════════════"
echo " 3. QUEM ESTÁ OCUPANDO"
echo "════════════════════════════════════════════════════"
echo "(pode levar 1-2 min em cartão microSD…)"
sudo du -h -d1 -x /usr /var /opt /home 2>/dev/null | sort -rh | head -14

echo
echo "════════════════════════════════════════════════════"
echo " 4. O QUE DÁ PRA LIMPAR COM SEGURANÇA"
echo "════════════════════════════════════════════════════"
APT_CACHE="$(sudo du -sh /var/cache/apt/archives 2>/dev/null | cut -f1)"
echo "  cache do apt (.deb já instalados) : ${APT_CACHE:-?}"
echo "     → sudo apt clean"
echo
echo "  logs do systemd                   : $(journalctl --disk-usage 2>/dev/null | grep -o '[0-9.]*[KMG]' | head -1 || echo '?')"
echo "     → sudo journalctl --vacuum-size=100M"
echo
echo "  pacotes órfãos                    :"
sudo apt-get -s autoremove 2>/dev/null | grep -E "^Remv|will be removed" | head -3 || echo "     (nenhum)"
echo "     → sudo apt autoremove"
echo
echo "  ⚠️  NÃO apague /usr/src/tensorrt — é onde vive o trtexec."
echo "  ⚠️  Só remova samples de CUDA (/usr/local/cuda/samples) se precisar mesmo."

echo
echo "════════════════════════════════════════════════════"
echo " 5. ARQUIVOS DO PROJETO"
echo "════════════════════════════════════════════════════"
for f in *.onnx *.engine; do
    [ -e "$f" ] && printf "  %-45s %s\n" "$f" "$(du -h "$f" | cut -f1)"
done 2>/dev/null
echo "  (a .engine gerada fica com tamanho parecido com o .onnx)"
