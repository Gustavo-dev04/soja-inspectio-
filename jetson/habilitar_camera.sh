#!/usr/bin/env bash
# Vígil.ia — habilita a câmera CSI mesclando o overlay no DTB, offline.
#
#     sudo ./habilitar_camera.sh            # IMX219 no CAM0
#     sudo ./habilitar_camera.sh imx219-C   # CAM1
#     sudo ./habilitar_camera.sh --desfazer # volta ao estado anterior
#
# Por que não usar o jetson-io: no JetPack 6/7 ele quebra com "No DTB found",
# porque o DTB base passou a vir da partição e não do /boot.
#
# Por que não confiar só no OVERLAYS do extlinux.conf: o bootloader aplica (ou
# descarta) o overlay em silêncio, e a única forma de saber é reiniciar e olhar
# o dmesg. Aqui a mesclagem é feita com fdtoverlay ANTES do boot, e o script
# CONFERE que o nó imx219 entrou no DTB resultante. Se não entrou, ele para e
# não mexe no boot — em vez de te mandar reiniciar na esperança.
set -uo pipefail

VERDE=$'\033[32m'; VERM=$'\033[31m'; AMAR=$'\033[33m'; FIM=$'\033[0m'
ok()   { echo "${VERDE}OK${FIM}   $*"; }
aviso(){ echo "${AMAR}AVISO${FIM} $*"; }
erro() { echo "${VERM}ERRO${FIM} $*"; }

EXTLINUX=/boot/extlinux/extlinux.conf
SAIDA=/boot/vigilia-camera.dtb
CAMERA="${1:-imx219-A}"

if [ "$(id -u)" != "0" ]; then
    erro "precisa de sudo (mexe em /boot)"; exit 1
fi

# ------------------------------------------------------------------ desfazer
if [ "$CAMERA" = "--desfazer" ]; then
    if [ -f "$EXTLINUX.vigilia.bak" ]; then
        cp "$EXTLINUX.vigilia.bak" "$EXTLINUX"
        ok "extlinux.conf restaurado. Reinicie: sudo reboot"
    else
        erro "não achei $EXTLINUX.vigilia.bak"
    fi
    exit 0
fi

command -v fdtoverlay >/dev/null || { erro "instale: sudo apt install -y device-tree-compiler"; exit 1; }
command -v dtc >/dev/null        || { erro "instale: sudo apt install -y device-tree-compiler"; exit 1; }

echo "=============================================================="
echo " 1. Qual placa está bootando"
echo "=============================================================="

# A primeira string de 'compatible' identifica o par carrier+módulo. É ela que
# precisa bater com o DTB base — usar o DTB de outro módulo boota errado.
COMPAT="$(tr '\0' '\n' < /proc/device-tree/compatible | head -1)"
MODELO="$(tr -d '\0' < /proc/device-tree/model)"
echo "  modelo    : $MODELO"
echo "  compatible: $COMPAT"

# nvidia,p3768-0000+p3767-0005  ->  tegra234-p3768-0000+p3767-0005
ALVO="tegra234-${COMPAT#nvidia,}"
echo "  procurando: /boot/${ALVO}*.dtb"

echo
echo "=============================================================="
echo " 2. DTB base"
echo "=============================================================="

# Ordem de preferência: -nv é o que a NVIDIA usa em produção; o simples é o
# genérico. O -super só entra se a placa declarar 'super' no compatible.
CANDIDATOS=()
case "$COMPAT" in
    *-super) CANDIDATOS=("/boot/${ALVO}-nv-super.dtb" "/boot/${ALVO}-super.dtb") ;;
esac
CANDIDATOS+=("/boot/${ALVO}-nv.dtb" "/boot/${ALVO}.dtb")

BASE=""
for c in "${CANDIDATOS[@]}"; do
    [ -f "$c" ] && { BASE="$c"; break; }
done
if [ -z "$BASE" ]; then
    erro "nenhum DTB base para '$ALVO' em /boot"
    ls /boot/*.dtb 2>/dev/null | head -20 | sed 's/^/    /'
    exit 1
fi
ok "base: $BASE"

OVERLAY="/boot/tegra234-p3767-camera-p3768-${CAMERA}.dtbo"
if [ ! -f "$OVERLAY" ]; then
    erro "overlay não encontrado: $OVERLAY"
    echo "  disponíveis:"
    ls /boot/*p3768-imx*.dtbo 2>/dev/null | sed 's/^/    /'
    exit 1
fi
ok "overlay: $OVERLAY"

# O overlay traz a lista de placas em que ele pode ser aplicado. Conferir aqui
# evita gerar um DTB silenciosamente inútil.
if dtc -I dtb -O dts "$OVERLAY" 2>/dev/null | grep -q "$COMPAT"; then
    ok "o overlay declara compatibilidade com esta placa"
else
    aviso "o overlay NÃO lista '$COMPAT' — pode não aplicar corretamente"
fi

echo
echo "=============================================================="
echo " 3. Mesclando (offline, sem tocar no boot ainda)"
echo "=============================================================="

if ! fdtoverlay -i "$BASE" -o "$SAIDA" "$OVERLAY" 2>&1 | sed 's/^/    /'; then
    erro "fdtoverlay falhou"; exit 1
fi
[ -f "$SAIDA" ] || { erro "não gerou $SAIDA"; exit 1; }
ok "gerado: $SAIDA ($(du -h "$SAIDA" | cut -f1))"

echo
echo "=============================================================="
echo " 4. A VERIFICAÇÃO — o nó da câmera entrou mesmo?"
echo "=============================================================="

# É este passo que muda o jogo: em vez de reiniciar e torcer, a gente lê o DTB
# resultante agora. Se o imx219 não está aqui, não vai estar depois do boot.
N="$(dtc -I dtb -O dts "$SAIDA" 2>/dev/null | grep -c -i 'imx219' || true)"
if [ "$N" -lt 1 ]; then
    erro "o DTB mesclado NÃO contém nenhum nó imx219 ($N ocorrências)."
    echo "  A mesclagem não pegou. O boot NÃO foi alterado — nada a desfazer."
    echo "  Compare com o DTB base (deve dar 0) e me mostre a diferença:"
    echo "    dtc -I dtb -O dts $BASE 2>/dev/null | grep -c -i imx219"
    exit 1
fi
ok "$N referências a imx219 no DTB mesclado"
dtc -I dtb -O dts "$SAIDA" 2>/dev/null | grep -i -m3 'imx219' | sed 's/^/    /'

echo
echo "=============================================================="
echo " 5. Apontando o boot para o DTB mesclado"
echo "=============================================================="

cp "$EXTLINUX" "$EXTLINUX.vigilia.bak"
ok "backup: $EXTLINUX.vigilia.bak"

# Tira OVERLAYS/FDT antigos do bloco primary: o overlay já está DENTRO do DTB
# agora, e deixar os dois caminhos ativos só cria ambiguidade.
sed -i '/^      OVERLAYS /d; /^      FDT /d' "$EXTLINUX"
sed -i "/^      APPEND /a\\      FDT $SAIDA" "$EXTLINUX"

echo
sed -n '/^LABEL primary/,/^$/p' "$EXTLINUX" | sed 's/^/    /'

if [ "$(grep -c '^      FDT ' "$EXTLINUX")" != "1" ]; then
    erro "esperava exatamente 1 linha FDT — restaurando por segurança"
    cp "$EXTLINUX.vigilia.bak" "$EXTLINUX"
    exit 1
fi

echo
echo "=============================================================="
ok "pronto — reinicie:  sudo reboot"
echo "=============================================================="
cat <<'FIM'

Depois do boot:
    ls /dev/video*
    sudo dmesg | grep -i imx219

Se a placa NÃO bootar (tela preta, sem rede), o DTB não serviu. Recupere
montando o disco em outro computador e restaurando:
    cp /boot/extlinux/extlinux.conf.vigilia.bak /boot/extlinux/extlinux.conf

Ou, se ainda bootar, desfaça daqui mesmo:
    sudo ./habilitar_camera.sh --desfazer
FIM
