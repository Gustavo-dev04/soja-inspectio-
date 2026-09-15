#!/usr/bin/env bash
# Vígil.ia — lançador de APRESENTAÇÃO.
#
#     ./demo.sh              # acha a câmera sozinho e abre em tela cheia
#     ./demo.sh nano         # usa a engine do nano em vez do small
#     ./demo.sh --video x.mp4  # roda um arquivo (plano B sem rede)
#     ./demo.sh --atalho     # cria o ícone clicável na área de trabalho
#
# Feito para o pior cenário de demo: o IP do celular mudou de manhã, o Wi-Fi do
# auditório é outro, e você tem trinta segundos com a plateia olhando. Ele tenta
# quatro caminhos até achar a câmera, guarda o que funcionou, e se nada
# responder cai num vídeo gravado em vez de travar na sua frente.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUI"

PORTA=4747
CACHE="$HOME/.vigilia_camera"          # último IP que funcionou
CONF=0.10                              # demo é fora do domínio de treino:
                                       # confiança baixa mostra mais caixa
VARIANTE="small"
VIDEO=""
CRIAR_ATALHO=0

while [ $# -gt 0 ]; do
    case "$1" in
        nano|small)  VARIANTE="$1"; shift ;;
        --atalho)    CRIAR_ATALHO=1; shift ;;
        --video)     VIDEO="${2:-}"; shift 2 ;;
        --conf)      CONF="${2:-}"; shift 2 ;;
        *)           echo "opção desconhecida: $1"; exit 1 ;;
    esac
done

# Atalho na área de trabalho: o caminho é resolvido AGORA, na instalação, em
# vez de ficar fixo num arquivo versionado que quebraria em outra máquina.
if [ "$CRIAR_ATALHO" = "1" ]; then
    APPS="$HOME/.local/share/applications"
    mkdir -p "$APPS"
    cat > "$APPS/vigilia-demo.desktop" <<ATALHO
[Desktop Entry]
Type=Application
Name=Vígil.ia — Inspeção de Soja
Comment=Inspeção de grãos de soja ao vivo (RF-DETR + TensorRT)
Exec=bash -c 'cd "$AQUI" && ./demo.sh; echo; echo "[Enter para fechar]"; read _'
Path=$AQUI
Terminal=true
Categories=Science;
ATALHO
    chmod +x "$APPS/vigilia-demo.desktop"
    for DESK in "$HOME/Área de Trabalho" "$HOME/Desktop" "$HOME/Área de trabalho"; do
        if [ -d "$DESK" ]; then
            cp "$APPS/vigilia-demo.desktop" "$DESK/"
            chmod +x "$DESK/vigilia-demo.desktop"
            gio set "$DESK/vigilia-demo.desktop" metadata::trusted true 2>/dev/null
            echo "atalho na área de trabalho: $DESK/vigilia-demo.desktop"
        fi
    done
    echo "atalho no menu de aplicativos: $APPS/vigilia-demo.desktop"
    echo
    echo "Se o ícone aparecer com um aviso, clique com o botão direito ->"
    echo "'Permitir execução' (o GNOME pede isso uma vez por atalho novo)."
    exit 0
fi

echo "=============================================="
echo " Vígil.ia — inspeção de soja ao vivo"
echo "=============================================="

# --------------------------------------------------------------- 1. a engine
ENGINE="$(ls -1 *"${VARIANTE}"*.engine 2>/dev/null | head -1)"
if [ -z "$ENGINE" ]; then
    ENGINE="$(ls -1 *.engine 2>/dev/null | head -1)"
    [ -n "$ENGINE" ] && echo "(sem engine '$VARIANTE'; usando $ENGINE)"
fi
if [ -z "$ENGINE" ]; then
    echo "ERRO: nenhuma engine .engine nesta pasta."
    echo "      Construa com:  ./setup_jetson.sh"
    exit 1
fi
echo "modelo : $ENGINE"

# ------------------------------------------------------------- 2. a câmera
testa() {  # $1 = ip; responde rápido? (o DroidCam devolve 400 no HEAD, e isso
           # JÁ conta como vivo: o que importa é ter alguém escutando na porta)
    curl -s --max-time 1.5 -o /dev/null "http://$1:$PORTA/video" && return 0
    curl -s --max-time 1.5 -o /dev/null -I "http://$1:$PORTA/video" 2>/dev/null
    # 400/404 = servidor respondeu; só timeout/recusa é que reprovam
    [ $? -eq 0 ] || return 1
}

achar_camera() {
    local ip

    # (a) o que você exportou tem prioridade
    if [ -n "${VIGIL_CAMERA:-}" ]; then
        ip="$(echo "$VIGIL_CAMERA" | sed -E 's|https?://||; s|:.*||')"
        echo "  testando VIGIL_CAMERA ($ip)…" >&2
        testa "$ip" && { echo "$ip"; return 0; }
    fi

    # (b) o último que funcionou
    if [ -f "$CACHE" ]; then
        ip="$(cat "$CACHE")"
        echo "  testando o último que funcionou ($ip)…" >&2
        testa "$ip" && { echo "$ip"; return 0; }
    fi

    # (c) o gateway — quando o celular é o roteador, ele É o gateway
    ip="$(ip route | awk '/^default/ {print $3; exit}')"
    if [ -n "$ip" ]; then
        echo "  testando o gateway ($ip)…" >&2
        testa "$ip" && { echo "$ip"; return 0; }
    fi

    # (d) varre a rede local em paralelo (~3 s numa /24)
    local base
    base="$(ip -4 -o addr show scope global | awk '{print $4}' | head -1 \
            | cut -d/ -f1 | cut -d. -f1-3)"
    if [ -n "$base" ]; then
        echo "  varrendo ${base}.0/24 na porta $PORTA…" >&2
        for n in $(seq 1 254); do
            ( testa "${base}.$n" && echo "${base}.$n" > "$CACHE.tmp" ) &
        done 2>/dev/null
        wait
        if [ -f "$CACHE.tmp" ]; then
            ip="$(cat "$CACHE.tmp")"; rm -f "$CACHE.tmp"
            echo "$ip"; return 0
        fi
    fi
    return 1
}

if [ -n "$VIDEO" ]; then
    [ -f "$VIDEO" ] || { echo "ERRO: não achei $VIDEO"; exit 1; }
    FONTE=(--source "$VIDEO")
    echo "fonte  : $VIDEO (arquivo)"
else
    echo "câmera : procurando…"
    if IP="$(achar_camera)"; then
        echo "$IP" > "$CACHE"
        FONTE=(--camera "http://$IP:$PORTA/video")
        echo "câmera : http://$IP:$PORTA/video"
    else
        # Plano B: nada de travar na frente da plateia. Se houver vídeo gravado,
        # a demo continua — só que sem interação ao vivo.
        FALLBACK="$(ls -1 *.mp4 2>/dev/null | head -1)"
        if [ -n "$FALLBACK" ]; then
            echo
            echo "  Nenhuma câmera respondeu na rede."
            echo "  >>> Caindo para o vídeo gravado: $FALLBACK"
            echo
            FONTE=(--source "$FALLBACK")
        else
            echo
            echo "ERRO: nenhuma câmera na rede e nenhum .mp4 de reserva."
            echo "  1. o DroidCam está ABERTO no celular?"
            echo "  2. Jetson e celular na mesma rede?  ip -4 addr | grep inet"
            echo "  3. force o IP:  VIGIL_CAMERA=http://IP:$PORTA/video ./demo.sh"
            exit 1
        fi
    fi
fi

echo "conf   : $CONF   (baixa de propósito: a demo é fora do domínio de treino)"
echo
echo "  q sai  ·  c zera a contagem  ·  p pausa"
echo "=============================================="
echo

exec python3 vigil_jetson.py --engine "$ENGINE" "${FONTE[@]}" \
     --conf "$CONF" --tela-cheia
