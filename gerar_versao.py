"""
Gera o PDF de versão do modelo (v0, v1, ...) para registro de progressão.
Uso: python gerar_versao.py v0
"""
import sys
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v0"

VERSIONS = {
    "v0": {
        "label":       "v0 — Modelo Base",
        "subtitle":    "Treinado inteiramente no dataset Roboflow (sem dados reais do usuário)",
        "date":        "Junho 2026",
        "status":      "BASELINE — Em produção no HF Space",
        "status_color": colors.HexColor("#e67e22"),
        "sections": "v0",
    },
    "v1": {
        "label":       "v1 — Fine-tuned com Correções Reais",
        "subtitle":    "Fine-tuning sobre o v0 com ~600 fotos reais coletadas pelo usuário",
        "date":        "A definir",
        "status":      "PLANEJADO — Aguardando coleta completa",
        "status_color": colors.HexColor("#27ae60"),
        "sections": "v1",
    },
}

if VERSION not in VERSIONS:
    print(f"Versão desconhecida: {VERSION}. Use: {list(VERSIONS.keys())}")
    sys.exit(1)

V = VERSIONS[VERSION]
OUTPUT = f"modelo_{VERSION}.pdf"

# ── Estilos ───────────────────────────────────────────────────────────
W, H   = A4
MARGIN = 2.2 * cm

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN)

ss   = getSampleStyleSheet()
AZUL = colors.HexColor("#2c3e50")
VERD = colors.HexColor("#27ae60")
CINZ = colors.HexColor("#7f8c8d")
LROX = colors.HexColor("#8e44ad")

def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

titulo   = style("T",  fontSize=20, textColor=AZUL, alignment=TA_CENTER,
                 fontName="Helvetica-Bold", spaceAfter=2)
versao   = style("V",  fontSize=13, textColor=VERD, alignment=TA_CENTER,
                 fontName="Helvetica-Bold", spaceAfter=2)
sub      = style("S",  fontSize=9,  textColor=CINZ, alignment=TA_CENTER, spaceAfter=10)
h1       = style("H1", fontSize=13, textColor=VERD, fontName="Helvetica-Bold",
                 spaceBefore=16, spaceAfter=6)
h2       = style("H2", fontSize=10, textColor=AZUL, fontName="Helvetica-Bold",
                 spaceBefore=8,  spaceAfter=4)
body     = style("B",  fontSize=9,  leading=15, spaceAfter=5, alignment=TA_JUSTIFY,
                 textColor=colors.HexColor("#333333"))
badge    = style("BD", fontSize=9,  fontName="Helvetica-Bold", alignment=TA_CENTER)
note     = style("N",  fontSize=8,  textColor=CINZ, leading=13, spaceAfter=4,
                 leftIndent=10)

def hr():
    return HRFlowable(width="100%", thickness=0.8,
                      color=colors.HexColor("#dcdde1"),
                      spaceAfter=8, spaceBefore=4)

def tbl(data, widths, header=True, hcol=AZUL):
    t = Table(data, colWidths=widths)
    s = [
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
    ]
    if header:
        s += [
            ("BACKGROUND", (0,0), (-1,0), hcol),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(s))
    return t

def status_badge(text, bg):
    data = [[Paragraph(f"<b>{text}</b>", badge)]]
    t = Table(data, colWidths=[16*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg),
        ("TEXTCOLOR",     (0,0), (-1,-1), colors.white),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("ROUNDEDCORNERS",(0,0), (-1,-1), [4,4,4,4]),
    ]))
    return t

def metric_table(rows):
    """Tabela de métricas com destaque na coluna de valor."""
    data = [["Métrica", "Valor", "Observação"]] + rows
    t = Table(data, colWidths=[5*cm, 3*cm, 8.2*cm])
    s = [
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("BACKGROUND",    (0,0), (-1,0),  AZUL),
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (1,1), (1,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (1,1), (1,-1),  VERD),
    ]
    t.setStyle(TableStyle(s))
    return t

# ── Conteúdo ─────────────────────────────────────────────────────────
story = []

# Cabeçalho
story += [
    Spacer(1, 0.8*cm),
    Paragraph("Classificador de Grãos de Soja", titulo),
    Paragraph(f"Registro de Versão · {V['label']}", versao),
    Paragraph(V["subtitle"], sub),
    Paragraph(f"Data: {V['date']}  ·  Projeto: FATEC — Gustavo Barros", sub),
    Spacer(1, 0.3*cm),
    status_badge(V["status"], V["status_color"]),
    Spacer(1, 0.4*cm),
    hr(),
]

# ════════════════════════════════════════════════════════════════════════
if VERSION == "v0":

    # 1. Resumo executivo
    story.append(Paragraph("1. Resumo", h1))
    story.append(Paragraph(
        "O modelo v0 é a base do projeto: EfficientNet-B0 treinada com transfer learning "
        "sobre o dataset público SoyaBeans Classifications v2 (Roboflow, MIT). "
        "Ele demonstrou alta acurácia no dataset de laboratório (≥85%) mas sofre "
        "<b>domain shift severo</b> ao classificar fotos reais tiradas com celular — "
        "acurácia caía para ~29% no ambiente do usuário. "
        "A v1 resolve isso via fine-tuning com fotos reais coletadas pelo próprio usuário.", body))

    # 2. Identificação
    story.append(Paragraph("2. Identificação do Modelo", h1))
    story.append(tbl([
        ["Atributo",            "Valor"],
        ["Versão",              "v0 — Modelo Base"],
        ["Arquivo",             "soja_model_final.keras"],
        ["Tamanho",             "~29 MB (Git LFS)"],
        ["Arquitetura",         "EfficientNet-B0 + cabeça densa 5 classes"],
        ["Framework",           "TensorFlow / Keras ≥ 2.15"],
        ["Deploy",              "Hugging Face Space: Guguinhaxd/soja-inspection"],
        ["Repositório",         "Gustavo-dev04/soja-inspectio-"],
        ["Branch de dev",       "claude/soja-inspection-setup-b2jaG"],
        ["Data de treino",      "Junho 2026"],
        ["Dataset",             "SoyaBeans Classifications v2 (Roboflow — hansaka-sudusinghe)"],
        ["Licença do dataset",  "MIT (uso comercial liberado)"],
    ], [5*cm, 11.2*cm]))

    # 3. Dataset
    story.append(Paragraph("3. Dataset de Treinamento", h1))
    story.append(tbl([
        ["Split",      "Imagens", "Observação"],
        ["Treino",     "10.905",  "Com augmentation Roboflow (flip, rotação, crop, shear)"],
        ["Validação",  "1.038",   "—"],
        ["Teste",      "527",     "Nunca visto durante treino"],
        ["Total",      "12.470",  "400×400 px → resize 224×224 no pipeline Keras"],
    ], [3*cm, 3*cm, 10.2*cm]))
    story.append(Paragraph(
        "A pasta 'Part of the original soybean images' presente em cada split foi "
        "excluída via parâmetro class_names explícito no image_dataset_from_directory.", note))

    # 4. Arquitetura e Treino
    story.append(Paragraph("4. Arquitetura e Estratégia de Treino", h1))
    story.append(tbl([
        ["Componente",        "Detalhe"],
        ["Backbone",          "EfficientNetB0 (ImageNet, include_top=False, input 224×224×3)"],
        ["Pré-processamento", "efficientnet.preprocess_input"],
        ["Pooling",           "GlobalAveragePooling2D"],
        ["Regularização",     "BatchNormalization + Dropout(0.3)"],
        ["Cabeça",            "Dense(5, activation='softmax')"],
    ], [4.5*cm, 11.7*cm]))

    story.append(Paragraph("Estratégia em 2 fases:", h2))
    story.append(tbl([
        ["",            "Fase 1 — Cabeça",              "Fase 2 — Fine-tuning backbone"],
        ["Base",        "Congelada",                    "Últimas 30 camadas descongeladas"],
        ["BatchNorm",   "—",                            "Mantida congelada (preserva ImageNet)"],
        ["LR",          "1e-3 (Adam)",                  "1e-5 (Adam)"],
        ["Épocas",      "até 25 (EarlyStopping p=5)",   "até 15 (EarlyStopping p=5)"],
        ["class_weight","balanced (sklearn)",           "balanced (sklearn)"],
    ], [3*cm, 5.5*cm, 7.7*cm]))

    # 5. Métricas
    story.append(Paragraph("5. Métricas Observadas", h1))
    story.append(metric_table([
        ["Acurácia — dataset original (teste)", "≥ 85%",  "Conjunto de teste Roboflow (527 imgs nunca vistas)"],
        ["Acurácia — fotos reais (pré-correção)","~29%",  "Domain shift severo: classificava quase tudo como Quebrado 99%"],
        ["Acurácia — fotos reais (pós 57 correções)", "64%", "Fine-tuning de validação: overfitting (treino 89% / val 64%)"],
        ["Gap treino/val (fine-tuning rodada 1)", "~25 pp", "Causa: 57 fotos insuficientes para o peso aplicado"],
    ]))

    # 6. Problemas identificados
    story.append(Paragraph("6. Problemas Identificados", h1))
    story.append(tbl([
        ["Problema",             "Causa Raiz",                          "Solução na v1"],
        ["Domain shift severo",  "Modelo nunca viu fotos de celular do usuário",
                                 "Fine-tuning com 600 fotos reais do setup do usuário"],
        ["Overfitting (rodada 1)","57 fotos insuficientes para CORR_WEIGHT=2.0",
                                 "Coleta de ~600 fotos; peso ajusta para 1.5 automaticamente"],
        ["Classe Quebrado dominant.","Modelo devolvia Quebrado 99% para qualquer entrada",
                                 "Resolvido com fundo escuro + auto-crop; fine-tuning consolida"],
    ], [3.5*cm, 5.5*cm, 7.2*cm]))

    # 7. O que está em produção
    story.append(Paragraph("7. Estado Atual do Sistema (v0)", h1))
    story.append(tbl([
        ["Componente",           "Estado"],
        ["HF Space",             "Online — Guguinhaxd/soja-inspection"],
        ["Modelo no Space",      "soja_model_final.keras (v0, 29 MB, Git LFS)"],
        ["Aba 'Um grão'",        "Funcional — auto-crop + classificação + feedback de correção"],
        ["Aba 'Vários grãos'",   "Funcional — segmentação OpenCV + bounding boxes por classe"],
        ["Dataset de correções", "Guguinhaxd/soja-correction — 57 fotos coletadas (em crescimento)"],
        ["Secret configurado",   "SOJA_CORRECTIONS — token HF write para salvar correções"],
        ["Notebook de treino",   "model/train.ipynb — reproduzível no Google Colab (T4 GPU)"],
        ["Notebook fine-tuning", "model/finetune.ipynb — pronto para rodar com 600+ fotos"],
    ], [4.5*cm, 11.7*cm]))

    # 8. Definição de v1
    story.append(Paragraph("8. Critérios de Transição para v1", h1))
    story.append(Paragraph(
        "A versão v1 será gerada quando os seguintes critérios forem atendidos:", body))
    story.append(tbl([
        ["Critério",                        "Meta",     "Estado"],
        ["Correções coletadas no dataset",  "≥ 500 fotos", "Em andamento (~57 coletadas)"],
        ["Distribuição por classe",         "≥ 80 por classe", "Em andamento"],
        ["Acurácia nas fotos reais (val)",  "≥ 80%",    "Aguardando fine-tuning v1"],
        ["Acurácia no teste original",      "≥ 70%",    "Aguardando fine-tuning v1"],
        ["Gap treino/val",                  "< 15 pp",  "Aguardando fine-tuning v1"],
    ], [6*cm, 4*cm, 6.2*cm]))
    story.append(Paragraph(
        "O fine-tuning da v1 será executado via model/finetune.ipynb com CORR_WEIGHT "
        "automático (2.0 se <150 fotos, 1.5 se ≥150), mix 70/30, augmentation forte nas "
        "correções e split estratificado por classe.", note))

# ════════════════════════════════════════════════════════════════════════
elif VERSION == "v1":
    story.append(Paragraph("— Documento gerado após conclusão do fine-tuning —", h1))
    story.append(Paragraph(
        "Este arquivo é o template da v1. Preencha as métricas após rodar o finetune.ipynb "
        "com as correções completas e execute: python gerar_versao.py v1", body))

# Rodapé
story += [
    Spacer(1, 0.5*cm),
    hr(),
    Paragraph(
        f"Gerado em {date.today().strftime('%d/%m/%Y')}  ·  "
        f"Projeto Classificador de Grãos de Soja — FATEC  ·  {V['label']}",
        style("foot", fontSize=7.5, textColor=CINZ, alignment=TA_CENTER)
    ),
]

doc.build(story)
print(f"PDF gerado: {OUTPUT}")
