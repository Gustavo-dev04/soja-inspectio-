# -*- coding: utf-8 -*-
"""Gera a documentação completa do projeto Vígil.ia em PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

OUT = "documentacao_vigil_ia.pdf"
W, H = A4
MARGIN = 2.2 * cm

GREEN = colors.HexColor("#16a34a")
DARK = colors.HexColor("#0f172a")
GREY = colors.HexColor("#475569")
LIGHT = colors.HexColor("#f1f5f9")
LINE = colors.HexColor("#cbd5e1")

ss = getSampleStyleSheet()

def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)

H1 = style("H1", fontName="Helvetica-Bold", fontSize=16, textColor=DARK,
           spaceBefore=18, spaceAfter=8, leading=20)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=12, textColor=GREEN,
           spaceBefore=12, spaceAfter=5, leading=15)
BODY = style("BODY", fontName="Helvetica", fontSize=9.5, textColor=DARK,
             alignment=TA_JUSTIFY, leading=14, spaceAfter=6)
SMALL = style("SMALL", fontName="Helvetica", fontSize=8, textColor=GREY, leading=11)
BULLET = style("BULLET", parent=BODY, leftIndent=10, spaceAfter=3, alignment=TA_LEFT)
CODE = style("CODE", fontName="Courier", fontSize=8, textColor=DARK, leading=11,
             backColor=LIGHT, borderPadding=5, spaceAfter=6)
CELL = style("CELL", fontName="Helvetica", fontSize=8.2, textColor=DARK, leading=11)
CELLB = style("CELLB", parent=CELL, fontName="Helvetica-Bold")

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def sp(x=6): story.append(Spacer(1, x))
def rule(): story.append(HRFlowable(width="100%", thickness=0.6, color=LINE,
                                     spaceBefore=4, spaceAfter=8))

def bullets(items):
    flow = [ListItem(Paragraph(t, BULLET), leftIndent=8, value="•") for t in items]
    story.append(ListFlowable(flow, bulletType="bullet", start="•",
                              bulletColor=GREEN, bulletFontSize=8))
    sp(4)

def table(rows, widths, header=True, font=8.2):
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    sty = [
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        for i in range(len(rows)):
            if i and i % 2 == 0:
                sty.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(sty))
    story.append(t)
    sp(8)


# ----------------------------------------------------------------------------
# CAPA
# ----------------------------------------------------------------------------
story.append(Spacer(1, 4.5 * cm))
story.append(Paragraph("Vígil<font color='#16a34a'>.ia</font>",
             style("cover", fontName="Helvetica-Bold", fontSize=46,
                   alignment=TA_CENTER, textColor=DARK)))
sp(6)
story.append(Paragraph("Inspeção visual de grãos de soja por imagem",
             style("covsub", fontName="Helvetica", fontSize=13,
                   alignment=TA_CENTER, textColor=GREY)))
sp(18)
story.append(HRFlowable(width="40%", thickness=1.2, color=GREEN,
                        spaceBefore=4, spaceAfter=16, hAlign="CENTER"))
story.append(Paragraph("Documentação técnica do projeto",
             style("covdoc", fontName="Helvetica-Bold", fontSize=12,
                   alignment=TA_CENTER, textColor=DARK)))
sp(4)
story.append(Paragraph("Versão b1.0.0 (beta) &nbsp;·&nbsp; Demo FATEC &nbsp;·&nbsp; Junho de 2026",
             style("covmeta", fontName="Helvetica", fontSize=10,
                   alignment=TA_CENTER, textColor=GREY)))
sp(60)
story.append(Paragraph(
    "Classificação de grãos de soja em 5 classes de qualidade — YOLO11s-cls + OpenCV, "
    "com interface web e análise agronômica conversacional por IA.",
    style("covfoot", fontName="Helvetica-Oblique", fontSize=9,
          alignment=TA_CENTER, textColor=GREY)))
story.append(PageBreak())

# ----------------------------------------------------------------------------
# SUMÁRIO
# ----------------------------------------------------------------------------
h1("Sumário")
rule()
toc = [
    "1.  Visão geral do projeto",
    "2.  As cinco classes de grão",
    "3.  Arquitetura do sistema",
    "4.  Pipeline de inferência (modo 1 grão)",
    "5.  Stack tecnológico",
    "6.  O modelo: migração e desempenho",
    "7.  Deploy (Hugging Face Spaces, Vercel, Supabase)",
    "8.  Frontend e experiência do usuário",
    "9.  Bugs encontrados e correções",
    "10. Fine-tuning v1 — coleta das 500 imagens",
    "11. Histórico de desenvolvimento",
    "12. Estrutura do repositório",
    "13. Roadmap",
    "14. Limitações e observações honestas",
]
for t in toc:
    story.append(Paragraph(t, style("toc", fontName="Helvetica", fontSize=10.5,
                                     textColor=DARK, leading=20)))
story.append(PageBreak())

# ----------------------------------------------------------------------------
# 1. VISÃO GERAL
# ----------------------------------------------------------------------------
h1("1. Visão geral do projeto")
rule()
p("A <b>Vígil.ia</b> é uma aplicação de <b>classificação de grãos de soja por imagem</b>. "
  "O usuário fotografa um grão com a câmera do celular e o sistema identifica a classe de "
  "qualidade do grão, exibindo o resultado com a confiança do modelo e uma análise "
  "agronômica opcional gerada por IA.")
p("O objetivo imediato é <b>demonstrativo (demo para a FATEC)</b>: provar que o modelo "
  "reconhece cada classe de grão a partir de uma foto real. Não é, nesta fase, um sistema "
  "industrial — esteira, sensor NIR, soprador pneumático e hardware de borda são evolução "
  "futura e não fazem parte do escopo atual.")
h2("Fluxo, do começo ao fim")
story.append(Paragraph(
    "foto do celular &rarr; redução da imagem &rarr; classificação (YOLO11s-cls) &rarr; "
    "classe + confiança &rarr; tela de resultado &rarr; conversa opcional com a IA",
    CODE))
p("A marca <b>Vígil</b> remete a <i>vigilância / inspeção</i>; o sufixo <b>.ia</b> marca a "
  "camada de inteligência. O produto está em <b>beta (b1.0.0)</b>.")

# ----------------------------------------------------------------------------
# 2. CLASSES
# ----------------------------------------------------------------------------
h1("2. As cinco classes de grão")
rule()
p("O modelo classifica cada grão em uma de cinco classes. Os rótulos internos (em inglês) "
  "seguem o dataset; a interface usa os rótulos em português.")
table([
    ["Índice", "Rótulo (dataset)", "Rótulo (UI)", "Cor"],
    ["0", "broken", "Quebrado", "Roxo"],
    ["1", "immature", "Imaturo (verde)", "Verde-limão"],
    ["2", "intact", "Intacto", "Verde"],
    ["3", "skin-damaged", "Casca danificada", "Âmbar"],
    ["4", "spotted", "Manchado", "Vermelho"],
], widths=[1.6*cm, 4.2*cm, 4.2*cm, 3.2*cm])
p("A fronteira mais ambígua do dataset é <b>skin-damaged × broken</b> (ambos apresentam dano "
  "de superfície) — é onde o modelo mais erra.")

# ----------------------------------------------------------------------------
# 3. ARQUITETURA
# ----------------------------------------------------------------------------
h1("3. Arquitetura do sistema")
rule()
p("O sistema tem três peças independentes. O modelo de visão (torch/ultralytics) não cabe "
  "no limite serverless da Vercel, por isso a visão roda em um serviço separado.")
table([
    ["Parte", "Onde", "Responsabilidade"],
    ["Frontend + LLM", "Vercel (Next.js)",
     "Interface, tela de resultado e a rota /api/explain (chat com a IA via Groq). Não usa Python."],
    ["Visão (/inspect)", "Hugging Face Spaces (Docker)",
     "FastAPI + YOLO11s-cls + OpenCV. O modelo .pt embutido na imagem."],
    ["Banco de dados", "Supabase",
     "Tabelas inspecoes e lotes; persiste o resultado de cada inspeção."],
], widths=[3.4*cm, 4.0*cm, 9.0*cm])
h2("Fluxo entre as peças")
story.append(Paragraph(
    "Navegador &rarr; (foto) &rarr; HF Space /inspect (YOLO) &rarr; resultado + grava no Supabase<br/>"
    "Navegador &rarr; (pergunta) &rarr; Vercel /api/explain &rarr; Groq · Llama 3.3 (streaming)",
    CODE))

# ----------------------------------------------------------------------------
# 4. PIPELINE
# ----------------------------------------------------------------------------
h1("4. Pipeline de inferência (modo 1 grão)")
rule()
p("A versão atual opera em <b>modo 1 grão</b>: a foto inteira é classificada como um único "
  "grão. O modo multi-grão (segmentação OpenCV de vários grãos por foto) existe no código "
  "e é acionável por uma flag, ficando reservado para a fase de lote/esteira.")
bullets([
    "<b>Captura</b>: o usuário envia/fotografa um grão; a imagem é lida uma única vez.",
    "<b>Redução</b>: a imagem é reduzida para no máx. 1280&nbsp;px (JPEG) no navegador — cabe no "
    "armazenamento local e alivia o upload, sem afetar a classificação (o modelo usa 224&nbsp;px).",
    "<b>Classificação</b>: o backend recebe a imagem, o YOLO11s-cls prediz a classe e a confiança.",
    "<b>Mapeamento</b>: os nomes do modelo (ex.: 'Broken soybeans') são normalizados para os "
    "rótulos internos ('broken').",
    "<b>Resposta</b>: total de grãos, contagem por classe, caixa(s) e dimensões da imagem; o "
    "registro é gravado no Supabase.",
    "<b>Análise opcional</b>: na tela de resultado, o usuário pode conversar com a Vígil.ia "
    "sobre a classe identificada (IA não é chamada até que o usuário pergunte).",
])

# ----------------------------------------------------------------------------
# 5. STACK
# ----------------------------------------------------------------------------
h1("5. Stack tecnológico")
rule()
table([
    ["Camada", "Tecnologia"],
    ["Treino do modelo", "Ultralytics / PyTorch (Google Colab, GPU grátis)"],
    ["Modelo", "YOLO11s-cls (classificação) via transfer learning"],
    ["Segmentação (recorte)", "OpenCV clássico (grayscale → Otsu → maior contorno → bbox)"],
    ["Backend de visão", "FastAPI + Uvicorn (Docker no Hugging Face Spaces)"],
    ["LLM (análise)", "Groq · Llama 3.3 70B (API compatível com OpenAI)"],
    ["Frontend", "Next.js 14 + TypeScript + Tailwind CSS (Vercel)"],
    ["Banco de dados", "Supabase (PostgreSQL gerenciado)"],
    ["Dataset", "Roboflow 'SoyaBeans Classifications v2' — MIT, ~12.528 imagens, 400×400"],
], widths=[4.6*cm, 11.8*cm])
p("Pinos relevantes do backend: <font face='Courier'>torch==2.2.2</font>, "
  "<font face='Courier'>torchvision==0.17.2</font>, <font face='Courier'>ultralytics==8.3.0</font> "
  "— escolhidos para carregar o modelo YOLO11 sem erros (ver seção 9).")

# ----------------------------------------------------------------------------
# 6. MODELO
# ----------------------------------------------------------------------------
h1("6. O modelo: migração e desempenho")
rule()
p("O classificador de produção foi <b>migrado de EfficientNet-B0 (TensorFlow/Keras) para "
  "YOLO11s-cls (Ultralytics/PyTorch)</b> após um experimento controlado nas mesmas 57 fotos "
  "reais, mesmo recorte, mesmo split e augmentation equivalente. Mede-se a acurácia no "
  "<b>domínio real</b> (fotos de celular, fundo cinza).")
table([
    ["Modelo", "Acurácia (real)", "Gap treino-val"],
    ["EfficientNet-B0 — receita antiga", "64,0%", "25,0% (overfitting)"],
    ["EfficientNet-B0 — em igualdade de condições", "75,0%", "7,2%"],
    ["YOLO11s-cls — produção", "91,7%", "3,9%"],
], widths=[8.6*cm, 3.9*cm, 3.9*cm])
h2("Por que o YOLO generaliza melhor com poucas fotos")
bullets([
    "<b>AdamW + weight decay</b>: regularização no otimizador (vs Adam puro do EfficientNet).",
    "<b>EMA</b> (média móvel dos pesos): suaviza o checkpoint final e melhora a generalização.",
    "<b>Atenção espacial (C2PSA)</b>: o último bloco do YOLO11s foca na textura/cor do grão e "
    "ignora o fundo — exatamente o problema de domain shift.",
])
h2("O gargalo real: domain shift")
p("O modelo treina com fotos de <b>fundo preto</b> (dataset Roboflow); a foto do celular tem "
  "<b>fundo cinza/metálico</b> e luz variada. Sem adaptação, a acurácia despenca para <b>~3-8%</b>. "
  "O <b>fine-tuning no domínio real</b> (freeze parcial + augmentation forte de brilho/cor) "
  "recupera para 91,7%. Por isso o fine-tuning deixou de ser opcional.")
h2("Honestidade metodológica")
p("A validação real tinha apenas <b>12 fotos</b> — 91,7% equivale a 11/12. É uma tendência "
  "forte, não uma métrica estatística precisa. Por isso a próxima etapa é coletar 500 imagens "
  "(seção 10) para um val/test sólidos.")

# ----------------------------------------------------------------------------
# 7. DEPLOY
# ----------------------------------------------------------------------------
h1("7. Deploy")
rule()
h2("Backend de visão — Hugging Face Spaces")
bullets([
    "Space Docker <font face='Courier'>Guguinhaxd/soja-inspection-api</font> "
    "(URL: guguinhaxd-soja-inspection-api.hf.space).",
    "Dockerfile com usuário uid 1000, caches graváveis e porta 7860; modelo .pt embutido na imagem.",
    "Secrets do Space: <font face='Courier'>SUPABASE_URL</font> e "
    "<font face='Courier'>SUPABASE_KEY</font> (service_role).",
    "Healthcheck: <font face='Courier'>GET /health</font> → {\"status\":\"ok\"}.",
])
h2("Frontend — Vercel")
p("Import pelo painel com <b>Root Directory = frontend</b> e Framework Preset = Next.js. "
  "Variáveis de ambiente:")
table([
    ["Variável", "Uso"],
    ["NEXT_PUBLIC_SUPABASE_URL", "URL do Supabase (browser)"],
    ["NEXT_PUBLIC_SUPABASE_ANON_KEY", "Chave anon pública (browser)"],
    ["NEXT_PUBLIC_API_URL", "URL do HF Space (rota /inspect)"],
    ["GROQ_API_KEY", "Chave da Groq (server-side, sem prefixo NEXT_PUBLIC_)"],
], widths=[7.4*cm, 9.0*cm])
h2("Banco — Supabase")
p("Tabelas <font face='Courier'>inspecoes</font> (resultado de cada inspeção) e "
  "<font face='Courier'>lotes</font>, com RLS habilitado e políticas de leitura/inserção pública. "
  "Recomenda-se rotacionar as chaves após o deploy e revisar o RLS antes de uso público.")

# ----------------------------------------------------------------------------
# 8. FRONTEND / UX
# ----------------------------------------------------------------------------
h1("8. Frontend e experiência do usuário")
rule()
p("Identidade visual <b>preta e sofisticada</b>, inspirada na sobriedade do app do Claude, "
  "com um verde de marca como acento e um símbolo animado próprio.")
h2("Principais elementos")
bullets([
    "<b>Símbolo (InspectionLogo)</b>: uma íris/abertura de precisão (pupila, anéis, bezel de "
    "micro-ticks e palitinhos radiais) com um arco de varredura. Ao inspecionar, os palitinhos "
    "giram rápido e a íris devagar, em sentido horário — sensação de varredura.",
    "<b>Abertura (IntroSplash)</b>: ao entrar no site, o símbolo se desenha e a marca 'Vígil' "
    "aparece; depois o splash some e revela a tela inicial. Toca uma vez por sessão.",
    "<b>Tela inicial (hero)</b>: símbolo central + 'Inspecionar', aceitando clique ou arrastar.",
    "<b>Tela de resultado</b>: card de veredito com anel de confiança animado na cor da classe, "
    "selo de status (Íntegro × Defeito), tabela com barras de porcentagem, moldura da imagem com "
    "brilho na cor da classe e animações de entrada escalonadas.",
    "<b>Chat com a IA</b>: barra de digitação com respostas em <b>streaming</b> (token a token), "
    "histórico contextual, indicador 'digitando', botão limpar e <b>modo tela cheia</b>.",
    "<b>Aba Sobre</b>: explica o que é a Vígil.ia, o desempenho e a linha do tempo das atualizações.",
    "Acessibilidade: todas as animações respeitam <font face='Courier'>prefers-reduced-motion</font>.",
])

# ----------------------------------------------------------------------------
# 9. BUGS
# ----------------------------------------------------------------------------
h1("9. Bugs encontrados e correções")
rule()
p("A integração ponta a ponta revelou uma série de problemas reais, corrigidos um a um:")
table([
    ["Sintoma", "Causa", "Correção"],
    ["Space não subia (UnpicklingError)",
     "torch ≥ 2.6 usa weights_only=True; ultralytics 8.2.0 não allowlista os globals",
     "Fixar torch==2.2.2 / torchvision==0.17.2"],
    ["Erro 'Can't get attribute C3k2'",
     "O .pt é um YOLO11 (bloco C3k2), inexistente no ultralytics 8.2.0",
     "Subir ultralytics para 8.3.0"],
    ["ValueError 'Broken soybeans'",
     "O modelo usa rótulos longos; o código comparava com os curtos",
     "Normalizar os nomes de classe"],
    ["Upload dava erro mesmo com HTTP 200",
     "CORS com wildcard '*' + allow_credentials=True (inválido pelo spec)",
     "allow_credentials=False"],
    ["'Não consegui ler o arquivo'",
     "A mesma foto era lida duas vezes; a 2ª falhava (Safari/iOS)",
     "Ler o arquivo uma única vez e reusar"],
    ["Quota do sessionStorage estourava",
     "Data URL da foto do celular tem vários MB",
     "Reduzir a imagem antes de guardar/enviar"],
    ["Site travava na abertura",
     "sessionStorage lança no Safari/privado e o splash não terminava",
     "Blindar com try/catch + memória de sessão"],
    ["Tela cheia do chat 'quebrada' no celular",
     "Faltava min-h-0; a barra de digitação era empurrada pra fora",
     "min-h-0 + altura 100dvh + área segura"],
], widths=[4.7*cm, 6.0*cm, 5.7*cm])

# ----------------------------------------------------------------------------
# 10. FINE-TUNING 500
# ----------------------------------------------------------------------------
h1("10. Fine-tuning v1 — coleta das 500 imagens")
rule()
p("Para reduzir o domain shift e validar com solidez, a meta é coletar <b>~100 fotos por classe "
  "(500 no total)</b>, com captura manual, e usar o notebook <font face='Courier'>"
  "model/finetune_yolo.ipynb</font>, adaptado para ler do <b>Google Drive</b>.")
h2("Protocolo de captura")
bullets([
    "1 grão por foto, centralizado e em foco (o notebook recorta o maior contorno via Otsu).",
    "Mesmo setup da demo: o mesmo celular, fundo e luz que serão usados na apresentação.",
    "Variar de propósito dentro de cada classe (ângulo, iluminação, distância).",
    "Rotular corretamente: cada grão na pasta da sua classe verdadeira.",
    "Evitar: grãos se tocando, sombra dura, borrão, dois grãos no quadro.",
    "Atenção especial em skin-damaged × broken (a fronteira mais difícil).",
])
h2("Pipeline do notebook (FT-0 a FT-5)")
bullets([
    "FT-0 bootstrap: carrega o modelo base do Drive e define o recorte.",
    "FT-1: lê do Drive e faz split estratificado 70/15/15 (train/val/test).",
    "FT-2: baseline do modelo antes do fine-tuning.",
    "FT-3: fine-tuning (freeze parcial, augmentation forte, batch 32, 60 épocas).",
    "FT-4: antes × depois em val e test + matriz de confusão.",
    "FT-5: salva o modelo apenas se o val melhorar (gate de segurança), versionado.",
])

# ----------------------------------------------------------------------------
# 11. HISTÓRICO
# ----------------------------------------------------------------------------
h1("11. Histórico de desenvolvimento")
rule()
p("Resumo cronológico do trabalho (mais recente primeiro), agrupado por tema:")
h2("Modelo e migração")
bullets([
    "Migração de EfficientNet/Keras para YOLO11s-cls; experimento justo comparando as arquiteturas.",
    "Notebook de fine-tuning no domínio real (backbone congelado, augmentation forte).",
    "Adaptação do notebook para ler do Drive e escalar para 500 (split train/val/test).",
])
h2("Deploy e backend")
bullets([
    "Adaptação do backend de visão para Hugging Face Spaces (Docker) e publicação do Space.",
    "Correções de carga do modelo: torch<2.6, ultralytics 8.3.0, normalização de classes.",
    "Modo 1 grão por padrão (multi-grão preservado via flag).",
])
h2("Frontend e experiência")
bullets([
    "Tema preto + símbolo animado; marca Vígil.ia (b1.0.0 beta) e animação de abertura.",
    "Tela de resultado legível e agradável: veredito, anel de confiança, barras, status.",
    "IA opcional → chat com barra de digitação → streaming + tela cheia + limpar conversa.",
    "Remoção da Dashboard e criação da aba 'Sobre Vígil.ia'.",
])
h2("Estabilização (bugs)")
bullets([
    "CORS válido, leitura única do arquivo, redução de imagem (quota), erro de inspeção legível.",
    "Abertura sem flash e à prova de falha de storage; tela cheia do chat corrigida no celular.",
])

# ----------------------------------------------------------------------------
# 12. ESTRUTURA
# ----------------------------------------------------------------------------
h1("12. Estrutura do repositório")
rule()
story.append(Paragraph(
    "frontend/            App Next.js (UI, /api/explain, componentes)<br/>"
    "&nbsp;&nbsp;src/app/           páginas: home, resultado/[id], sobre, api/explain<br/>"
    "&nbsp;&nbsp;src/components/    InspectionLogo, IntroSplash, InspectHero, ResultVerdict,<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DefectTable, BoundingBoxCanvas, ExplainPanel<br/>"
    "&nbsp;&nbsp;src/lib/           api.ts, supabase.ts<br/>"
    "backend/             FastAPI (main.py, inference.py, database.py, Dockerfile)<br/>"
    "&nbsp;&nbsp;soja_yolo11s_finetuned.pt   modelo embutido<br/>"
    "model/               notebooks de treino e fine-tuning, classes, scripts<br/>"
    "supabase/migrations/ schema inicial (inspecoes, lotes)<br/>"
    "docs/                especificação e materiais de apoio",
    CODE))

# ----------------------------------------------------------------------------
# 13. ROADMAP
# ----------------------------------------------------------------------------
h1("13. Roadmap")
rule()
bullets([
    "Coletar 500 correções/fotos e treinar o fine-tuning v1.",
    "Reativar o modo multi-grão (inspeção de lote inteiro) via segmentação OpenCV.",
    "Router de domínio leve para despachar grãos × superfícies sob uma API única.",
    "Persistência das imagens e fluxo de correção humana para crescer o dataset.",
    "Evolução futura (fora do escopo da demo): esteira, NIR, ejeção pneumática, hardware de borda.",
])

# ----------------------------------------------------------------------------
# 14. LIMITAÇÕES
# ----------------------------------------------------------------------------
h1("14. Limitações e observações honestas")
rule()
bullets([
    "Validação real ainda pequena (12 fotos) — 91,7% é tendência, não métrica estatística.",
    "Domain shift é o gargalo: a captura precisa imitar o setup do dataset/uso real.",
    "O Hugging Face Spaces (free) hiberna após inatividade; a 1ª chamada tem cold start.",
    "skin-damaged × broken é a confusão mais provável do modelo.",
    "Segurança: revisar RLS e rotacionar chaves antes de qualquer uso público.",
])
sp(10)
rule()
story.append(Paragraph(
    "Vígil.ia · b1.0.0 (beta) · Documentação gerada em junho de 2026. "
    "Projeto de classificação de grãos de soja — demo acadêmica (FATEC).",
    style("end", fontName="Helvetica-Oblique", fontSize=8.5,
          alignment=TA_CENTER, textColor=GREY)))


# ----------------------------------------------------------------------------
# rodapé com número de página
# ----------------------------------------------------------------------------
def footer(canvas, doc_):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.4 * cm, W - MARGIN, 1.4 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    if doc_.page > 1:
        canvas.drawString(MARGIN, 1.0 * cm, "Vígil.ia — Documentação técnica")
        canvas.drawRightString(W - MARGIN, 1.0 * cm, f"Página {doc_.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=MARGIN, bottomMargin=1.8 * cm,
                        title="Vígil.ia — Documentação técnica",
                        author="Vígil.ia")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("OK ->", OUT)
