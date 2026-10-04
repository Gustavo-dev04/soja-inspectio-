# -*- coding: utf-8 -*-
"""Gera o documento de contexto do projeto Vígil.ia em PDF.

Objetivo: dar a um grupo novo tudo que precisa pra assumir o projeto — o que
existe, por que existe, o que já foi tentado e descartado, e o que falta.

    python3 gerar_contexto_pdf.py
"""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = 'contexto_projeto_vigilia.pdf'
W, H = A4
MARGIN = 2.0 * cm

VERDE = colors.HexColor('#16a34a')
ESCURO = colors.HexColor('#0f172a')
CINZA = colors.HexColor('#475569')
CLARO = colors.HexColor('#f1f5f9')
LINHA = colors.HexColor('#cbd5e1')
VERMELHO = colors.HexColor('#dc2626')
AMBAR = colors.HexColor('#d97706')

ss = getSampleStyleSheet()


def estilo(nome, **kw):
    base = kw.pop('parent', ss['Normal'])
    return ParagraphStyle(nome, parent=base, **kw)


H1 = estilo('H1', fontName='Helvetica-Bold', fontSize=15, textColor=ESCURO,
            spaceBefore=16, spaceAfter=7, leading=19)
H2 = estilo('H2', fontName='Helvetica-Bold', fontSize=11, textColor=VERDE,
            spaceBefore=11, spaceAfter=4, leading=14)
BODY = estilo('BODY', fontName='Helvetica', fontSize=9.3, textColor=ESCURO,
              alignment=TA_JUSTIFY, leading=13.5, spaceAfter=5)
BULLET = estilo('BULLET', parent=BODY, leftIndent=9, spaceAfter=2.5, alignment=TA_LEFT)
CODE = estilo('CODE', fontName='Courier', fontSize=7.8, textColor=ESCURO, leading=10.5,
              backColor=CLARO, borderPadding=5, spaceAfter=6)
CELL = estilo('CELL', fontName='Helvetica', fontSize=7.9, textColor=ESCURO, leading=10.5)
# cabeçalho: o textColor do Paragraph SOBREPÕE o TEXTCOLOR da Table, então a cor
# branca precisa estar aqui — senão o cabeçalho some (escuro sobre escuro).
CELLH = estilo('CELLH', parent=CELL, fontName='Helvetica-Bold', textColor=colors.white)
NOTA = estilo('NOTA', parent=BODY, fontSize=8.6, textColor=CINZA, leading=12)

story = []


def h1(t):
    story.append(Paragraph(t, H1))
    story.append(HRFlowable(width='100%', thickness=0.6, color=LINHA,
                            spaceBefore=2, spaceAfter=7))


def h2(t):
    story.append(Paragraph(t, H2))


def p(t):
    story.append(Paragraph(t, BODY))


def nota(t):
    story.append(Paragraph(t, NOTA))


def code(t):
    # o Paragraph colapsa espaços repetidos, o que desalinha diagrama e tabela
    # de arquivos — converte para espaço rígido antes de renderizar
    linhas = []
    for linha in t.split('\n'):
        fora = []
        i = 0
        while i < len(linha):
            if linha[i] == '<':                  # preserva tags (<b>, <font>)
                j = linha.find('>', i)
                if j == -1:
                    j = len(linha) - 1
                fora.append(linha[i:j + 1])
                i = j + 1
            elif linha[i] == ' ':
                fora.append('&nbsp;')
                i += 1
            else:
                fora.append(linha[i])
                i += 1
        linhas.append(''.join(fora))
    story.append(Paragraph('<br/>'.join(linhas), CODE))


def sp(x=5):
    story.append(Spacer(1, x))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=12, value='•') for t in items],
        bulletType='bullet', start='•', bulletColor=VERDE, bulletFontSize=7))
    sp(4)


def tabela(linhas, larguras, cabecalho=True, destaques=()):
    dados = [[Paragraph(str(c), CELLH if (cabecalho and i == 0) else CELL)
              for c in linha] for i, linha in enumerate(linhas)]
    t = Table(dados, colWidths=larguras, repeatRows=1 if cabecalho else 0)
    sty = [('GRID', (0, 0), (-1, -1), 0.4, LINHA),
           ('VALIGN', (0, 0), (-1, -1), 'TOP'),
           ('LEFTPADDING', (0, 0), (-1, -1), 4),
           ('RIGHTPADDING', (0, 0), (-1, -1), 4),
           ('TOPPADDING', (0, 0), (-1, -1), 3.5),
           ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5)]
    if cabecalho:
        sty += [('BACKGROUND', (0, 0), (-1, 0), ESCURO)]
        for i in range(1, len(linhas)):
            if i % 2 == 0:
                sty.append(('BACKGROUND', (0, i), (-1, i), CLARO))
    for i in destaques:
        sty.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#dcfce7')))
    t.setStyle(TableStyle(sty))
    story.append(t)
    sp(7)


def aviso(titulo, texto, cor=AMBAR):
    dados = [[Paragraph(f'<b>{titulo}</b><br/>{texto}',
                        estilo('av', parent=BODY, fontSize=8.8, leading=12.5))]]
    t = Table(dados, colWidths=[W - 2 * MARGIN])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffbeb') if cor == AMBAR
         else colors.HexColor('#fef2f2')),
        ('LINEBEFORE', (0, 0), (0, -1), 3, cor),
        ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    story.append(t)
    sp(8)


L = W - 2 * MARGIN  # largura útil

# ═══════════════════════════════════════════════════════════ CAPA
story.append(Spacer(1, 4.2 * cm))
story.append(Paragraph("Vígil<font color='#16a34a'>.ia</font>",
             estilo('capa', fontName='Helvetica-Bold', fontSize=44,
                    alignment=TA_CENTER, textColor=ESCURO)))
sp(5)
story.append(Paragraph('Inspeção automatizada de grãos de soja por visão computacional',
             estilo('sub', fontName='Helvetica', fontSize=12.5,
                    alignment=TA_CENTER, textColor=CINZA)))
sp(16)
story.append(HRFlowable(width='38%', thickness=1.2, color=VERDE,
                        spaceBefore=4, spaceAfter=15, hAlign='CENTER'))
story.append(Paragraph('Documento de contexto e transferência',
             estilo('doc', fontName='Helvetica-Bold', fontSize=13,
                    alignment=TA_CENTER, textColor=ESCURO)))
sp(6)
story.append(Paragraph('Tudo que existe, por que existe, o que já foi descartado '
                       'e o que falta fazer',
             estilo('doc2', fontName='Helvetica', fontSize=9.5,
                    alignment=TA_CENTER, textColor=CINZA)))
sp(55)
story.append(Paragraph(
    'Este documento foi escrito para uma equipe que vai <b>assumir o projeto</b>. '
    'Ele privilegia o <i>porquê</i> das decisões — inclusive as que deram errado — '
    'porque o repositório acumula quatro gerações de código e boa parte dele é '
    'histórico, não código vivo.',
    estilo('cf', fontName='Helvetica-Oblique', fontSize=9,
           alignment=TA_CENTER, textColor=CINZA)))
sp(30)
story.append(Paragraph('FATEC · Projeto Integrador · Agosto de 2026',
             estilo('meta', fontName='Helvetica', fontSize=9.5,
                    alignment=TA_CENTER, textColor=CINZA)))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════ SUMÁRIO
h1('Sumário')
itens = [
    ('1', 'Leia isto primeiro — como o repositório está organizado'),
    ('2', 'O que o projeto faz'),
    ('3', 'As quatro eras do projeto (e por que isso importa)'),
    ('4', 'Arquitetura: as peças e onde cada uma roda'),
    ('5', 'Histórico de modelos: o que foi testado e o que venceu'),
    ('6', 'O pipeline de dados (como o dataset é construído)'),
    ('7', 'As regras de inferência que fazem o sistema funcionar'),
    ('8', 'RF-DETR no Jetson Orin Nano — a frente mais recente'),
    ('9', 'Estado atual: o que está pronto'),
    ('10', 'O que falta — backlog priorizado'),
    ('11', 'Armadilhas conhecidas (leia antes de debugar)'),
    ('12', 'Mapa do repositório'),
    ('13', 'Como rodar cada peça'),
    ('14', 'Limitações honestas e o que NÃO afirmar'),
]
for n, t in itens:
    story.append(Paragraph(
        f'<font color="#16a34a"><b>{n}.</b></font>&nbsp;&nbsp;{t}',
        estilo('toc', fontName='Helvetica', fontSize=10, textColor=ESCURO, leading=19)))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════ 1
h1('1. Leia isto primeiro — como o repositório está organizado')
p('O repositório tem <b>quatro gerações de código sobrepostas</b>. Arquivos que '
  'parecem contraditórios normalmente são de eras diferentes, não erros. Antes de '
  'mexer em qualquer coisa, saiba em qual camada você está.')
tabela([
    ['Camada', 'Status', 'O que fazer com ela'],
    ['<b>Detecção multi-grão / Jetson</b><br/>(<font face="Courier">jetson/</font>, '
     '<font face="Courier">model/treino_rfdetr_*</font>)',
     '<b>VIVO</b> — frente atual', 'É aqui que o trabalho acontece hoje.'],
    ['<b>App local Steam Deck</b><br/>(<font face="Courier">deck/</font>)',
     '<b>VIVO</b> — funcional',
     'Referência da lógica de voto/veredito. O app do Jetson espelha ele.'],
    ['<b>Web: foto + 1 grão</b><br/>(<font face="Courier">frontend/</font>, '
     '<font face="Courier">backend/</font>)',
     'EM PRODUÇÃO — estável',
     'Está no ar e funciona. Não é o foco atual, mas não quebre.'],
    ['<b>EfficientNet / Gradio</b><br/>(<font face="Courier">model/train.ipynb</font>, '
     '<font face="Courier">gerar_relatorio.py</font>)',
     'HISTÓRICO', 'Só leitura. Não usar como referência técnica.'],
    ['<b>Docs antigos</b><br/>(<font face="Courier">ROADMAP.md</font>, '
     '<font face="Courier">docs/ESPECIFICACAO.md</font>, <font face="Courier">README.md</font>)',
     '<font color="#dc2626"><b>DESATUALIZADO</b></font>',
     'Falam de <font face="Courier">soja_boa</font>, modelos "Sinnet/Magnus", '
     'esteira/NIR. <b>Não reflete o projeto atual.</b>'],
], [5.2 * cm, 3.4 * cm, L - 8.6 * cm])
aviso('Os documentos que valem',
      '<font face="Courier">CLAUDE.md</font> (contexto de desenvolvimento), '
      '<font face="Courier">CONTEXTO_PROJETO.md</font> (estado consolidado) e '
      '<font face="Courier">model/COMPARATIVO_YOLO11S_VS_RTDETR.md</font> (todos os '
      'experimentos de modelo, com os resultados negativos). Este PDF resume os três.')

# ═══════════════════════════════════════════════════════════ 2
h1('2. O que o projeto faz')
p('Classificar grãos de soja por imagem, separando grão bom de grão com defeito. '
  'Existem <b>duas frentes de produto</b>, nascidas em momentos diferentes e que '
  'coexistem:')
bullets([
    '<b>Modo foto (1 grão)</b> — o usuário fotografa um grão e recebe a classe. '
    'Está no ar como aplicação web.',
    '<b>Modo vídeo (multi-grão)</b> — vários grãos no mesmo quadro, ao vivo, com '
    'caixa e classe por grão. É a frente atual, mirando inspeção de lote.',
])
h2('As cinco classes')
tabela([
    ['Índice', 'Rótulo (código)', 'Rótulo (interface)', 'Observação'],
    ['0', 'broken', 'Quebrado', ''],
    ['1', 'immature', 'Imaturo', 'classe mais fraca hoje — ver §10'],
    ['2', 'intact', 'Intacto', '<b>única classe "Premium"</b>'],
    ['3', 'skin-damaged', 'Casca danificada', 'fronteira ambígua com broken'],
    ['4', 'spotted', 'Manchado', 'segunda classe mais fraca'],
], [1.5 * cm, 3.6 * cm, 3.8 * cm, L - 8.9 * cm])
h2('A decisão que o sistema realmente toma')
p('Apesar das cinco classes, a decisão de negócio é <b>binária</b>: '
  '<font face="Courier">intact</font> → <b>Premium</b>; qualquer outra → '
  '<b>expulso da linha</b>. Isso importa na hora de avaliar o modelo — confundir '
  '<font face="Courier">spotted</font> com <font face="Courier">skin-damaged</font> '
  'não muda a decisão; confundir qualquer defeito com '
  '<font face="Courier">intact</font> muda.')
p('O custo é <b>assimétrico</b>: deixar passar um grão ruim é pior que descartar um '
  'grão bom. A regra de voto (§7) foi calibrada com isso em mente.')

# ═══════════════════════════════════════════════════════════ 3
h1('3. As quatro eras do projeto (e por que isso importa)')
p('Entender a sequência evita reabrir decisões já resolvidas e explica por que há '
  'código aparentemente redundante.')
tabela([
    ['Era', 'Stack', 'Como terminou'],
    ['<b>1. EfficientNet + Gradio</b>',
     'TensorFlow/Keras, modelo <font face="Courier">.keras</font>, app Gradio',
     '<b>~29% de acurácia</b> em fotos reais. Domain shift severo — o modelo só '
     'respondia "Quebrado". Arquivada.'],
    ['<b>2. YOLO11s-cls + web</b>',
     'Ultralytics/PyTorch, FastAPI, Next.js, Supabase',
     '<b>91,7%</b> após fine-tuning no domínio real. <b>Está em produção.</b>'],
    ['<b>3. Detecção em vídeo</b>',
     'YOLO11 n/s/l/x e RT-DETR, cenas sintéticas, rastreamento',
     'YOLO11x virou campeão de vídeo. Roda em GPU de servidor.'],
    ['<b>4. RF-DETR para edge</b>',
     'RF-DETR Small, TensorRT, Jetson Orin Nano',
     '<b>Validado: 53 qps no Jetson.</b> Frente atual.'],
], [3.6 * cm, 4.6 * cm, L - 8.2 * cm])
aviso('A lição que atravessa as quatro eras: <i>domain shift</i>',
      'O gargalo nunca foi a arquitetura — foi a diferença entre as imagens de treino '
      'e as de uso real. Na era 1 isso custou ~60 pontos de acurácia (29% vs ~85% no '
      'dataset de laboratório). Toda vez que a câmera, a luz ou o fundo mudam, o '
      'modelo precisa de dado novo. <b>Guarde isso: é a explicação de quase todo '
      'resultado ruim deste projeto.</b>')

# ═══════════════════════════════════════════════════════════ 4
story.append(PageBreak())
h1('4. Arquitetura: as peças e onde cada uma roda')
p('São quatro formas independentes de rodar inferência. Elas compartilham as classes '
  'e a lógica de veredito, mas têm modelos e ambientes diferentes.')
tabela([
    ['Peça', 'Onde roda', 'Modelo', 'Para quê'],
    ['<b>Web — modo foto</b>', 'Hugging Face Spaces (Docker) + Vercel',
     'YOLO11s-cls', 'Produção. Foto de 1 grão → classe + análise por IA.'],
    ['<b>Web — modo ao vivo</b>', '<b>Navegador do usuário</b> (ONNX Runtime Web)',
     'YOLO11s-cls (ONNX)', 'Demo instantânea, sem servidor.'],
    ['<b>App Steam Deck</b>', 'Local, 100% offline',
     'YOLO11n', 'Inspeção ao vivo sem internet. Referência da lógica de voto.'],
    ['<b>Servidor Colab</b>', 'Colab (GPU) + túnel Cloudflare',
     'YOLO11x / RT-DETR', 'Demo dos modelos pesados com câmera de celular.'],
    ['<b>Jetson Orin Nano</b>', 'Embarcado, TensorRT',
     '<b>RF-DETR Small FT4</b>', '<b>Frente atual</b> — caminho para o produto.'],
], [3.3 * cm, 4.3 * cm, 3.3 * cm, L - 10.9 * cm], destaques=[5])
h2('Fluxo do modo web (produção)')
code('foto → POST /inspect (FastAPI no HF Spaces) → YOLO11s-cls classifica\n'
     '     → grava no Supabase → tela de resultado\n'
     '     → (opcional) usuário pergunta → /api/explain → Groq · Llama 3.3')
h2('Fluxo do modo vídeo (Jetson, frente atual)')
code('câmera → letterbox 512×512 → engine TensorRT (RF-DETR)\n'
     '       → NMS agnóstico de classe → rastreamento por IoU\n'
     '       → voto ponderado por grão → veredito travado → desenho na tela')
nota('O celular entra como câmera via DroidCam (URL de rede) ou modo webcam USB. '
     'O Jetson também tem conectores CSI, mas trocar de câmera introduz domain '
     'shift novo — ver §11.')

# ═══════════════════════════════════════════════════════════ 5
h1('5. Histórico de modelos: o que foi testado e o que venceu')
p('Todos os comparativos seguiram o mesmo método: <b>mesmo dado, mesmo pipeline, o '
  'experimento decide</b>. Vários resultados contrariaram a intuição inicial.')

h2('5.1 Classificação (1 grão) — decidido')
tabela([
    ['Modelo', 'Acurácia (domínio real)', 'Gap treino-val'],
    ['EfficientNet-B0 — receita antiga', '64,0%', '25,0% (overfitting)'],
    ['EfficientNet-B0 — em igualdade de condições', '75,0%', '7,2%'],
    ['<b>YOLO11s-cls — em produção</b>', '<b>91,7%</b>', '<b>3,9%</b>'],
], [8.0 * cm, 4.0 * cm, L - 12.0 * cm], destaques=[3])
nota('Ressalva: essa validação usou apenas 12 fotos. 91,7% = 11 de 12. É tendência '
     'forte, não métrica estatística.')

h2('5.2 Detecção em vídeo — a hipótese inicial estava errada')
p('A aposta era que o RT-DETR (transformer) venceria a família YOLO. Depois de '
  'corrigir uma assimetria metodológica — os YOLOs não estavam recebendo o mesmo '
  'estágio base de 12,5 mil imagens — o resultado se inverteu:')
tabela([
    ['Modelo', 'Parâmetros', 'GFLOPs', 'Acurácia premium (visual)'],
    ['YOLO11s (sem estágio base)', '9,4 M', '21,5', '~65-75%'],
    ['YOLO11l', '25,3 M', '86,9', '≈ RT-DETR'],
    ['RT-DETR-l', '32,0 M', '103,5', '~80-85%'],
    ['<b>YOLO11x</b>', '<b>56,9 M</b>', '<b>194,9</b>', '<b>~95% — campeão</b>'],
], [5.6 * cm, 2.6 * cm, 2.2 * cm, L - 10.4 * cm], destaques=[4])
p('<b>Conclusão:</b> o gap era <b>capacidade + caminho de dados</b>, não arquitetura. '
  'A tese "transformer generaliza melhor com dado escasso" vale apenas no regime de '
  'capacidade igual ou menor.')

h2('5.3 RF-DETR para o Jetson — quatro tentativas de fine-tuning')
p('O YOLO11x é modelo de servidor, inviável no Orin Nano. O RF-DETR Small foi testado '
  'como candidato a edge, pelo mesmo caminho de estágios. As quatro variações de '
  'fine-tuning contam uma história útil:')
tabela([
    ['Rodada', 'O que foi treinado', 'Resultado'],
    ['Estágio 1', 'Base 12,5k, pseudo-rótulo Otsu',
     'mAP 0,985 — saturou rápido (tarefa fácil demais)'],
    ['<b>FT1</b>', 'Fotos reais + vídeo de defeito + cenas sintéticas',
     'mAP 0,826, todas as classes saudáveis'],
    ['FT2', 'Só as capturas do <font face="Courier">vigil_deck</font>',
     '<font color="#dc2626"><b>Piorou</b></font> — mAP 0,375, esquecimento '
     'catastrófico. Perdeu até para o FT1 sozinho.'],
    ['FT3', 'Capturas + 30% de replay do FT1',
     'Resolveu o esquecimento. Melhor que FT1 e FT2.'],
    ['<b>FT4</b>', 'Fonte do recorte escolhida <b>por classe</b>',
     '<b>Campeão.</b> Parou de confundir imaturo com intacto.'],
], [2.0 * cm, 6.0 * cm, L - 8.0 * cm], destaques=[5])
h2('O achado mais útil do FT4 (e não é o que parecia)')
p('A hipótese ao montar o FT4 era de <b>qualidade de imagem</b> — cada fonte teria '
  'fotos melhores de classes diferentes. O que aconteceu foi outra coisa: as fotos de '
  '<font face="Courier">immature</font> do FT1 estavam <b>mal rotuladas</b> e ensinavam '
  'que "imaturo se parece com intacto". Trocar a fonte dessa classe não deu imagens '
  'melhores — <b>removeu o contra-exemplo errado</b>.')
p('Ou seja, o mecanismo funcionou como <b>filtro de qualidade de rótulo</b>. É um '
  'contorno, não uma correção: o dado ruim continua na pasta de origem.')

# ═══════════════════════════════════════════════════════════ 6
story.append(PageBreak())
h1('6. O pipeline de dados (como o dataset é construído)')
p('Esta é a parte mais importante para quem for treinar. Quase todo ganho do projeto '
  'veio de mexer <b>aqui</b>, não em hiperparâmetro nem em arquitetura.')
h2('Fontes')
tabela([
    ['Fonte', 'O que é', 'Papel'],
    ['Dataset Roboflow', '12.528 imagens 400×400, licença MIT, 1 grão por foto',
     'Estágio base. Vira detecção via pseudo-rótulo Otsu (fundo preto → caixa '
     'automática, sem anotação manual).'],
    ['Fotos reais', 'Fotos do celular, 1 grão, fundo controlado',
     'Fine-tuning no domínio real. Sem isso a acurácia despenca.'],
    ['Capturas', 'Recortes salvos pelo <font face="Courier">vigil_deck --save-dir</font>',
     'Único dado vindo do setup de uso real. Recortes apertados.'],
    ['Vídeos de defeito', 'Frames de vídeo por classe',
     'Enriquece o FT1. <font face="Courier">intact</font> é excluído de propósito '
     'para manter a avaliação limpa.'],
], [2.9 * cm, 5.2 * cm, L - 8.1 * cm])
h2('Cenas sintéticas multi-grão — o truque central')
p('Fotos de 1 grão não ensinam um detector a lidar com quadro cheio. A solução foi '
  'gerar cenas artificiais:')
bullets([
    'Recortar cada grão da foto real com máscara por saturação, <b>erodida em 1px</b> '
    'para eliminar o halo que inflava a caixa.',
    'Colar de 6 a 25 grãos num fundo escuro gerado, com rotação e sobreposição '
    'controlada, borda nítida (feather pequeno).',
    'Aplicar <i>motion blur</i> em ~40% das cenas, simulando movimento de esteira.',
    'Balancear o <b>pool de recortes</b> por classe — o gerador sorteia uniforme, '
    'então pool torto produz cena torta.',
])
aviso('Resultado negativo importante: NÃO balanceie o treino por classe',
      'Balancear o <i>train</i> (e não só o <i>val</i>) deixou o modelo viciado em '
      'defeito. O vídeo real é majoritariamente <font face="Courier">intact</font>, e '
      'treinar com 20% de cada classe ensina que defeito é comum. '
      '<b>O val deve ser balanceado; o train deve espelhar a realidade.</b>')
h2('Proporção final do dataset campeão (FT4)')
tabela([
    ['Origem', 'Imagens', 'Caixas', '% das caixas'],
    ['Fotos soltas (1 grão)', '1.892', '1.892', '9%'],
    ['<b>Cenas sintéticas (6-25 grãos)</b>', '<b>1.200</b>', '<b>18.477</b>', '<b>91%</b>'],
], [6.2 * cm, 3.0 * cm, 3.0 * cm, L - 12.2 * cm])
p('Essa proporção explica um comportamento que parece defeito e não é: o modelo vai '
  '<b>muito bem em multi-grão e mal em grão solto</b>. Com 91% do treino em cena densa, '
  'e considerando que a família DETR aprende um <i>prior de quantos objetos existem</i>, '
  'um grão sozinho está fora da distribuição. Está alinhado com o uso real.')

# ═══════════════════════════════════════════════════════════ 7
h1('7. As regras de inferência que fazem o sistema funcionar')
p('O modelo sozinho oscila entre quadros. O que torna o sistema utilizável são três '
  'regras aplicadas <b>por cima</b> da detecção. Elas estão em '
  '<font face="Courier">deck/vigil_deck.py</font> e replicadas no app do Jetson.')
h2('1. Voto exigente por classe')
p('Cada grão acumula votos ponderados pela confiança ao longo dos quadros. Um defeito '
  'só é aceito se atingir uma fração mínima dos votos; senão o grão recebe o benefício '
  'da dúvida e vira <font face="Courier">intact</font>.')
tabela([
    ['Classe', 'Fração mínima de votos', 'Leitura'],
    ['broken', '0,85', 'mais exigente — era a que mais alucinava'],
    ['skin-damaged', '0,80', ''],
    ['spotted', '0,75', ''],
    ['immature', '0,75', ''],
    ['intact', '—', 'ganha direto (é o padrão de segurança)'],
], [3.4 * cm, 4.4 * cm, L - 7.8 * cm])
h2('2. Veredito travado por grão')
p('A classe de um grão só é fixada depois de <b>8 quadros</b> rastreando <b>e</b> '
  '<b>3 segundos</b> de observação. Antes disso ele aparece como "analisando...". '
  'Isso mata a troca de rótulo a cada quadro.')
h2('3. Caixa suavizada e anti-piscada')
p('A caixa é suavizada por média móvel (fator 0,4) para não tremer, e detecções que '
  'aparecem em menos de 3 quadros são descartadas como ruído.')
nota('Esses números foram calibrados observando vídeo real. Se o setup de captura '
     'mudar, eles precisam ser recalibrados — mas só depois da padronização (§10).')

# ═══════════════════════════════════════════════════════════ 8
story.append(PageBreak())
h1('8. RF-DETR no Jetson Orin Nano — a frente mais recente')
p('Objetivo: sair da GPU de servidor e rodar embarcado. O RF-DETR é da família DETR, '
  'com backbone DINOv2 e arquitetura definida por busca automática (NAS).')
h2('Por que RF-DETR Small')
p('As variantes Nano a Large têm <b>quase o mesmo número de parâmetros</b> (~30-34 M) — '
  'o que muda entre elas é principalmente a <b>resolução de entrada</b>. O Small (512px) '
  'foi escolhido como meio-termo: mais detalhe que o Nano (384px), o que importa em '
  'objeto pequeno como grão.')
h2('Resultado medido no aparelho')
tabela([
    ['Métrica', 'Valor'],
    ['Throughput', '<b>53,2 qps</b> (TensorRT FP16, modo 15W)'],
    ['GPU Compute (média)', '18,68 ms'],
    ['Engine em disco', '59 MB'],
    ['Veredito', '<b>Cabe em tempo real, com folga</b>'],
], [5.5 * cm, L - 5.5 * cm], destaques=[4])
p('Não foi preciso INT8, nem cair para o RF-DETR Nano, nem usar DeepStream. '
  '<b>Nota de método:</b> a estimativa prévia era 35-40 qps, extrapolada de um AGX Orin '
  'pela razão de TOPS. O medido veio acima — escalar performance por TOPS subestima.')
h2('Como o modelo chega ao Jetson')
code('.pth  (Colab)  →  treina e avalia; exige rfdetr + torch\n'
     '.onnx (ponte)  →  formato neutro, exportado NO COLAB\n'
     '.engine        →  gerado NO PRÓPRIO JETSON, atado ao aparelho\n'
     '                  e à versão do TensorRT')
aviso('A engine não é portátil',
      'Ela é compilada para este hardware e esta versão de TensorRT. Não adianta gerar '
      'no PC e copiar. Se o JetPack for atualizado, reconstrua.')

# ═══════════════════════════════════════════════════════════ 9
h1('9. Estado atual: o que está pronto')
tabela([
    ['Item', 'Status'],
    ['Web modo foto (produção)', '✅ No ar — HF Spaces + Vercel + Supabase'],
    ['Web modo ao vivo (ONNX no navegador)', '✅ Funcional'],
    ['App Steam Deck (offline)', '✅ Funcional, com captura para treino'],
    ['Servidor de demo no Colab + túnel', '✅ Funcional'],
    ['Modelo de vídeo em GPU (YOLO11x)', '✅ Campeão validado'],
    ['<b>RF-DETR Small no Jetson</b>', '<b>✅ 53 qps medidos, app ao vivo rodando</b>'],
    ['Pipeline de treino reproduzível', '✅ Notebook único, resumível por estágio'],
    ['Documentação dos experimentos', '✅ Inclui os resultados negativos'],
], [8.5 * cm, L - 8.5 * cm], destaques=[6])

# ═══════════════════════════════════════════════════════════ 10
h1('10. O que falta — backlog priorizado')
aviso('Prioridade 1 — e ela bloqueia quase tudo: padronizar a captura', mensagem_p1 := (
    'O modelo foi treinado com fotos do celular. O rig do Jetson tem outra câmera, '
    'outra luz e outra distância — ou seja, <b>outro domínio</b>. Qualquer número de '
    'acurácia medido agora não se transfere.<br/><br/>'
    'Enquanto não houver padrão, meça só o que independe de domínio: fps, latência, '
    'estabilidade de caixa, integração. <b>Acurácia, recall por classe e calibragem '
    'dos limiares ficam suspensos.</b><br/><br/>'
    'O padrão precisa fixar: distância, fundo, enquadramento, câmera e '
    '<b>iluminação travada</b> — sem exposição nem balanço de branco automáticos '
    'variando entre sessões, que é o jeito mais fácil de introduzir domain shift '
    'sem perceber.'), cor=VERMELHO)
tabela([
    ['#', 'Tarefa', 'Por quê / observação'],
    ['1', '<b>Definir o padrão de inspeção e captura</b>',
     'Bloqueia todo o resto. Sem isso, dado e métrica nascem descartáveis.'],
    ['2', 'Coletar ~120 grãos únicos de <font face="Courier">immature</font> e '
     '<font face="Courier">spotted</font>',
     '<b>Só depois do item 1.</b> Hoje há 56 e 35. São as duas classes fracas. '
     'Decidido: capturar grão novo, <b>não</b> corrigir rótulo (o erro no FT1 foi '
     'sistemático — as fotos não são de grãos imaturos).'],
    ['3', 'Gravar vídeo de lote propositalmente ruim',
     'O vídeo de teste atual é quase todo <font face="Courier">intact</font>, o que '
     'não distingue "modelo bom" de "modelo viciado em dizer intacto".'],
    ['4', 'Re-treinar no domínio padronizado',
     'Com 1-3 prontos, esta vira a rodada de fine-tuning que realmente conta.'],
    ['5', 'Recalibrar limiares de voto e confiança',
     'Os valores atuais foram calibrados noutro domínio.'],
    ['6', 'Testar RF-DETR Nano (em andamento)',
     'Hipótese: modelo menor sofre menos overfit com dado escasso. Contra-pressão: '
     '384px pode perder detalhe em objeto pequeno.'],
], [0.9 * cm, 5.6 * cm, L - 6.5 * cm])

# ═══════════════════════════════════════════════════════════ 11
story.append(PageBreak())
h1('11. Armadilhas conhecidas (leia antes de debugar)')
p('Esta seção existe porque alguns erros se repetiram várias vezes. Reconhecer o '
  'padrão economiza horas.')

h2('Deslocamento de índice de classe — aconteceu três vezes')
p('O modelo devolve índices numéricos; o código traduz para nomes. Toda vez que a '
  'convenção de indexação foi <b>presumida</b> em vez de verificada, os rótulos '
  'saíram deslocados em 1 — e o sistema continuou funcionando, só que mentindo.')
tabela([
    ['Sintoma observado', 'Causa real'],
    ['Vídeo mostrava <font face="Courier">immature</font> dominando, e '
     '<font face="Courier">spotted</font> nunca aparecia',
     'Índice do modelo era 0-based; o código mapeava por ID do COCO (1-based)'],
    ['App do Jetson dizia <font face="Courier">immature</font> para tudo',
     'A cabeça exporta 6 colunas para 5 classes; assumiu-se que a coluna extra era '
     'fundo <b>na frente</b>, mas o DETR põe o "no-object" <b>no fim</b>'],
], [7.0 * cm, L - 7.0 * cm])
p('<b>Como detectar:</b> se uma classe some completamente do resultado, ou se uma '
  'classe rara domina, suspeite de deslocamento antes de suspeitar do modelo. O '
  '<font face="Courier">jetson/vigil_jetson.py --diag</font> imprime um histograma '
  'das colunas cruas e resolve a dúvida com dado.')

h2('Falha silenciosa em biblioteca')
p('Duas vezes uma dependência mudou de comportamento sem erro: o rastreador '
  '<font face="Courier">ByteTrack</font> do <font face="Courier">supervision</font> foi '
  'depreciado e passou a devolver identificadores vazios — o modelo detectava '
  'normalmente, mas <b>nenhum voto era contado</b>. E o '
  '<font face="Courier">cuda-python</font> 13 reorganizou os módulos, quebrando o '
  'import.')
p('<b>Lição aplicada:</b> o app do Jetson usa um rastreador próprio de ~40 linhas em '
  'vez de depender de biblioteca externa, e checa códigos de retorno da GPU. Onde há '
  'falha silenciosa possível, prefira falhar alto.')

h2('Métrica de validação que mede outra coisa')
p('Numa rodada, o mAP caiu de 0,92 para 0,43 e parecia colapso do modelo. Não era: os '
  'dois estágios validavam em conjuntos <b>diferentes</b>, com distribuições de classe '
  'diferentes. No vídeo real o modelo "pior" era melhor.')
p('<b>Regra prática:</b> só compare números medidos na mesma régua. Entre estágios e '
  'entre versões, <b>o vídeo real é o único termômetro comparável</b>.')

h2('Benchmark em modo de energia reduzido')
p('O Jetson inicia em modo econômico (15W). Medir sem ajustar isso dá número '
  'artificialmente baixo. Igualmente importante: <b>não misture modos</b> ao comparar '
  'modelos — número de 15W não se compara com número de MAXN.')
nota('Variação de ~15% entre execuções do mesmo modelo é normal neste setup '
     '(estado térmico, clocks). Só diferença grande é sinal.')

# ═══════════════════════════════════════════════════════════ 12
story.append(PageBreak())
h1('12. Mapa do repositório')
code(
    '<b>jetson/</b>                    FRENTE ATUAL — embarcado\n'
    '  vigil_jetson.py           app ao vivo (TensorRT + voto + veredito)\n'
    '  bench_trt.sh              constrói a engine e mede\n'
    '  inspect_engine.py         mostra entradas/saídas da engine\n'
    '  comparar_engines.py       compara modelos no mesmo vídeo\n'
    '  check_espaco.sh           diagnóstico de armazenamento\n'
    '  README.md                 guia completo do fluxo\n'
    '\n'
    '<b>deck/</b>                      app local Steam Deck (offline)\n'
    '  vigil_deck.py             REFERÊNCIA da lógica de voto/veredito\n'
    '\n'
    '<b>model/</b>                     treino e experimentos\n'
    '  treino_rfdetr_small_completo.ipynb   pipeline RF-DETR (nano|small)\n'
    '  COMPARATIVO_YOLO11S_VS_RTDETR.md     TODOS os experimentos\n'
    '  PROTOCOLO_ANOTACAO_VIDEO.md          protocolo de captura\n'
    '  aprendizado_ativo.ipynb              correção humana → re-treino\n'
    '  testar_video.py           caixas cruas de um modelo\n'
    '  demo_servidor_colab.ipynb servidor GPU + túnel\n'
    '  train.ipynb, finetune.ipynb          (histórico — EfficientNet)\n'
    '\n'
    '<b>backend/</b>                   FastAPI de produção (modo foto)\n'
    '<b>frontend/</b>                  Next.js (web + modo ao vivo ONNX)\n'
    '\n'
    'CLAUDE.md                   contexto de desenvolvimento\n'
    'CONTEXTO_PROJETO.md         estado consolidado\n'
    'ROADMAP.md, docs/           <font color="#dc2626">DESATUALIZADOS</font>')

# ═══════════════════════════════════════════════════════════ 13
story.append(PageBreak())
h1('13. Como rodar cada peça')
h2('Treinar (Google Colab, GPU)')
code('Abrir model/treino_rfdetr_small_completo.ipynb\n'
     'Ajustar VARIANTE = "nano" ou "small"\n'
     'Executar de cima para baixo. Cada estágio salva no Drive e pula\n'
     'se já existir — dá para dividir em várias sessões.')
h2('Levar o modelo ao Jetson')
code('# no Colab: rodar a célula "Export ONNX"\n'
     '# copiar para o Jetson:\n'
     'scp soja_rfdetr_*_CAMPEAO.onnx bench_trt.sh usuario@IP:~/\n'
     '\n'
     '# no Jetson: construir a engine e medir\n'
     'chmod +x bench_trt.sh\n'
     './bench_trt.sh soja_rfdetr_small_CAMPEAO.onnx fp16')
h2('Rodar a inspeção ao vivo no Jetson')
code('sudo apt install -y python3-pip python3-dev\n'
     'pip3 install cuda-python --break-system-packages\n'
     '\n'
     'python3 vigil_jetson.py                       # celular (DroidCam)\n'
     'python3 vigil_jetson.py --camera 0            # webcam USB\n'
     'python3 vigil_jetson.py --camera csi          # câmera CSI\n'
     'python3 vigil_jetson.py --source video.mp4 --out saida.mp4\n'
     'python3 vigil_jetson.py --diag 100            # diagnóstico de classes\n'
     '\n'
     '# teclas: q sai · c zera contagem · p pausa')
h2('Rodar o app do Steam Deck')
code('python3 -m venv ~/vigilia && source ~/vigilia/bin/activate\n'
     'pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n'
     'pip install ultralytics opencv-python\n'
     'python vigil_deck.py --camera http://IP_DO_CELULAR:4747/video\n'
     'python vigil_deck.py --save-dir capturas   # coleta para treino')

# ═══════════════════════════════════════════════════════════ 14
h1('14. Limitações honestas e o que NÃO afirmar')
p('Esta seção existe para evitar que números sejam repetidos fora de contexto — '
  'inclusive em apresentação ou relatório.')
tabela([
    ['O que se pode dizer', 'O que NÃO se pode dizer'],
    ['O RF-DETR Small roda a 53 qps no Orin Nano (medido, TensorRT FP16, 15W)',
     'Que o sistema "tem 95% de acurácia" — os números de vídeo são <b>avaliação '
     'visual</b>, não benchmark rotulado'],
    ['O modelo vai bem em cena multi-grão no vídeo de teste',
     'Que ele funciona em qualquer lote — o vídeo de teste é quase todo '
     '<font face="Courier">intact</font>'],
    ['Três das cinco classes estão sólidas no val (broken 0,86 · intact 0,81 · '
     'skin-damaged 0,67)',
     'Que as cinco classes funcionam — <font face="Courier">immature</font> tem '
     '<b>recall 0,041</b> e <font face="Courier">spotted</font> AP 0,30'],
    ['O modo foto atingiu 91,7% no domínio real',
     'Que isso é estatisticamente robusto — foram <b>12 fotos</b> de validação'],
], [(L) / 2, (L) / 2])
aviso('A limitação mais importante de todas', mensagem_final := (
    'O modelo <b>não detecta grão imaturo</b> na prática (recall 4%). O vídeo de teste '
    'não contém nenhum grão imaturo, então isso não aparece na avaliação — "não '
    'confundir" e "não detectar" produzem a mesma imagem quando a classe está ausente. '
    '<b>Um lote com grão imaturo passaria batido.</b> Foi uma decisão consciente '
    '(é o melhor ponto para o dado disponível hoje), mas precisa ser revisitada '
    'antes de qualquer uso real.'), cor=VERMELHO)

sp(10)
story.append(HRFlowable(width='100%', thickness=0.6, color=LINHA,
                        spaceBefore=4, spaceAfter=8))
story.append(Paragraph(
    'Vígil.ia — documento de contexto e transferência · Agosto de 2026<br/>'
    'Fontes: CONTEXTO_PROJETO.md, model/COMPARATIVO_YOLO11S_VS_RTDETR.md, '
    'jetson/README.md e o histórico de commits do repositório.',
    estilo('fim', fontName='Helvetica-Oblique', fontSize=8,
           alignment=TA_CENTER, textColor=CINZA)))


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINHA)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.3 * cm, W - MARGIN, 1.3 * cm)
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(CINZA)
    if doc.page > 1:
        canvas.drawString(MARGIN, 0.95 * cm, 'Vígil.ia — contexto e transferência')
        canvas.drawRightString(W - MARGIN, 0.95 * cm, f'{doc.page}')
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                        title='Vígil.ia — Documento de contexto e transferência',
                        author='Vígil.ia')
doc.build(story, onFirstPage=rodape, onLaterPages=rodape)
print('OK ->', OUT)
