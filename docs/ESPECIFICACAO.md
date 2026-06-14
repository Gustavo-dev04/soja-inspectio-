# Soja Inspection — Documento de Especificação do Produto

**Instituição:** FATEC — Faculdade de Tecnologia do Estado de São Paulo  
**Versão:** 1.0  
**Data:** Maio de 2026  
**Status:** Documento vivo — atualizado conforme o produto evolui

---

## 1. Problema de Negócio

EmpresAs familiares industriais do setor agrícola realizam a classificação de grãos de soja manualmente. Esse processo é lento, sujeito a erro humano e impossível de escalar. O resultado direto é perda de receita: grãos premium classificados erroneamente como matéria-prima, e grãos defeituosos que escapam para a linha de produção.

**Impacto financeiro direto:**  
Cada 1% a mais de grãos premium corretamente identificados por tonelada representa receita adicional que hoje é perdida para imprecisão humana.

---

## 2. Decisão de Negócio por Grão

O sistema classifica cada grão em uma de três saídas de negócio:

| Saída | Classes | Destino |
|---|---|---|
| **Premium** | `soja_boa` (robusta, íntegra) | Linha de produto final de alto valor |
| **Matéria-prima** | `soja_boa` (boa porém não premium) | Processamento industrial |
| **Expulso** | `soja_verde` `soja_quebrada` `soja_ardida` `soja_meia_lua` | Removido da linha de produção |

**Regra crítica:** `soja_ardida` tem tolerância zero. Risco à segurança alimentar.

### Thresholds de confiança padrão (configuráveis pela empresa)

| Classe | Threshold | Motivo |
|---|---|---|
| `soja_ardida` | 0.40 | Tolerância zero — melhor expulsar um bom que deixar um ardido |
| `soja_verde` | 0.55 | Risco de qualidade alto |
| `soja_quebrada` | 0.55 | Compromete apresentação do produto |
| `soja_meia_lua` | 0.60 | Defeito visual, menor risco |

---

## 3. Escopo 1 — Inspeção por Imagem

### Visão
Software acessível via navegador. O operador fotografa o lote de grãos, o sistema retorna a classificação visual e o relatório de qualidade em segundos. Nenhum hardware adicional necessário.

### Fluxo do usuário
```
Operador tira foto do lote
        ↓
Faz upload na interface web
        ↓
Modelo detecta e classifica cada grão
        ↓
 Tela exibe:
  ┌─────────────────────────────────┐
  │ Premium:       342 grãos  68% │
  │ Matéria-prima: 112 grãos  22% │
  │ Expulsos:       48 grãos  10% │
  └─────────────────────────────────┘
  + imagem com bounding boxes coloridos
  + tabela detalhada por classe
        ↓
Resultado salvo automaticamente no histórico
```

### Modelos

**Sinnet** — Modelo Leve  
- Arquitetura: YOLOv8s  
- Parâmetros: ~11 milhões  
- Velocidade: < 8ms por imagem (GPU)  
- GPU de treino: T4 gratuita (Google Colab)  
- Uso: inspecções rápidas, lotes grandes, dispositivos sem GPU dedicada  
- Meta: mAP50 > 0.85  

**Magnus** — Modelo Pesado  
- Arquitetura: RT-DETR-X (Transformer)  
- Parâmetros: ~67–100 milhões  
- Velocidade: < 15ms por imagem (GPU)  
- GPU de treino: A100 (RunPod ~R$15/sessão)  
- Uso: máxima acurácia, imagens HD, base para Escopos 2 e 3  
- Meta: mAP50 > 0.91  

### Estratégia de dados

```
Experimento A → treina com dataset próprio (fábrica)
Experimento B → treina com dataset Roboflow
Experimento C → transfer learning: Roboflow → fine-tune fábrica

Avalia os 3 na câmera real do cliente.
O melhor resultado vira o modelo de produção.
```

### Entregaveis
- [ ] Dataset próprio coletado e anotado (200 grãos físicos + suplemento online)
- [ ] Sinnet treinado e validado (mAP50 > 0.85)
- [ ] Magnus treinado e validado (mAP50 > 0.91)
- [ ] Relatório comparativo dos 6 experimentos (A1/A2/B1/B2/C1/C2)
- [ ] Interface web: upload + resultado visual + dashboard
- [ ] API REST documentada
- [ ] Deploy: Vercel (frontend) + Hugging Face Spaces (modelos)

### Métricas de sucesso
| Métrica | Sinnet | Magnus |
|---|---|---|
| mAP50 | > 0.85 | > 0.91 |
| Precision | > 0.83 | > 0.89 |
| Recall | > 0.81 | > 0.87 |
| Inferência | < 8ms | < 15ms |

---

## 4. Escopo 2 — NIR + Escala

### Visão
Integração de sensor NIR (Near-Infrared, 700–2500nm) à câmera RGB. O sistema passa a detectar defeitos internos invisíveis ao olho humano e à câmera comum: fermentação interna, umidade excessiva, contaminação fúngica inicial.

### O que o NIR detecta que o RGB não vê
| Defeito | Detecção RGB | Detecção NIR |
|---|---|---|
| Ardido externo | ✅ visível | ✅ |
| Ardido interno | ❌ invisível | ✅ detecta pela temperatura espectral |
| Verde externo | ✅ visível | ✅ |
| Umidade interna | ❌ invisível | ✅ detecta por absorção NIR |
| Fungo inicial | ❌ invisível | ✅ detecta antes de ser visível |

### Arquitetura do modelo
```
Entrada RGB  →  backbone visual (Magnus)
                            ↓
                    late fusion
                            ↑
Entrada NIR  →  encoder espectral
                            ↓
                Classificação final
```

### Hardware necessário
| Componente | Especificação | Custo estimado |
|---|---|---|
| Sensor NIR | Espectrômetro 700–2500nm | R$ 3.000–15.000 |
| Câmera industrial | USB3 ou GigE, 5MP+ | R$ 1.500–4.000 |
| Computador | GPU dedicada (RTX 3080+) | R$ 4.000–8.000 |

### Pré-requisito
Escopo 1 validado e aprovado pelo cliente com mAP50 > 0.88.

### Métricas de sucesso
- Detecção de defeitos internos com precisão > 0.85
- Redução de falsos negativos vs Escopo 1 em > 15%
- Integração NIR + RGB sem aumento de latência > 20ms

---

## 5. Escopo 3 — Industrial: Esteira + Soprador

### Visão
Máquina classificadora autônoma. Os grãos entram em fluxo contínuo, a câmera detecta em tempo real e o soprador pneumático ejeta os defeituosos automaticamente. Equivalente às máquinas Bühler Sortex e Satake Scanmaster — construído com os modelos do Escopo 1.

### Fluxo físico
```
Grãos entram
     ↓
  Esteira
     ↓
Câmera line-scan captura
     ↓
Jetson AGX Xavier processa (< 5ms)
     ↓
  Defeituoso?  ── SIM ──►  Soprador ejeta
      ↓ NÃO
  Grão segue para linha
```

### Especificações técnicas
| Parâmetro | Valor |
|---|---|
| Latência máxima detecção → soprador | < 5ms |
| Capacidade de processamento | 3–5 toneladas/hora |
| Resolução da câmera | 4096px (line-scan) |
| Computador de borda | NVIDIA Jetson AGX Xavier |
| Formato de modelo | TensorRT (.engine) com FP16 |

### Hardware necessário
| Componente | Custo estimado |
|---|---|
| Câmera line-scan industrial | R$ 8.000–25.000 |
| NVIDIA Jetson AGX Xavier | R$ 5.000–8.000 |
| Soprador pneumático (array) | R$ 3.000–10.000 |
| Esteira industrial customizada | R$ 5.000–15.000 |
| Estrutura mecânica + integração | R$ 5.000–15.000 |
| **Total estimado** | **R$ 26.000–73.000** |

### Pré-requisito
Escopo 2 validado com acurácia > 0.92 mAP50 em ambiente controlado.

### Métricas de sucesso
- Latência < 5ms (detecção até ativação do soprador)
- Throughput ≥ 3 ton/hora sem degradação de acurácia
- Taxa de erro < 2% (grãos ruins que passam + grãos bons expulsos)

---

## 6. Linha do Tempo

```
AGORA         ESCOPO 1              ESCOPO 2        ESCOPO 3
  │
  ├─ Semana 1-2: fotografar 200 grãos + anotação
  ├─ Semana 3-4: treino Sinnet + Magnus (Exp. A e B)
  ├─ Semana 5:   transfer learning (Exp. C)
  ├─ Semana 6:   comparativo + melhor modelo definido
  ├─ Semana 7:   deploy + dashboard + relatório
  └─ Semana 8:   apresentação Escopo 1 finalizado
                          │
                          ├─ Mês 3-4: aquisição sensor NIR
                          ├─ Mês 4-5: dataset NIR + treinamento fusão
                          └─ Mês 6:   validação Escopo 2
                                              │
                                              ├─ Mês 7-9: hardware esteira
                                              ├─ Mês 10-11: integração Jetson
                                              └─ Mês 12: Escopo 3 operacional
```

---

## 7. Proposta de Valor por Escopo

| Escopo | O que a empresa ganha | Como medir |
|---|---|---|
| 1 | Rastreabilidade + redução de erro humano na classificação | % de retrabalho antes vs depois |
| 2 | Detecção de defeitos invisíveis, menos recall de produto | Devoluções por qualidade |
| 3 | Automação total, 3-5 ton/hora sem operador manual | Custo operacional por tonelada |

---

## 8. Tecnologias por Escopo

| Componente | Escopo 1 | Escopo 2 | Escopo 3 |
|---|---|---|---|
| Modelo | YOLOv8s / RT-DETR-X | RT-DETR + NIR fusion | TensorRT no Jetson |
| Backend | FastAPI (Python) | FastAPI + driver NIR | C++ / CUDA (latência) |
| Frontend | Next.js + Vercel | Next.js + painel NIR | Painel industrial embarcado |
| Banco | Supabase | Supabase + séries temporais | Edge + sync nuvem |
| Deploy | Hugging Face Spaces | GPU dedicada | Jetson AGX Xavier |

---

*Documento mantido em `docs/ESPECIFICACAO.md`. Atualizado a cada evolução do produto.*
