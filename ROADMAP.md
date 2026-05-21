# Soja Inspection — Roadmap de Produto

> Sistema de inspeção visual automatizada de grãos de soja,
> do protótipo digital à máquina classificadora industrial.

---

## Escopo 1 — Inspeção por Imagem `[FOCO ATUAL]`

**Objetivo:** entregar um produto funcional que já gera valor real ao cliente.

```
Entrada  →  foto tirada pelo usuário (celular ou câmera)
Modelo   →  IA detecta e classifica cada grão na imagem
Saída    →  bounding boxes coloridos por classe + relatório
Hardware →  nenhum adicional — roda no browser
```

**Classes detectadas:**
| Classe | Descrição |
|---|---|
| `soja_boa` | Grão íntegro, sem defeitos |
| `soja_verde` | Imaturo, colhido cedo |
| `soja_meia_lua` | Formato deformado |
| `soja_ardida` | Manchas por fermentação/calor |
| `soja_quebrada` | Partido ou fragmentado |

**Modelos:**
- **Modelo Leve** — 11-32M parâmetros, otimizado para velocidade, imagens estáticas
- **Modelo Pesado** — ~100M parâmetros, máxima acurácia, imagens HD

**Meta de qualidade:** mAP50 > 0.90

**Entregaveis:**
- [ ] Dataset próprio coletado na fábrica
- [ ] Modelo leve treinado e validado
- [ ] Modelo pesado treinado e validado
- [ ] Comparatívo: dataset próprio vs Roboflow vs transfer learning
- [ ] Interface web com upload + resultado visual + dashboard
- [ ] API REST documentada
- [ ] Deploy Vercel (frontend) + Hugging Face Spaces (modelo)

---

## Escopo 2 — NIR + Escala `[PÓS-VALIDAÇÃO]`

**Objetivo:** detectar defeitos internos invisíveis ao olho humano e à câmera RGB.

```
Entrada  →  câmera RGB + sensor NIR (Near-Infrared 700-2500nm)
Modelo   →  fusão de dados visuais + espectrais
Saída    →  classificação + umidade + proteína + micotoxinas
Hardware →  espectrômetro NIR (~R$3.000-15.000)
```

**O que o NIR detecta que o RGB não vê:**
- Fermentação interna (grão ardido antes de escurecer)
- Umidade excessiva (grão verde por dentro)
- Contaminação fúngica inicial
- Conteúdo de proteína e óleo

**Arquitetura do modelo:**
- Backbone visual (RGB) + encoder espectral (NIR)
- Late fusion: as duas representações combinadas antes da classificação
- ~100M parâmetros totais

**Prerequisito:** Escopo 1 validado e aprovado pelo cliente.

---

## Escopo 3 — Industrial: Esteira + Soprador `[PRODUTO FINAL]`

**Objetivo:** máquina classificadora autônoma equivalente às Bühler Sortex / Satake.

```
Fluxo físico:
  Grãos entram → esteira → câmera line-scan
                          → inferência < 5ms
                          → soprador pneumático ejeta defeituoso
                          → grãos bons seguem

Escala: 3-5 toneladas/hora
```

**Hardware necessário:**
| Componente | Especificação | Custo estimado |
|---|---|---|
| Câmera line-scan | GigE Vision, 4096px | R$8.000-25.000 |
| Computador de borda | NVIDIA Jetson AGX Xavier | R$5.000-8.000 |
| Soprador pneumático | Array de bicos por canal | R$3.000-10.000 |
| Esteira industrial | Customizada | R$5.000-15.000 |
| Sensor NIR (opcional) | Integrado da esteira | R$10.000-30.000 |
| **Total estimado** | | **R$31.000-88.000** |

**Requisito de latência:** < 5ms entre detecção e ativação do soprador.

**Prerequisito:** Escopo 2 validado com acurácia > 0.92 mAP50.

---

## Linha do Tempo

```
Escopo 1  ────────────[██████████]  2 meses (fase atual)
Escopo 2  ────────────────────[██████]  +3-4 meses
Escopo 3  ────────────────────────────[██████████]  +6-12 meses
```

---

## Referências de mercado

- **Bühler Sortex M** — máquina classificadora óptica industrial
- **Satake Scanmaster** — sistema de inspecção de grãos por câmera
- **FOSS NIR** — analisadores espectrais para grãos
- **EMBRAPA** — pesquisas brasileiras em classificação de soja por visão computacional
