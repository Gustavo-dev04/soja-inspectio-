# Soja Inspection
## Sistema de Inspeção Visual Automatizada de Grãos de Soja

**Instituição:** FATEC — Faculdade de Tecnologia do Estado de São Paulo  
**Disciplina:** Projeto Integrador / Visão Computacional  
**Data:** Maio de 2026  

---

## Resumo

Este projeto implementa um sistema de inspeção visual automatizada de grãos de soja utilizando redes neurais convolucionais (YOLOv8). O sistema detecta e classifica defeitos em imagens de grãos, auxiliando no controle de qualidade agrícola. A arquitetura é composta por um backend em FastAPI, um frontend em Next.js e um modelo YOLOv8 hospedado no Hugging Face Spaces.

**Palavras-chave:** visão computacional, YOLOv8, inspeção de qualidade, soja, FastAPI, Next.js

---

## 1. Introdução

O controle de qualidade de grãos é etapa crítica na cadeia produtiva do agronegócio brasileiro. A identificação manual de defeitos em grãos de soja é um processo lento, sujeito a erros humanos e difícil de escalar. Este projeto propõe a automação dessa inspeção por meio de visão computacional, utilizando o modelo de detecção de objetos YOLOv8.

### 1.1 Objetivos

- **Geral:** Desenvolver um sistema web para inspeção visual automatizada de grãos de soja.
- **Específicos:**
  - Treinar um modelo YOLOv8 para detectar 5 classes de grãos;
  - Implementar uma API REST com FastAPI para realizar a inferência;
  - Desenvolver uma interface web com Next.js para upload e visualização dos resultados;
  - Persistir os resultados no Supabase e exibir estatísticas em um dashboard.

---

## 2. Tecnologias Utilizadas

| Componente | Tecnologia | Versão |
|---|---|---|
| Modelo | YOLOv8 (Ultralytics) | 8.2.0 |
| Backend | FastAPI + Python | 3.11 / 0.111.0 |
| Frontend | Next.js + React + Tailwind CSS | 14.2.3 |
| Banco de Dados | Supabase (PostgreSQL) | — |
| Deploy Frontend | Vercel | — |
| Deploy Modelo | Hugging Face Spaces (Gradio) | 4.36.0 |
| Containerização | Docker + Docker Compose | — |
| Dataset | Roboflow | — |

---

## 3. Arquitetura do Sistema

```
[Usuário]
    │
    ▼
[Frontend Next.js]  ───────────────────────────────► [Supabase]
    │  └─ Página de Upload                                  │  inspecoes
    │  └─ Página de Resultado (bounding boxes)             │  lotes
    │  └─ Dashboard de Lotes (gráficos)  ◄───────────────┘
    │
    ▼  POST /inspect (base64)
[Backend FastAPI]
    │
    ▼
[YOLOv8 Inference]
    │  └─ Detecção de classes + bounding boxes
    │
    ▼
[Supabase]  ──► Persiste resultado_json
```

---

## 4. Classes de Defeito

| Classe | Descrição Agronômica |
|---|---|
| `soja_boa` | Grão íntegro, amadurecido, sem defeitos visíveis |
| `soja_verde` | Grão imaturo, colhido antes da maturação completa |
| `soja_meia_lua` | Grão com formato irregular (deformação de meia-lua) |
| `soja_ardida` | Grão com manchas escuras por fermentação ou calor |
| `soja_quebrada` | Grão partido ou fragmentado |

---

## 5. Configuração e Instalação

### Pré-requisitos

- Docker e Docker Compose
- Node.js 20+
- Python 3.11+

### 5.1 Clone o repositório

```bash
git clone https://github.com/Gustavo-dev04/soja-inspectio-.git
cd soja-inspectio-
```

### 5.2 Configure as variáveis de ambiente

```bash
cp .env.example .env
# Edite .env com suas chaves do Supabase
```

```bash
cd frontend
cp .env.local.example .env.local
# As chaves do Supabase já estão preenchidas no arquivo
```

### 5.3 Execução com Docker Compose

```bash
docker-compose up --build
```

Acesse:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Documentação API: http://localhost:8000/docs

### 5.4 Execução local (sem Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
copy .env.example .env   # Windows
# ou: cp .env.example .env  # Linux/macOS
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 6. Referência da API

### `POST /inspect`

Realiza a inferência YOLOv8 em uma imagem.

**Request:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "imagem_url": ""
}
```

**Response:**
```json
{
  "id": "uuid-da-inspecao",
  "total_graos": 47,
  "class_counts": {
    "soja_boa": 32,
    "soja_verde": 8,
    "soja_ardida": 4,
    "soja_quebrada": 3
  },
  "detections": [
    {
      "class": "soja_boa",
      "class_id": 0,
      "confidence": 0.9231,
      "bbox": [120, 45, 198, 123]
    }
  ],
  "image_width": 640,
  "image_height": 480
}
```

### `GET /inspecoes`

Lista as últimas inspeções salvas.

### `GET /lotes` / `POST /lotes`

Lista e cria lotes de inspeção.

---

## 7. Banco de Dados (Supabase)

**Projeto:** `soja-inspection`  
**URL:** `https://aclabwtsdzzdmpumtjbx.supabase.co`

### Tabela `inspecoes`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | uuid | Chave primária (auto-gerada) |
| `created_at` | timestamptz | Data/hora da inspeção |
| `imagem_url` | text | URL da imagem (opcional) |
| `total_graos` | integer | Total de grãos detectados |
| `resultado_json` | jsonb | Payload completo da inferência |

### Tabela `lotes`

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | uuid | Chave primária (auto-gerada) |
| `nome` | text | Nome do lote |
| `data` | date | Data do lote |
| `total_inspecoes` | integer | Quantidade de inspeções |
| `percentual_defeito` | numeric(5,2) | % de grãos com defeito |

---

## 8. Treinamento do Modelo

### 8.1 Google Colab

```python
# 1. Instalar dependências
!pip install ultralytics roboflow huggingface_hub -q

# 2. Importar e executar
import sys
sys.path.append('/content/soja-inspectio-')
from model.train import download_roboflow_dataset, train, evaluate

# 3. Download do dataset
dataset_path = download_roboflow_dataset(
    api_key="SUA_API_KEY_ROBOFLOW",
    workspace="seu-workspace",
    project="soja-inspection",
    version=1,
)

# 4. Treinar
weights = train(data_yaml=f"{dataset_path}/data.yaml", epochs=100)
evaluate(str(weights))
```

### 8.2 Parâmetros Recomendados

| Parâmetro | Valor | Descrição |
|---|---|---|
| `model_variant` | `yolov8n.pt` | Nano (mais rápido) |
| `epochs` | 100 | Número de épocas |
| `imgsz` | 640 | Tamanho da imagem |
| `batch` | 16 | Tamanho do batch (ajustar por VRAM) |
| `patience` | 20 | Early stopping |

---

## 9. Deploy

### Frontend — Vercel

1. Conecte o repositório na Vercel
2. Configure as variáveis de ambiente:
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `NEXT_PUBLIC_API_URL` (URL do backend deployado)
3. Deploy automático a cada push na branch principal

### Modelo — Hugging Face Spaces

1. Crie um Space do tipo **Gradio** em huggingface.co
2. Faça upload dos arquivos da pasta `model/`
3. Coloque `best.pt` em `weights/best.pt` no Space
4. O `app.py` será detectado automaticamente

---

## 10. Resultados Esperados

Após o treinamento com dataset adequado:

| Métrica | Alvo |
|---|---|
| mAP50 | > 0.85 |
| Precision | > 0.80 |
| Recall | > 0.80 |
| Inferência (GPU) | < 50ms/imagem |

---

## Referências

- Jocher, G. et al. **Ultralytics YOLOv8**. 2023. Disponível em: https://github.com/ultralytics/ultralytics
- **Supabase Documentation**. Disponível em: https://supabase.com/docs
- **Roboflow Documentation**. Disponível em: https://docs.roboflow.com
- **Next.js Documentation**. Disponível em: https://nextjs.org/docs
- Redmon, J.; Farhadi, A. **YOLOv3: An Incremental Improvement**. arXiv, 2018.
- ABNT NBR 14724:2011 — Informação e documentação: trabalhos acadêmicos.
