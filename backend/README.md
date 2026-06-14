---
title: Soja Inspection API
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Soja Inspection API

Backend de visão da inspeção de grãos de soja: **FastAPI + YOLO11s-cls + OpenCV**.
O frontend (Next.js) e o `/api/explain` (LLM) rodam na Vercel; este Space cuida só
da rota `/inspect` (segmenta os grãos com OpenCV e classifica cada um com o YOLO).

## Endpoints
- `GET /health` — healthcheck
- `POST /inspect` — recebe imagem base64, devolve contagem + classes + bounding boxes
- `GET /inspecoes` / `GET /lotes` / `POST /lotes`

## Variáveis (Settings → Variables and secrets)
| Nome | Tipo | Valor |
|---|---|---|
| `SUPABASE_URL` | secret | URL do projeto Supabase |
| `SUPABASE_KEY` | secret | service_role key |
| `CORS_ORIGIN` | secret | URL da Vercel (ex.: `https://soja-inspection.vercel.app`) |

O modelo `soja_yolo11s_finetuned.pt` já vem embutido na imagem.
