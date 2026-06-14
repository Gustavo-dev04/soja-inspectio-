# Deploy — Vercel (frontend + LLM) + Render (visão)

Arquitetura de produção em duas peças, porque o modelo de visão (`torch`/`ultralytics`)
**não cabe** no limite de ~250MB das funções serverless da Vercel:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Vercel (Next.js)           │         │  Render / Railway (Docker)   │
│                             │         │                              │
│  - UI responsiva            │ /inspect│  - FastAPI                   │
│  - /api/explain  ───────────┼────────►│  - YOLO11s-cls + OpenCV      │
│    (Phi-4-mini /            │  (HTTP) │  - baixa o .pt do HF Space   │
│     GitHub Models)          │         │                              │
└─────────────────────────────┘         └──────────────────────────────┘
        │                                         │
        └──────────────► Supabase ◄───────────────┘
```

- **`/inspect`** (detecção + classificação) → backend Python no Render.
- **`/api/explain`** (texto do Phi-4-mini) → roda na própria Vercel, sem Python.

---

## 1. Backend de visão no Render

1. Crie um token (se o HF Space for **privado**): https://huggingface.co/settings/tokens (read).
   Se o Space `Guguinhaxd/soja-inspection` for público, pule — não precisa de token.
2. No Render: **New → Blueprint** e aponte para este repositório. O `render.yaml` já
   configura `rootDir: backend`, Docker e health check em `/health`.
3. Preencha as variáveis marcadas `sync: false`:
   - `SUPABASE_URL`, `SUPABASE_KEY` (service-role key)
   - `HF_TOKEN` (só se o Space for privado)
   - `CORS_ORIGIN` → a URL final da Vercel (ex.: `https://soja-inspection.vercel.app`).
     Pode pôr mais de uma separando por vírgula.
4. Deploy. O primeiro boot baixa `soja_yolo11s_finetuned.pt` do HF Space (~10MB) e
   sobe o modelo. Anote a URL pública (ex.: `https://soja-inspection-api.onrender.com`).

> O plano `free` do Render hiberna após inatividade (cold start de ~30-60s). Para a
> demo do professor, considere o `starter` para evitar o atraso na primeira foto.

---

## 2. Token do GitHub Models (Phi-4-mini)

1. https://github.com/settings/tokens → **Generate new token (fine-grained)**.
2. Em **Permissions → Account permissions → Models**, marque **Read**.
3. Copie o token (`github_pat_...`). É grátis e tem limite generoso de requisições.

---

## 3. Frontend na Vercel

1. Importe o repo na Vercel. Em **Root Directory**, selecione **`frontend`**.
2. Em **Environment Variables**, adicione:

   | Variável | Valor |
   |---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | URL do Supabase |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon key |
   | `NEXT_PUBLIC_API_URL` | URL do Render (passo 1) |
   | `GITHUB_MODELS_TOKEN` | token do passo 2 (**sem** `NEXT_PUBLIC_` — fica só no servidor) |

3. Deploy. A Vercel detecta Next.js automaticamente.
4. Volte ao Render e ajuste `CORS_ORIGIN` para a URL final da Vercel.

---

## Variáveis opcionais (Phi-4-mini)

Já têm default, só mexa se precisar trocar de modelo/endpoint:

| Variável | Default |
|---|---|
| `PHI_MODEL` | `microsoft/Phi-4-mini-instruct` |
| `GITHUB_MODELS_ENDPOINT` | `https://models.github.ai/inference/chat/completions` |
