# Deploy — Vercel (frontend + LLM) + Render (visão)

Arquitetura de produção em duas peças, porque o modelo de visão (`torch`/`ultralytics`)
**não cabe** no limite de ~250MB das funções serverless da Vercel:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Vercel (Next.js)           │         │  Render / Railway (Docker)   │
│                             │         │                              │
│  - UI responsiva            │ /inspect│  - FastAPI                   │
│  - /api/explain  ───────────┼────────►│  - YOLO11s-cls + OpenCV      │
│    (Llama 3.3 70b /         │  (HTTP) │  - modelo .pt embutido       │
│     Groq)                   │         │    na imagem Docker          │
└─────────────────────────────┘         └──────────────────────────────┘
        │                                         │
        └──────────────► Supabase ◄───────────────┘
```

- **`/inspect`** (detecção + classificação) → backend Python no Render.
- **`/api/explain`** (texto do Phi-4-mini) → roda na própria Vercel, sem Python.

---

## 1. Backend de visão no Render

1. No Render: **New → Blueprint** e aponte para este repositório. O `render.yaml` já
   configura `rootDir: backend`, Docker e health check em `/health`.
2. Preencha as variáveis marcadas `sync: false`:
   - `SUPABASE_URL`, `SUPABASE_KEY` (service-role key)
   - `CORS_ORIGIN` → a URL final da Vercel (ex.: `https://soja-inspection.vercel.app`).
     Pode pôr mais de uma separando por vírgula.
3. Deploy. O modelo `soja_yolo11s_finetuned.pt` (~10MB) já vem **embutido na imagem**
   (em `backend/`), então não há download no boot. Anote a URL pública
   (ex.: `https://soja-inspection-api.onrender.com`).

> O plano `free` do Render hiberna após inatividade (cold start de ~30-60s). Para a
> demo do professor, considere o `starter` para evitar o atraso na primeira foto.

---

## 2. Chave do Groq (Llama 3.3 70b)

1. https://console.groq.com/keys → **Create API Key**.
2. Copie a chave (`gsk_...`). É grátis e tem limite generoso de requisições.

---

## 3. Frontend na Vercel

1. Importe o repo na Vercel. Em **Root Directory**, selecione **`frontend`**.
2. Em **Environment Variables**, adicione:

   | Variável | Valor |
   |---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | URL do Supabase |
   | `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon key |
   | `NEXT_PUBLIC_API_URL` | URL do Render (passo 1) |
   | `GROQ_API_KEY` | chave do passo 2 (**sem** `NEXT_PUBLIC_` — fica só no servidor) |

3. Deploy. A Vercel detecta Next.js automaticamente.
4. Volte ao Render e ajuste `CORS_ORIGIN` para a URL final da Vercel.

---

## Variáveis opcionais (LLM)

Já têm default, só mexa se precisar trocar de modelo/provedor (a route é
compatível com qualquer API no formato OpenAI):

| Variável | Default |
|---|---|
| `LLM_MODEL` | `llama-3.3-70b-versatile` |
| `LLM_ENDPOINT` | `https://api.groq.com/openai/v1/chat/completions` |
