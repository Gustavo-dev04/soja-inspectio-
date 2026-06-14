# HANDOFF — Deploy do soja-inspection (contexto pra nova sessão)

> Documento de transição. Branch de trabalho: **`claude/soja-inspection-setup-b2jaG`**.
> Atualizado em **2026-06-14**. Mudança principal vs. versão anterior: o backend de
> visão **deixou de ir pro Render** (pedia cartão no plano `starter`) e vai pro
> **Hugging Face Spaces** (grátis, 16GB RAM, sem cartão).

---

## STATUS (2026-06-14)

- ✅ **Credenciais validadas** nesta máquina:
  - Supabase **anon** → REST `inspecoes`/`lotes` = `200`
  - Supabase **service_role** → REST `200` (`role=service_role`, `ref=btjboljaylsiezpcdqfp`)
  - **Groq + Llama 3.3 70b** (`llama-3.3-70b-versatile`) → `200`, respondeu
  - Egress OK p/ `api.groq.com` e `btjboljaylsiezpcdqfp.supabase.co`
  - MCP do Supabase acessível (projeto `soja-inspection`, ACTIVE_HEALTHY)
- ✅ Backend adaptado pra HF Spaces (Docker) e commitado (`backend/Dockerfile`, `backend/README.md`).
- ⏳ **PENDENTE:** criar o Space e subir; depois Vercel; depois CORS.
- ⚠️ **Segurança:** tabelas `modelos`, `datasets`, `melhorias` estão com **RLS desabilitado**
  (expostas pela anon key). Ver seção de segurança no fim.

---

## Arquitetura (2 peças — por quê)

O modelo de visão (`torch`/`ultralytics`) **não cabe** no limite serverless da Vercel
(~250MB). Então:

| Parte | Onde | O quê |
|---|---|---|
| Frontend Next.js | **Vercel** | UI responsiva, modos Acadêmico/Industrial |
| `/api/explain` | **Vercel** (API route Next.js) | LLM **Llama 3.3 70b via Groq** (OpenAI-compatible). Não usa Python. |
| `/inspect` | **Hugging Face Spaces** (Docker) | FastAPI + YOLO11s-cls + OpenCV. Modelo `.pt` embutido na imagem. |
| Banco | **Supabase** | tabelas `inspecoes` + `lotes` |

Fluxo: foto → OpenCV recorta grãos → YOLO11s-cls classifica → tabela + boxes →
usuário clica numa classe → `/api/explain` chama o Llama → resposta agronômica +
follow-ups. Toggle **Acadêmico** vs **Industrial**.

---

## Arquivos-chave

- `frontend/src/app/api/explain/route.ts` — LLM (Groq/Llama), trocável via `LLM_MODEL`/`LLM_ENDPOINT`
- `frontend/src/components/ExplainPanel.tsx` — UI da resposta + follow-ups
- `frontend/src/app/resultado/[id]/page.tsx` — toggle de modo + tabs por classe
- `backend/main.py` — FastAPI: `/health`, `/inspect`, `/inspecoes`, `/lotes`
- `backend/inference.py` — OpenCV segmenta + YOLO classifica; usa `.pt` local, fallback HF
- `backend/database.py` — exige `SUPABASE_URL` + `SUPABASE_KEY` no ambiente
- `backend/soja_yolo11s_finetuned.pt` — modelo embutido (11MB, exceção no `.gitignore`)
- `backend/Dockerfile` — usuário uid 1000, caches graváveis, porta 7860 (`$PORT` p/ fallback)
- `backend/README.md` — **Space card** (`sdk: docker`, `app_port: 7860`)
- `render.yaml` / `docs/deploy.md` — legado do plano Render (referência)

---

## Credenciais (projeto Supabase: `btjboljaylsiezpcdqfp`)

> ⚠️ **service_role** e **GROQ_API_KEY** são segredos — NÃO ficam neste arquivo.
> O usuário cola na sessão nova quando pedir. A **anon** é pública (vai pro browser),
> então está inline pra facilitar.

- **SUPABASE_URL:** `https://btjboljaylsiezpcdqfp.supabase.co`
- **anon** (pública, `ref=btjboljaylsiezpcdqfp`, `role=anon`):
  ```
  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0amJvbGpheWxzaWV6cGNkcWZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkyODQwODYsImV4cCI6MjA5NDg2MDA4Nn0.rN2PQOv6h5SdukY_XwT4Y3aYULR7IqrH9qlyIba9dok
  ```
- **service_role** (`role=service_role`): **o usuário cola** (Settings → API → service_role)
- **GROQ_API_KEY** (`gsk_...`): **o usuário cola** (console.groq.com/keys)

Sanidade: todo JWT do Supabase tem `ref` no payload; confira que URL + anon + service_role
têm o mesmo `ref`. Decodificar:
```bash
echo "<jwt>" | cut -d. -f2 | tr '_-' '/+' | sed 's/$/===/' | base64 -d | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['role'],d['ref'])"
```

---

## Egress (rede) — PRECISA AJUSTE PRA AUTOMAÇÃO DO HF

Allowlist **Custom** atual:
- `api.groq.com`
- `btjboljaylsiezpcdqfp.supabase.co`

> ➕ Pra eu criar/subir o Space pela API, **adicionar `huggingface.co` à allowlist**
> (mudança de egress só vale em **sessão nova**). Sem isso, `huggingface.co` dá 403.

---

## PENDENTE — Deploy

### 1) Backend no Hugging Face Spaces (automatizável na próxima sessão)

Pré-requisitos do usuário (fazer ANTES da sessão nova):
1. Adicionar `huggingface.co` à allowlist de egress.
2. Criar um **token de escrita** do HF: hf.co/settings/tokens → New token → **Write**.

Conta HF: `Guguinhaxd`. Space novo: **`Guguinhaxd/soja-inspection-api`**
→ URL `https://guguinhaxd-soja-inspection-api.hf.space`.
(O `Guguinhaxd/soja-inspection` que já existe é um demo **Gradio** — NÃO mexer.)

Receita (rodar na sessão nova, com o token e a service_role colados):
```bash
pip install -q huggingface_hub   # não vem instalado na shell
```
```python
from huggingface_hub import HfApi
api = HfApi(token="<HF_WRITE_TOKEN>")
repo_id = "Guguinhaxd/soja-inspection-api"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
api.upload_folder(
    repo_id=repo_id, repo_type="space", folder_path="backend",
    ignore_patterns=["__pycache__", "*.pyc", ".env"],
)
api.add_space_secret(repo_id, "SUPABASE_URL", "https://btjboljaylsiezpcdqfp.supabase.co")
api.add_space_secret(repo_id, "SUPABASE_KEY", "<service_role>")
# CORS_ORIGIN: setar depois da URL da Vercel (api.add_space_secret(repo_id,"CORS_ORIGIN",<url>))
```
O build (Docker, torch+ultralytics) leva alguns minutos. Healthcheck: `GET /health` → `{"status":"ok"}`.

### 2) Frontend na Vercel — Root Directory = `frontend`

Conta Vercel conectada: team `gustavo-dev04's projects`. O MCP `deploy_to_vercel`
NÃO seta Root Directory nem env vars → usar o **import pelo painel**.

| Env var | Valor |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://btjboljaylsiezpcdqfp.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon (inline acima) |
| `NEXT_PUBLIC_API_URL` | `https://guguinhaxd-soja-inspection-api.hf.space` |
| `GROQ_API_KEY` | `gsk_...` (sem `NEXT_PUBLIC_` — server-side) |

### 3) Fechar o CORS
- Setar a secret `CORS_ORIGIN` do Space = URL final da Vercel (re-deploy do Space).
- Testar: upload de foto → boxes + tabela → clicar numa classe → resposta do Llama.

---

## Segurança (RLS) — pendente, decisão do usuário

3 tabelas com **RLS desabilitado** (`modelos`, `datasets`, `melhorias`): com a anon key
(pública) qualquer um lê/escreve. NÃO aplicar cegamente — habilitar RLS sem policy
bloqueia todo acesso. SQL de correção (revisar uso antes):
```sql
ALTER TABLE public.modelos   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.melhorias ENABLE ROW LEVEL SECURITY;
```

---

## Observações honestas

- **val do modelo = 12 fotos** (91,7% = 11/12). Erra em casos ambíguos (skin-damaged ↔ broken).
  Plano: juntar ≥500 imagens antes de re-treinar.
- HF Spaces free dorme após inatividade; acorda na 1ª requisição (cold start enquanto
  carrega o torch). Pra demo do professor, fazer uma chamada de "aquecimento" antes.
- Rotacionar `GROQ_API_KEY` e `service_role` depois do deploy (passaram pelo chat).
