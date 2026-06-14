# HANDOFF — Deploy do soja-inspection (contexto pra nova sessão)

> Documento de transição. A sessão anterior terminou aqui porque a allowlist de
> egress só passa a valer em **sessão nova**. Este arquivo tem o estado completo.
> Branch de trabalho: **`claude/soja-inspection-setup-b2jaG`**.

---

## Objetivo atual

1. **Validar** as credenciais aqui na sessão nova (egress já liberado — ver abaixo).
2. **Deployar**: frontend + LLM na **Vercel**, backend de visão no **Render**.

A parte de código está **pronta e commitada**. Falta validar e subir.

---

## Arquitetura (2 peças — por quê)

O modelo de visão (`torch`/`ultralytics`) passa de 1GB → **não cabe** no limite
serverless da Vercel (~250MB). Então:

| Parte | Onde | O quê |
|---|---|---|
| Frontend Next.js | **Vercel** | UI responsiva, modos Acadêmico/Industrial |
| `/api/explain` | **Vercel** (API route Next.js) | LLM **Llama 3.3 70b via Groq** (OpenAI-compatible). Não usa Python. |
| `/inspect` | **Render** (Docker) | FastAPI + YOLO11s-cls + OpenCV. Modelo `.pt` embutido na imagem. |
| Banco | **Supabase** | tabelas `inspecoes` + `lotes` |

Fluxo: foto → OpenCV recorta grãos → YOLO11s-cls classifica → tabela + boxes →
usuário clica numa classe → `/api/explain` chama o Llama → resposta agronômica +
botões de follow-up. Toggle **Acadêmico** (técnico-científico) vs **Industrial**
(comercial/CONAB).

---

## O que já foi feito (commits na branch)

- `feat: integra Groq LLM com modos acadêmico e industrial` — ExplainPanel, toggle, tabs por classe
- `feat: deploy Vercel (frontend+LLM) + Render (visão)` — `/api/explain` route, `render.yaml`, download do modelo
- `feat: usa Llama 3.3 70b (Groq)` — `/api/explain` aponta pro Groq; modelo `.pt` (10MB) embutido em `backend/`
- `chore: migração SQL versionada` — `supabase/migrations/0001_init.sql`
- Build da Vercel validado localmente (`tsc --noEmit` + `next build` OK; route `/api/explain` registra como função)

Arquivos-chave:
- `frontend/src/app/api/explain/route.ts` — LLM (Groq/Llama), trocável via `LLM_MODEL`/`LLM_ENDPOINT`
- `frontend/src/components/ExplainPanel.tsx` — UI da resposta + follow-ups
- `frontend/src/app/resultado/[id]/page.tsx` — toggle de modo + tabs por classe
- `backend/inference.py` — OpenCV segmenta + YOLO classifica; usa `.pt` local, fallback HF
- `backend/soja_yolo11s_finetuned.pt` — modelo embutido (10MB, exceção no `.gitignore`)
- `render.yaml` — blueprint Docker, `rootDir: backend`, health `/health`
- `docs/deploy.md` — passo a passo detalhado

---

## Credenciais (projeto Supabase: `btjboljaylsiezpcdqfp`)

> ⚠️ **service_role** e **GROQ_API_KEY** são segredos — NÃO ficam neste arquivo.
> O usuário cola na sessão nova quando pedir. A **anon** é pública (vai pro browser),
> então está inline pra facilitar a validação.

- **SUPABASE_URL:** `https://btjboljaylsiezpcdqfp.supabase.co`
- **anon** (pública, `ref=btjboljaylsiezpcdqfp`, `role=anon`):
  ```
  eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ0amJvbGpheWxzaWV6cGNkcWZwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzkyODQwODYsImV4cCI6MjA5NDg2MDA4Nn0.rN2PQOv6h5SdukY_XwT4Y3aYULR7IqrH9qlyIba9dok
  ```
- **service_role** (`ref=btjboljaylsiezpcdqfp`, `role=service_role`): **o usuário cola** (Settings → API → service_role no projeto btjbol)
- **GROQ_API_KEY** (`gsk_...`): **o usuário cola** (console.groq.com/keys)

Dica de sanidade: todo JWT do Supabase tem `ref` no payload. Sempre confira que
URL + anon + service_role têm o **mesmo** `ref` (`btjboljaylsiezpcdqfp`). Decodificar:
```bash
echo "<jwt>" | cut -d. -f2 | tr '_-' '/+' | sed 's/$/===/' | base64 -d | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['role'],d['ref'])"
```

---

## Egress (rede)

Já configurado como **Custom** com estes hosts adicionados (+ defaults dos package managers):
- `api.groq.com`
- `btjboljaylsiezpcdqfp.supabase.co`

Em sessão nova, `curl` normal já deve sair pra esses hosts (sem precisar contornar sandbox).

---

## PENDENTE 1 — Validar (rodar nesta sessão nova)

Pedir as 2 chaves secretas ao usuário, depois:

```bash
# 1) Supabase anon lê tabelas (confirma conexão + tabelas + RLS)
ANON="<anon inline acima>"
BASE="https://btjboljaylsiezpcdqfp.supabase.co/rest/v1"
curl -sS "$BASE/inspecoes?select=id&limit=1" -H "apikey: $ANON" -H "Authorization: Bearer $ANON" -w "\n[%{http_code}]\n"
curl -sS "$BASE/lotes?select=id&limit=1"     -H "apikey: $ANON" -H "Authorization: Bearer $ANON" -w "\n[%{http_code}]\n"

# 2) Groq + Llama responde
curl -sS https://api.groq.com/openai/v1/chat/completions \
  -H "Authorization: Bearer <GROQ_KEY>" -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","max_tokens":30,"messages":[{"role":"user","content":"responda: ok"}]}' -w "\n[%{http_code}]\n"
```

- `200` na inspecoes/lotes = tabelas existem e RLS deixa anon ler. Se vier erro
  `relation "inspecoes" does not exist`, rodar `supabase/migrations/0001_init.sql`
  no SQL Editor do Supabase (idempotente).
- `200` no Groq = chave + modelo OK.

---

## PENDENTE 2 — Deploy

### Render (backend) — New → Blueprint → repo, branch `claude/soja-inspection-setup-b2jaG`
| Env var | Valor |
|---|---|
| `SUPABASE_URL` | `https://btjboljaylsiezpcdqfp.supabase.co` |
| `SUPABASE_KEY` | **service_role** (btjbol) |
| `CORS_ORIGIN` | (vazio no início; depois a URL da Vercel) |

→ anotar URL pública (ex.: `https://soja-inspection-api.onrender.com`).
Modelo já vem embutido na imagem; sem download no boot.

### Vercel (frontend) — importar repo, **Root Directory = `frontend`**
| Env var | Valor |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://btjboljaylsiezpcdqfp.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | anon (inline acima) |
| `NEXT_PUBLIC_API_URL` | URL do Render |
| `GROQ_API_KEY` | `gsk_...` (sem `NEXT_PUBLIC_` — server-side) |

### Depois
- Voltar no Render e setar `CORS_ORIGIN` = URL final da Vercel.
- Testar: upload de foto → boxes + tabela → clicar numa classe → resposta do Llama.

---

## Observações honestas

- **val do modelo = 12 fotos** (91,7% = 11/12). Erra em casos ambíguos (skin-damaged ↔ broken)
  e tipos de soja fora do treino. Plano: juntar ≥500 imagens antes de re-treinar.
- Plano `free` do Render hiberna (cold start ~30-60s). Pra demo do professor, considerar `starter`.
- MCP do Supabase precisa de aprovação manual do usuário; se aprovado, dá pra criar/verificar
  tabelas e pegar chaves sem mexer no egress.
