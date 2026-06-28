import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import supabase
from inference import decode_base64_image, get_model, run_inference

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()  # warm-up + download do modelo na inicialização
    yield


app = FastAPI(title="Soja Inspection API", version="1.0.0", lifespan=lifespan)

# /explain (LLM Phi-4-mini) roda na Vercel como API route do Next.js.
# Este backend cuida só de visão (/inspect) + dados.
_origins = [o.strip() for o in os.getenv("CORS_ORIGIN", "*").split(",") if o.strip()]
# Não usamos cookies/credenciais no /inspect. Com credentials=True o wildcard "*"
# vira inválido pelo spec de CORS (o browser bloqueia a resposta mesmo com 200).
# Mantendo credentials=False, o "*" funciona pra qualquer origem da Vercel.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InspectRequest(BaseModel):
    image: str
    imagem_url: str = ""
    multi: bool = False  # padrão atual: 1 grão; True liga o multi-grão
    persist: bool = True  # False = não grava no banco (modo câmera ao vivo)


class InspectResponse(BaseModel):
    id: str
    total_graos: int
    class_counts: dict
    detections: list
    image_width: int
    image_height: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inspect", response_model=InspectResponse)
def inspect(body: InspectRequest):
    try:
        image = decode_base64_image(body.image)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Imagem inválida: {exc}")

    try:
        result = run_inference(image, multi=body.multi)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferência: {exc}")

    # Modo câmera ao vivo (persist=False): não grava — evita flood na tabela.
    if not body.persist:
        return InspectResponse(id="live", **result)

    payload = {
        "imagem_url": body.imagem_url,
        "total_graos": result["total_graos"],
        "resultado_json": result,
    }
    # A gravação é secundária: se o banco estiver fora (ex.: projeto Supabase
    # pausado), NÃO derruba a inspeção — gera um id e segue. O front usa o
    # sessionStorage pra abrir o resultado, então o fluxo continua normal.
    try:
        resp = supabase.table("inspecoes").insert(payload).execute()
        record_id = resp.data[0]["id"]
    except Exception as exc:
        print(f"[warn] falha ao gravar no banco (seguindo sem persistir): {exc}", flush=True)
        record_id = str(uuid.uuid4())

    return InspectResponse(id=record_id, **result)


@app.get("/inspecoes")
def list_inspecoes(limit: int = 50):
    resp = (
        supabase.table("inspecoes")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


@app.get("/lotes")
def list_lotes():
    resp = supabase.table("lotes").select("*").order("data", desc=True).execute()
    return resp.data


class LoteCreate(BaseModel):
    nome: str


@app.post("/lotes", status_code=201)
def create_lote(body: LoteCreate):
    resp = supabase.table("lotes").insert({"nome": body.nome}).execute()
    return resp.data[0]
