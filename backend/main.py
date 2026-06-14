import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

from database import supabase
from inference import decode_base64_image, get_model, run_inference

load_dotenv()

CLASS_PT = {
    "intact": "Intacto",
    "immature": "Imaturo (verde)",
    "broken": "Quebrado",
    "skin-damaged": "Casca danificada",
    "spotted": "Manchado",
}

SUGESTOES: dict[str, list[str]] = {
    "immature": [
        "Como evitar grãos imaturos na colheita?",
        "Qual o impacto comercial do grão imaturo?",
        "Como identificar o ponto certo de colheita?",
    ],
    "broken": [
        "Por que os grãos quebram durante a colheita?",
        "Como reduzir perdas por quebra?",
        "Qual % de quebra é tolerado pela CONAB?",
    ],
    "skin-damaged": [
        "O que causa danos à casca do grão?",
        "Como prevenir danos na colheita e transporte?",
        "Grão com casca danificada pode ser comercializado?",
    ],
    "spotted": [
        "O que causa manchas nos grãos de soja?",
        "Manchas indicam contaminação fúngica?",
        "Como tratar grãos manchados no armazenamento?",
    ],
    "intact": [
        "Como manter os grãos íntegros durante a colheita?",
        "Quais cuidados no armazenamento?",
        "Como a qualidade é avaliada no mercado?",
    ],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield


app = FastAPI(title="Soja Inspection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InspectRequest(BaseModel):
    image: str
    imagem_url: str = ""


class InspectResponse(BaseModel):
    id: str
    total_graos: int
    class_counts: dict
    detections: list
    image_width: int
    image_height: int


class ExplainRequest(BaseModel):
    classe: str
    pergunta: str | None = None
    modo: str = "academico"


class ExplainResponse(BaseModel):
    resposta: str
    sugestoes: list[str]


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
        result = run_inference(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro de inferência: {exc}")

    payload = {
        "imagem_url": body.imagem_url,
        "total_graos": result["total_graos"],
        "resultado_json": result,
    }
    try:
        resp = supabase.table("inspecoes").insert(payload).execute()
        record_id = resp.data[0]["id"]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro no banco de dados: {exc}")

    return InspectResponse(id=record_id, **result)


@app.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY não configurada")

    client = Groq(api_key=groq_key)
    classe_pt = CLASS_PT.get(body.classe, body.classe)

    if body.modo == "academico":
        modo_instrucao = (
            "Use linguagem técnico-científica. Mencione processos fisiológicos, impactos nutricionais "
            "e parâmetros de qualidade segundo normas MAPA. Resposta em até 4 parágrafos."
        )
    else:
        modo_instrucao = (
            "Seja direto e objetivo. Foque em impacto comercial, tolerâncias da Tabela de Classificação "
            "MAPA/CONAB e ações corretivas imediatas no campo ou silo. Resposta em até 3 parágrafos curtos."
        )

    pergunta = body.pergunta or (
        f"Por que o grão de soja está classificado como '{classe_pt}' e quais são as principais causas?"
    )

    system = (
        "Você é um especialista em agronomia com foco em produção e qualidade de grãos de soja. "
        "Responda sempre em português brasileiro. " + modo_instrucao
    )

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Grão classificado como: {classe_pt}. Pergunta: {pergunta}"},
        ],
        max_tokens=512,
        temperature=0.4,
    )

    resposta = completion.choices[0].message.content.strip()
    return ExplainResponse(resposta=resposta, sugestoes=SUGESTOES.get(body.classe, []))


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
