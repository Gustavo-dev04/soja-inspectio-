"""
Cria o conjunto de HOLDOUT fixo no dataset HF de correções.

O holdout é um split separado (~100 imagens, estratificado por classe) que NUNCA
entra no treino. Ele é a régua fixa que mede a progressão real do modelo entre
versões — sem ele, a métrica de validação flutua a cada rodada e melhorias
ficam impossíveis de comprovar.

Funcionamento:
  1. Lista as imagens por classe no dataset Guguinhaxd/soja-correction
  2. Sorteia (seed=42, determinístico) até 20 por classe → move para holdout/<classe>/
  3. Gera holdout_manifest.json na raiz do dataset com a lista de arquivos
  4. O finetune.ipynb usa holdout/ para avaliar e EXCLUI esses arquivos do treino

Uso:
  HF_TOKEN=hf_xxx python build_holdout.py            # cria (aborta se já existe)
  HF_TOKEN=hf_xxx python build_holdout.py --rebuild  # adiciona novos ao holdout existente

O script é idempotente: rodar duas vezes produz o mesmo holdout (seed fixa,
lista ordenada antes do sorteio). Imagens já em holdout/ nunca voltam ao treino.
"""
import os
import sys
import json
import random
import tempfile
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

REPO_ID    = "Guguinhaxd/soja-correction"
SEED       = 42
PER_CLASS  = 20          # 20 × 5 classes = ~100 imagens no holdout
FRACTION   = 0.20        # se a classe tiver < 100 fotos, usa 20% dela

CLASS_NAMES = [
    "Broken soybeans",
    "Immature soybeans",
    "Intact soybeans",
    "Skin-damaged soybeans",
    "Spotted soybeans",
]

IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    rebuild = "--rebuild" in sys.argv
    token = os.getenv("HF_TOKEN")
    if not token:
        from getpass import getpass
        token = getpass("Cole seu token do Hugging Face (write): ")

    api = HfApi(token=token)
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    manifest_exists = "holdout_manifest.json" in all_files
    if manifest_exists and not rebuild:
        print("holdout_manifest.json já existe — holdout já foi criado.")
        print("Use --rebuild para incorporar fotos novas ao holdout.")
        return

    # Arquivos já no holdout (preservados em qualquer cenário)
    existing_holdout = sorted(
        f for f in all_files
        if f.startswith("holdout/") and f.lower().endswith(IMG_EXTS)
    )

    # Candidatos: imagens nas pastas de classe (fora do holdout)
    rng = random.Random(SEED)
    operations = []
    manifest = {
        "seed": SEED,
        "created": datetime.now(timezone.utc).isoformat(),
        "files": {cls: [] for cls in CLASS_NAMES},
    }

    # Mantém o que já estava no holdout
    for path in existing_holdout:
        parts = path.split("/")            # holdout/<classe>/<arquivo>
        if len(parts) == 3 and parts[1] in manifest["files"]:
            manifest["files"][parts[1]].append(parts[2])

    tmpdir = tempfile.mkdtemp(prefix="holdout_")
    total_moved = 0

    for cls in CLASS_NAMES:
        candidates = sorted(
            f for f in all_files
            if f.startswith(f"{cls}/") and f.lower().endswith(IMG_EXTS)
        )
        already = len(manifest["files"][cls])
        # Quota alvo da classe: 20 fotos OU 20% (o que for menor), descontando
        # o que já está no holdout de rodadas anteriores
        quota = min(PER_CLASS, max(1, int(len(candidates) * FRACTION)))
        need  = max(0, quota - already)
        picked = rng.sample(candidates, min(need, len(candidates)))

        for src in picked:
            name = os.path.basename(src)
            local = hf_hub_download(
                REPO_ID, src, repo_type="dataset", token=token,
                local_dir=tmpdir,
            )
            operations.append(CommitOperationAdd(
                path_in_repo=f"holdout/{cls}/{name}",
                path_or_fileobj=local,
            ))
            operations.append(CommitOperationDelete(path_in_repo=src))
            manifest["files"][cls].append(name)
            total_moved += 1

        print(f"{cls:>24}: {len(candidates):>4} no treino | "
              f"{already:>3} já no holdout | +{len(picked)} movidas")

    # Manifest sempre regravado (mesmo sem movimentação, para criar na 1ª vez)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    operations.append(CommitOperationAdd(
        path_in_repo="holdout_manifest.json",
        path_or_fileobj=manifest_bytes,
    ))

    if total_moved == 0 and manifest_exists:
        print("\nNada novo para mover — holdout já está completo.")
        return

    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"holdout: {total_moved} imagens movidas (seed={SEED})",
    )

    n_total = sum(len(v) for v in manifest["files"].values())
    print(f"\n✅ Holdout criado/atualizado: {n_total} imagens em holdout/")
    print("   holdout_manifest.json gravado na raiz do dataset.")
    print("   Essas imagens NUNCA devem entrar no treino — o finetune.ipynb já as exclui.")


if __name__ == "__main__":
    main()
