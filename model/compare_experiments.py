"""
Comparação de Experimentos — Sinnet vs Magnus x Dataset

Experimentos:
  A = dataset próprio (fábrica)
  B = dataset Roboflow
  C = transfer learning (Roboflow → fine-tune fábrica)

Gera tabela comparativa de mAP50, Precision, Recall e
velocidade de inferência para todos os 6 experimentos.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class ExperimentResult:
    nome: str
    modelo: str        # sinnet | magnus
    dataset: str       # proprio | roboflow | transfer
    map50: float
    map50_95: float
    precision: float
    recall: float
    inferencia_ms: float


def benchmark_speed(model: YOLO, imgsz: int = 640, n: int = 100) -> float:
    """Média de tempo de inferência em ms para n imagens aleatórias."""
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)  # warm-up

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


def run_experiment(
    nome: str,
    weights: str,
    data_yaml: str,
    modelo: str,
    dataset: str,
) -> ExperimentResult:
    print(f"\n[{nome}] Avaliando...")
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, verbose=False)
    speed = benchmark_speed(model)

    result = ExperimentResult(
        nome=nome,
        modelo=modelo,
        dataset=dataset,
        map50=round(metrics.box.map50, 4),
        map50_95=round(metrics.box.map, 4),
        precision=round(metrics.box.mp, 4),
        recall=round(metrics.box.mr, 4),
        inferencia_ms=round(speed, 2),
    )

    print(f"  mAP50={result.map50}  P={result.precision}  R={result.recall}  "
          f"Speed={result.inferencia_ms}ms")
    return result


def print_table(results: list[ExperimentResult]):
    header = f"{'Experimento':<20} {'Modelo':<10} {'Dataset':<12} "\
             f"{'mAP50':>8} {'P':>8} {'R':>8} {'ms':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(f"{r.nome:<20} {r.modelo:<10} {r.dataset:<12} "
              f"{r.map50:>8.4f} {r.precision:>8.4f} {r.recall:>8.4f} "
              f"{r.inferencia_ms:>8.1f}")
    print("=" * len(header))


if __name__ == "__main__":
    # Defina os caminhos dos pesos treinados em cada experimento
    EXPERIMENTS = [
        # (nome, weights, data_yaml, modelo, dataset)
        ("A1-Sinnet-Proprio",   "weights/sinnet_proprio.pt",   "model/sinnet/data.yaml", "sinnet", "proprio"),
        ("A2-Magnus-Proprio",   "weights/magnus_proprio.pt",   "model/magnus/data.yaml", "magnus", "proprio"),
        ("B1-Sinnet-Roboflow",  "weights/sinnet_roboflow.pt",  "model/sinnet/data.yaml", "sinnet", "roboflow"),
        ("B2-Magnus-Roboflow",  "weights/magnus_roboflow.pt",  "model/magnus/data.yaml", "magnus", "roboflow"),
        ("C1-Sinnet-Transfer",  "weights/sinnet_transfer.pt",  "model/sinnet/data.yaml", "sinnet", "transfer"),
        ("C2-Magnus-Transfer",  "weights/magnus_transfer.pt",  "model/magnus/data.yaml", "magnus", "transfer"),
    ]

    results = []
    for nome, weights, data_yaml, modelo, dataset in EXPERIMENTS:
        if not Path(weights).exists():
            print(f"[SKIP] {nome} — pesos não encontrados em {weights}")
            continue
        results.append(run_experiment(nome, weights, data_yaml, modelo, dataset))

    if results:
        print_table(results)
        with open("experiment_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print("\nResultados salvos em experiment_results.json")
