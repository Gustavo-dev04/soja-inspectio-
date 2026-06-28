// Recorte do grão no navegador — replica o crop_single_grain do treino
// (Otsu -> maior componente -> bounding box + padding). Assim o input ao vivo
// fica parecido com o do treino (grão preenchendo o quadro), em vez de "grão
// pequeno no meio de fundo".
//
// Pressupõe grão MAIS CLARO que o fundo (fundo escuro + luz de cima), como no
// dataset. Sem grão claro o suficiente, devolve null -> o chamador usa fallback.

export type Rect = { sx: number; sy: number; sw: number; sh: number };

export function grainRect(
  source: CanvasImageSource,
  fullW: number,
  fullH: number,
  work: HTMLCanvasElement
): Rect | null {
  const maxDim = 256; // segmentação barata; mapeia de volta pro frame cheio
  const scale = Math.min(1, maxDim / Math.max(fullW, fullH));
  const w = Math.max(1, Math.round(fullW * scale));
  const h = Math.max(1, Math.round(fullH * scale));
  work.width = w;
  work.height = h;
  const ctx = work.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  ctx.drawImage(source, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  const N = w * h;

  // grayscale + histograma
  const gray = new Uint8Array(N);
  const hist = new Int32Array(256);
  for (let i = 0; i < N; i++) {
    const v = ((data[i * 4] * 299 + data[i * 4 + 1] * 587 + data[i * 4 + 2] * 114) / 1000) | 0;
    gray[i] = v;
    hist[v]++;
  }

  // Otsu
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0;
  let wB = 0;
  let maxVar = -1;
  let thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (!wB) continue;
    const wF = N - wB;
    if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > maxVar) {
      maxVar = v;
      thr = t;
    }
  }

  // máscara (grão = mais claro que o fundo)
  const fg = new Uint8Array(N);
  let fgCount = 0;
  for (let i = 0; i < N; i++)
    if (gray[i] > thr) {
      fg[i] = 1;
      fgCount++;
    }
  if (!fgCount) return null;

  // maior componente conectado (BFS, 4-vizinhos) -> bbox + área
  const seen = new Uint8Array(N);
  const stack = new Int32Array(N);
  let best = { area: 0, x0: 0, y0: 0, x1: 0, y1: 0 };
  for (let start = 0; start < N; start++) {
    if (!fg[start] || seen[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    seen[start] = 1;
    let area = 0;
    let x0 = w;
    let y0 = h;
    let x1 = 0;
    let y1 = 0;
    while (sp) {
      const p = stack[--sp];
      const px = p % w;
      const py = (p / w) | 0;
      area++;
      if (px < x0) x0 = px;
      if (px > x1) x1 = px;
      if (py < y0) y0 = py;
      if (py > y1) y1 = py;
      if (px > 0 && fg[p - 1] && !seen[p - 1]) {
        seen[p - 1] = 1;
        stack[sp++] = p - 1;
      }
      if (px < w - 1 && fg[p + 1] && !seen[p + 1]) {
        seen[p + 1] = 1;
        stack[sp++] = p + 1;
      }
      if (py > 0 && fg[p - w] && !seen[p - w]) {
        seen[p - w] = 1;
        stack[sp++] = p - w;
      }
      if (py < h - 1 && fg[p + w] && !seen[p + w]) {
        seen[p + w] = 1;
        stack[sp++] = p + w;
      }
    }
    if (area > best.area) best = { area, x0, y0, x1, y1 };
  }

  const ratio = best.area / N;
  if (ratio <= 0.02 || ratio >= 0.95) return null; // sem grão claro -> fallback

  // mapeia pro frame cheio + padding (igual ao treino)
  const inv = 1 / scale;
  const bx = best.x0 * inv;
  const by = best.y0 * inv;
  const bw = (best.x1 - best.x0 + 1) * inv;
  const bh = (best.y1 - best.y0 + 1) * inv;
  const pad = Math.max(15, 0.12 * Math.min(bw, bh));
  const sx = Math.max(0, bx - pad);
  const sy = Math.max(0, by - pad);
  const ex = Math.min(fullW, bx + bw + pad);
  const ey = Math.min(fullH, by + bh + pad);
  return { sx, sy, sw: ex - sx, sh: ey - sy };
}
