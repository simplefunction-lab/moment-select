// 두 사진이 같은 장면의 비슷한 컷인지 판단한다 (v37).
// v36까지는 16x12 흑백을 같은 자리끼리 비교했다. 지중해 사진 111장에서 146쌍을 비슷하다고 봤는데 진짜는 2쌍이었다.
// 시험 사진에서는 진짜 연사 3쌍을 놓쳤다 (조금 움직이거나 확대한 컷).
// v37: 1) 구도: 한 사진을 1.0/1.12/1.25배로 키우고 위치를 ±5칸 옮겨 가장 잘 맞는 자리의 흑백 상관(NCC)을 찾는다.
//      2) 색: 그 자리에서 칸마다 상대 사진의 주변 8칸 중 가장 가까운 색(LAB a,b)과의 차이를 평균한다. 옷 색이 다른 사람을 가린다.
// 기준: 구도 0.80 이상, 색 5 이하. 사진 255장(시험 42, 카메라 60, 지중해 111)의 모든 쌍을 이 코드로 재고 눈으로 확인했다.
//   같은 장면 연사: 색 0.4~1.9. 같은 장소 다른 각도: 색 2.4~7.7. 구도 0.80 이상인 다른 장면: 색 5.6 이상 (일몰끼리 5.6, 바다끼리 7.4)
//   색 5 이하로 두면 다른 장면은 0쌍 통과하고, 같은 장소를 다른 각도로 찍은 4쌍(색 5.3~7.7)은 묶지 않는다
export const SIM_GRAY_W = 64, SIM_GRAY_H = 48;   // 저장하는 흑백 크기
export const SIM_COLOR_W = 32, SIM_COLOR_H = 24; // 저장하는 색(a,b) 크기
export const SIM_STRUCTURE_MIN = 0.80;
export const SIM_COLOR_MAX = 5;
const GRID_W = 48, GRID_H = 36;                  // 정밀하게 비교하는 격자
const QUICK_W = 16, QUICK_H = 12;                // 먼저 대충 맞추는 격자 (흑백 64x48을 4x4칸씩 평균)
const SCALES = [1.0, 1.12, 1.25];
const SHIFT = 5;                                 // 정밀 격자에서 위치를 옮기는 최대 칸 수
const MIN_OVERLAP = 0.75;                        // 옮긴 뒤 겹치는 부분이 가로·세로 각각 이 비율 이상일 때만 본다
const QUICK_MIN = 0.5;                           // 대충 맞춘 구도가 이보다 낮으면 정밀 비교를 하지 않는다 (통과한 쌍의 최솟값 0.536)
const QUICK_TOP = 3;                             // 정밀 비교는 대충 맞춘 결과가 좋은 (방향, 배율) 3개에서만 한다
const RESCUE_MIN = 0.70;                         // 정밀 비교 구도가 0.70~0.80이고 색이 기준 안이면 모든 자리를 다 보고 다시 판단한다

// sRGB 0~255 → LAB a, b
const linear = new Float32Array(256);
for (let value = 0; value < 256; value++) {
  const c = value / 255;
  linear[value] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
const labF = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
function labAB(red, green, blue) {
  const r = linear[red], g = linear[green], b = linear[blue];
  const fx = labF((0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047);
  const fy = labF(0.2126 * r + 0.7152 * g + 0.0722 * b);
  const fz = labF((0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883);
  return [500 * (fx - fy), 200 * (fy - fz)];
}

// 흑백 64x48 RGBA와 색 32x24 RGBA에서 저장할 비교용 값을 만든다 (장당 3072 + 1536 바이트)
export function simSignature(grayRgba, colorRgba) {
  const gray = new Uint8Array(SIM_GRAY_W * SIM_GRAY_H);
  for (let index = 0; index < gray.length; index++) {
    gray[index] = Math.round(0.299 * grayRgba[index * 4] + 0.587 * grayRgba[index * 4 + 1] + 0.114 * grayRgba[index * 4 + 2]);
  }
  const color = new Int8Array(SIM_COLOR_W * SIM_COLOR_H * 2);
  for (let index = 0; index < SIM_COLOR_W * SIM_COLOR_H; index++) {
    const [a, b] = labAB(colorRgba[index * 4], colorRgba[index * 4 + 1], colorRgba[index * 4 + 2]);
    color[index * 2] = Math.max(-128, Math.min(127, Math.round(a)));
    color[index * 2 + 1] = Math.max(-128, Math.min(127, Math.round(b)));
  }
  return { gray, color };
}

// 저장한 작은 그림을 width x height로 다시 샘플한다 (양선형). channels 개 값이 한 칸
function resample(source, sourceW, sourceH, channels, width, height) {
  const out = new Float32Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    const sy = Math.min(sourceH - 1, Math.max(0, (y + 0.5) * sourceH / height - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(sourceH - 1, y0 + 1), wy = sy - y0;
    for (let x = 0; x < width; x++) {
      const sx = Math.min(sourceW - 1, Math.max(0, (x + 0.5) * sourceW / width - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(sourceW - 1, x0 + 1), wx = sx - x0;
      for (let c = 0; c < channels; c++) {
        const top = source[(y0 * sourceW + x0) * channels + c] * (1 - wx) + source[(y0 * sourceW + x1) * channels + c] * wx;
        const bottom = source[(y1 * sourceW + x0) * channels + c] * (1 - wx) + source[(y1 * sourceW + x1) * channels + c] * wx;
        out[(y * width + x) * channels + c] = top * (1 - wy) + bottom * wy;
      }
    }
  }
  return out;
}

// 사진마다 배율별 격자를 한 번만 만든다: 정밀 격자 48x36(흑백, 색)과 대충 격자 16x12(흑백)
const scaledCache = new WeakMap();
function scaled(signature) {
  let levels = scaledCache.get(signature);
  if (!levels) {
    const quickGray = new Float32Array(QUICK_W * QUICK_H);
    for (let y = 0; y < SIM_GRAY_H; y++) {
      for (let x = 0; x < SIM_GRAY_W; x++) quickGray[(y >> 2) * QUICK_W + (x >> 2)] += signature.gray[y * SIM_GRAY_W + x] / 16;
    }
    levels = {
      fine: SCALES.map((scale) => {
        const width = Math.round(GRID_W * scale), height = Math.round(GRID_H * scale);
        return {
          width, height,
          gray: resample(signature.gray, SIM_GRAY_W, SIM_GRAY_H, 1, width, height),
          color: resample(signature.color, SIM_COLOR_W, SIM_COLOR_H, 2, width, height),
        };
      }),
      quick: SCALES.map((scale) => {
        const width = Math.round(QUICK_W * scale), height = Math.round(QUICK_H * scale);
        return { width, height, gray: resample(quickGray, QUICK_W, QUICK_H, 1, width, height) };
      }),
    };
    scaledCache.set(signature, levels);
  }
  return levels;
}

// level(키운 사진)을 (dx, dy)만큼 옮겨 base(1.0배) 위에 놓았을 때 겹치는 범위. 너무 적게 겹치면 null
function overlap(level, base, dx, dy) {
  const cy = ((level.height - base.height) >> 1) + dy, cx = ((level.width - base.width) >> 1) + dx;
  const y0 = Math.max(0, cy), x0 = Math.max(0, cx), y1 = Math.min(level.height, cy + base.height), x1 = Math.min(level.width, cx + base.width);
  if (y1 - y0 < base.height * MIN_OVERLAP || x1 - x0 < base.width * MIN_OVERLAP) return null;
  return { cx, cy, x0, y0, x1, y1 };
}

// 겹친 부분의 흑백 상관 (NCC, -1~1)
function ncc(level, base, area) {
  const { cx, cy, x0, y0, x1, y1 } = area;
  let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
  for (let y = y0; y < y1; y++) {
    const rowA = y * level.width, rowB = (y - cy) * base.width - cx;
    for (let x = x0; x < x1; x++) {
      const a = level.gray[rowA + x], b = base.gray[rowB + x];
      sumA += a; sumB += b; sumAA += a * a; sumBB += b * b; sumAB += a * b;
    }
  }
  const count = (y1 - y0) * (x1 - x0);
  const energyA = sumAA - sumA * sumA / count, energyB = sumBB - sumB * sumB / count;
  return energyA > 0 && energyB > 0 ? (sumAB - sumA * sumB / count) / Math.sqrt(energyA * energyB) : 0;
}

// 겹친 부분에서 칸마다 base의 주변 3x3 칸(가장자리는 가장자리 칸 반복) 중 가장 가까운 색과의 차이 (|Δa|+|Δb|)/2 의 평균
function colorDifference(level, base, area) {
  const { cx, cy, x0, y0, x1, y1 } = area;
  const width = x1 - x0, height = y1 - y0;
  let total = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const indexA = ((y + y0) * level.width + (x + x0)) * 2;
      const a = level.color[indexA], b = level.color[indexA + 1];
      let best = Infinity;
      for (let dy = -1; dy <= 1; dy++) {
        const by = Math.min(height - 1, Math.max(0, y + dy)) + y0 - cy;
        for (let dx = -1; dx <= 1; dx++) {
          const bx = Math.min(width - 1, Math.max(0, x + dx)) + x0 - cx;
          const indexB = (by * base.width + bx) * 2;
          const difference = (Math.abs(a - base.color[indexB]) + Math.abs(b - base.color[indexB + 1])) / 2;
          if (difference < best) best = difference;
        }
      }
      total += best;
    }
  }
  return total / (width * height);
}

// positions 중 구도가 가장 잘 맞는 자리 {structure, area, level, base, dx, dy}
function bestAt(level, base, positions, shiftLimit) {
  let best = { structure: -1 };
  for (const [dx, dy] of positions) {
    if (Math.abs(dx) > shiftLimit || Math.abs(dy) > shiftLimit) continue;
    const area = overlap(level, base, dx, dy);
    if (!area) continue;
    const structure = ncc(level, base, area);
    if (structure > best.structure) best = { structure, area, level, base, dx, dy };
  }
  return best;
}
function around(centerX, centerY, radius) {
  const positions = [];
  for (let dy = centerY - radius; dy <= centerY + radius; dy++) for (let dx = centerX - radius; dx <= centerX + radius; dx++) positions.push([dx, dy]);
  return positions;
}
const QUICK_POSITIONS = around(0, 0, 1);
const ALL_POSITIONS = around(0, 0, SHIFT);

// 두 사진의 {structure, color}. 두 방향(A를 키워 B에 맞추기, B를 키워 A에 맞추기)과 세 배율을 본다.
// 1) 16x12 격자에서 (방향, 배율)마다 ±1칸 중 가장 잘 맞는 자리를 찾는다. 가장 좋은 값이 0.5보다 낮으면 여기서 끝 (색은 무한대).
// 2) 결과가 좋은 (방향, 배율) 3개에서 48x36 격자로 그 자리 주변 ±2칸을 다시 맞추고, 가장 잘 맞는 자리의 구도와 색을 잰다.
// 3) 2)의 구도가 0.70~0.80이고 색이 기준 안이면 여섯 (방향, 배율)의 ±5칸을 모두 본다 (2)가 자리를 잘못 고른 경우를 되찾는다).
// 쌍당 평균 약 0.1ms (데스크톱 크롬). 처음부터 모든 자리를 보면 2.3ms
export function compareSignatures(signatureA, signatureB) {
  const levelsA = scaled(signatureA), levelsB = scaled(signatureB);
  const directions = [[levelsA, levelsB], [levelsB, levelsA]];
  const rough = [];
  for (const [moved, fixed] of directions) {
    moved.quick.forEach((level, scaleIndex) => {
      const found = bestAt(level, fixed.quick[0], QUICK_POSITIONS, Infinity);
      if (found.area) rough.push({ ...found, moved, fixed, scaleIndex });
    });
  }
  rough.sort((left, right) => right.structure - left.structure);
  if (!rough.length || rough[0].structure < QUICK_MIN) return { structure: Math.max(0, rough[0]?.structure ?? 0), color: Infinity };
  const ratio = GRID_W / QUICK_W;
  let best = { structure: -1 };
  for (const guess of rough.slice(0, QUICK_TOP)) {
    const found = bestAt(guess.moved.fine[guess.scaleIndex], guess.fixed.fine[0], around(guess.dx * ratio, guess.dy * ratio, 2), SHIFT);
    if (found.structure > best.structure) best = found;
  }
  if (!best.area) return { structure: rough[0].structure, color: Infinity };
  let color = colorDifference(best.level, best.base, best.area);
  if (best.structure >= RESCUE_MIN && best.structure < SIM_STRUCTURE_MIN && color <= SIM_COLOR_MAX) {
    for (const [moved, fixed] of directions) {
      for (const level of moved.fine) {
        const found = bestAt(level, fixed.fine[0], ALL_POSITIONS, SHIFT);
        if (found.structure > best.structure) best = found;
      }
    }
    color = colorDifference(best.level, best.base, best.area);
  }
  return { structure: best.structure, color };
}

export const isSimilarMatch = (match) => match.structure >= SIM_STRUCTURE_MIN && match.color <= SIM_COLOR_MAX;
