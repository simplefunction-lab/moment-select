// 분석용 그림(긴 변 1024px)의 화소 계산: 밝기, 선명도, 노이즈, 노출.
// 화면(app.js)과 디코딩 워커(decode-worker.js)가 같은 코드를 쓴다. 같은 그림에 같은 계산이라 어디서 돌려도 결과가 같다
export const FOCUS_GRID = 4;                     // 초점 부분을 찾으려고 사진을 4x4 칸으로 나눈다
export const FOCUS_TOP_TILES = 3;                // 가장 선명한 3칸 평균을 '초점 맞은 부분 선명도'로 쓴다
export const NOISE_LAPLACIAN_GAIN = 0.40625;      // 노이즈 표준편차 σ가 흐린 라플라시안 분산에 더하는 양은 약 0.406σ² (커널 계수 제곱합)

export function toGrayscale(context, width, height) {
  const rgba = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height); // 픽셀 밝기 0~255
  for (let rgbaIndex = 0, pixelIndex = 0; rgbaIndex < rgba.length; rgbaIndex += 4, pixelIndex++) {
    gray[pixelIndex] = 0.299 * rgba[rgbaIndex] + 0.587 * rgba[rgbaIndex + 1] + 0.114 * rgba[rgbaIndex + 2];
  }
  return gray;
}

// 3x3 가우시안으로 살짝 흐리게: 픽셀 단위 노이즈는 줄고 실제 경계는 남는다
export function gaussianSmooth(gray, width, height) {
  const smooth = new Float32Array(gray.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      smooth[index] = (
        gray[index - width - 1] + 2 * gray[index - width] + gray[index - width + 1] +
        2 * gray[index - 1] + 4 * gray[index] + 2 * gray[index + 1] +
        gray[index + width - 1] + 2 * gray[index + width] + gray[index + width + 1]) / 16;
    }
  }
  return smooth;
}

// 라플라시안 분산: 경계가 또렷할수록 커진다. 흐린 영상에 적용해서 노이즈에 덜 속게 한다
export function laplacianVariance(image, width, height, left = 0, top = 0, right = width, bottom = height) {
  const x0 = Math.max(2, Math.floor(left)), x1 = Math.min(width - 2, Math.ceil(right));
  const y0 = Math.max(2, Math.floor(top)), y1 = Math.min(height - 2, Math.ceil(bottom));
  let sum = 0, sumOfSquares = 0, sampleCount = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const index = y * width + x;
      const laplacian = 4 * image[index] - image[index - 1] - image[index + 1] - image[index - width] - image[index + width];
      sum += laplacian; sumOfSquares += laplacian * laplacian; sampleCount++;
    }
  }
  if (!sampleCount) return 0;
  const mean = sum / sampleCount;
  return sumOfSquares / sampleCount - mean * mean;
}

// 초점 맞은 부분 선명도: 4x4 칸 중 가장 선명한 3칸의 평균. 배경을 일부러 흐리게 한 사진도 손해 보지 않는다
export function focusSharpness(smooth, width, height, noiseFloor) {
  const tileValues = [];
  for (let row = 0; row < FOCUS_GRID; row++) {
    for (let column = 0; column < FOCUS_GRID; column++) {
      tileValues.push(Math.max(0, laplacianVariance(smooth, width, height,
        (column * width) / FOCUS_GRID, (row * height) / FOCUS_GRID, ((column + 1) * width) / FOCUS_GRID, ((row + 1) * height) / FOCUS_GRID) - noiseFloor));
    }
  }
  tileValues.sort((left, right) => right - left);
  return tileValues.slice(0, FOCUS_TOP_TILES).reduce((sum, value) => sum + value, 0) / FOCUS_TOP_TILES;
}

// 노이즈 추정 (Immerkær 1996): 경계 성분을 지우는 3x3 마스크의 절댓값 평균으로 노이즈 표준편차를 추정
export function noiseSigma(gray, width, height) {
  let total = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      total += Math.abs(
        gray[index - width - 1] - 2 * gray[index - width] + gray[index - width + 1]
        - 2 * gray[index - 1] + 4 * gray[index] - 2 * gray[index + 1]
        + gray[index + width - 1] - 2 * gray[index + width] + gray[index + width + 1]);
    }
  }
  return Math.sqrt(Math.PI / 2) * total / (6 * (width - 2) * (height - 2));
}

export function luminanceStats(gray, width, left = 0, top = 0, right = width, bottom = gray.length / width) {
  let total = 0, brightClipped = 0, darkClipped = 0, sampleCount = 0;
  for (let y = Math.floor(top); y < Math.floor(bottom); y++) {
    for (let x = Math.floor(left); x < Math.floor(right); x++) {
      const value = gray[y * width + x];
      total += value; sampleCount++;
      if (value >= 250) brightClipped++;
      else if (value <= 5) darkClipped++;
    }
  }
  return { mean: total / Math.max(1, sampleCount), brightClip: brightClipped / Math.max(1, sampleCount), darkClip: darkClipped / Math.max(1, sampleCount) };
}

// 분석용 그림 한 장의 화소 계산을 한 번에 한다 (얼굴 선명도·밝기에 다시 쓰는 gray, smooth도 돌려준다)
export function measurePixels(context, width, height) {
  const gray = toGrayscale(context, width, height);
  const smooth = gaussianSmooth(gray, width, height);
  const noise = noiseSigma(gray, width, height);
  const noiseFloor = NOISE_LAPLACIAN_GAIN * noise * noise; // 노이즈만으로 생기는 라플라시안 분산
  return { gray, smooth, noise, noiseFloor, light: luminanceStats(gray, width), focusSharpness: focusSharpness(smooth, width, height, noiseFloor) };
}
