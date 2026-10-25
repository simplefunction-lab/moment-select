import { measurePixels } from './pixel-math.js';

// 사진 디코딩 워커: 원본을 풀어 분석용(긴 변 1024px) 그림과 표시용 썸네일(긴 변 480px)을 만들고, 분석용 그림의 화소 계산까지 한다.
// 사파리는 디코딩을 부른 스레드에서 바로 하므로, 여기서 하면 화면(메인 스레드)이 얼굴 모델을 돌리는 동안 다음 사진을 함께 풀 수 있다.
// 원본은 읽기만 하고, 줄이는 방식은 메인 스레드의 drawScaled와 같다 (기본 보간, 자주 읽는 캔버스)
function scaledCanvas(source, longSide) {
  const scale = Math.min(1, longSide / Math.max(source.width, source.height));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(source.width * scale)), Math.max(1, Math.round(source.height * scale)));
  canvas.getContext('2d', { willReadFrequently: true }).drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

self.onmessage = async ({ data: { id, file, analysisLongSide, thumbLongSide } }) => {
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    const { width, height } = bitmap;
    const analysis = scaledCanvas(bitmap, analysisLongSide);
    bitmap.close();
    // 선명도·노이즈·노출: 화면이 앞 사진의 얼굴 모델을 돌리는 동안 여기서 미리 계산한다
    const pixels = measurePixels(analysis.getContext('2d', { willReadFrequently: true }), analysis.width, analysis.height);
    const thumb = scaledCanvas(analysis, thumbLongSide);
    const thumbBlob = await thumb.convertToBlob({ type: 'image/jpeg', quality: 0.82 });
    const analysisBitmap = analysis.transferToImageBitmap();
    self.postMessage({ id, width, height, analysisBitmap, thumbBlob, pixels }, [analysisBitmap, pixels.gray.buffer, pixels.smooth.buffer]);
  } catch (error) {
    self.postMessage({ id, error: String(error) });
  }
};
