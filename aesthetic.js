/* =====================================================================
   미적 점수 (NIMA, idealo/image-quality-assessment MobileNet aesthetic, Apache 2.0)
   - 사진 평점 데이터(AVA, 약 25만 장, 1~10점)로 학습된 모델이 "사람들이 몇 점을 줄지"의 분포를 낸다
   - 가중치는 vendor/nima/export.py 로 배치 정규화를 접어 넣고 float16으로 바꿨다
   - 구조는 예전 Keras MobileNet v1 그대로: 3x3 합성곱 앞에 대칭 패딩 1, ReLU6, 전역 평균, 10칸 softmax
   - TensorFlow.js(face-api에 들어 있는 것)의 연산으로 직접 돌린다. 사진은 기기 밖으로 나가지 않는다
   ===================================================================== */

// 13개 깊이별 합성곱 블록의 보폭 (MobileNet v1)
const BLOCK_STRIDES = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1];
export const AESTHETIC_INPUT = 224; // 모델 입력 크기 (정사각형)

// float16 비트(Uint16) 하나를 float32 값으로
function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

// manifest(JSON)와 가중치(bin)로 텐서를 만든다. tf는 face-api가 내보내는 TensorFlow.js
export function buildAestheticModel(tf, manifest, weightBuffer) {
  const weights = {}; // 이름(conv1/kernel, dw3/bias 등)에서 텐서로
  for (const entry of manifest.tensors) {
    const bytes = weightBuffer.slice(entry.offset, entry.offset + entry.bytes); // 복사해서 정렬 문제를 피한다
    const values = entry.dtype === 'float32'
      ? new Float32Array(bytes)
      : Float32Array.from(new Uint16Array(bytes), halfToFloat);
    weights[entry.name] = tf.tensor(values, entry.shape, 'float32');
  }
  return { tf, weights };
}

// 224x224 캔버스 한 장의 미적 점수: 평균(1~10)과 표준편차
export async function scoreAesthetic(model, canvas) {
  const { tf, weights } = model;
  const probabilities = tf.tidy(() => {
    const relu6 = (tensor) => tf.clipByValue(tensor, 0, 6);
    // Keras MobileNet 전처리: 0~255를 -1~1로
    let x = tf.browser.fromPixels(canvas).toFloat().div(127.5).sub(1).expandDims(0);
    x = relu6(tf.conv2d(x, weights['conv1/kernel'], 2, 1).add(weights['conv1/bias']));
    BLOCK_STRIDES.forEach((stride, index) => {
      const block = index + 1;
      x = relu6(tf.depthwiseConv2d(x, weights[`dw${block}/kernel`], stride, 1).add(weights[`dw${block}/bias`]));
      x = relu6(tf.conv2d(x, weights[`pw${block}/kernel`], 1, 'valid').add(weights[`pw${block}/bias`]));
    });
    const pooled = x.mean([1, 2]); // 전역 평균 풀링 (1x1024)
    return tf.softmax(pooled.matMul(weights['dense/kernel']).add(weights['dense/bias']));
  });
  const distribution = await probabilities.data(); // 1점부터 10점까지 확률
  probabilities.dispose();
  let mean = 0;
  distribution.forEach((probability, index) => { mean += probability * (index + 1); });
  let variance = 0;
  distribution.forEach((probability, index) => { variance += probability * (index + 1 - mean) ** 2; });
  return { mean, std: Math.sqrt(variance) };
}
