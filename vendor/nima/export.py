# NIMA 가중치를 브라우저용으로 내보낸다
#  - 배치 정규화(BN)를 앞 합성곱에 접어 넣어 (가중치 x 배율, 편향) 한 쌍으로 만든다
#  - 출력 채널마다 uint8로 양자화한다 (채널별 최솟값과 간격을 함께 저장)
#  - 양자화 전후 점수 차이를 확인한다
import sys, json, numpy as np, h5py, importlib.util
from PIL import Image
WEIGHTS, OUT_DIR, IMAGE_PATHS = sys.argv[1], sys.argv[2], sys.argv[3:]
f = h5py.File(WEIGHTS, 'r')
def w(layer, name): return np.array(f[f'{layer}/{layer}/{name}:0'], dtype=np.float32)
def fold(layer_bn):
    gamma, beta, mean, var = (w(layer_bn, k) for k in ('gamma', 'beta', 'moving_mean', 'moving_variance'))
    scale = gamma / np.sqrt(var + 1e-3)
    return scale, beta - mean * scale
tensors = []  # (이름, 배열, 채널 축)
scale, bias = fold('conv1_bn')
tensors += [('conv1/kernel', w('conv1', 'kernel') * scale, 3), ('conv1/bias', bias, None)]
for block in range(1, 14):
    scale, bias = fold(f'conv_dw_{block}_bn')
    tensors += [(f'dw{block}/kernel', w(f'conv_dw_{block}', 'depthwise_kernel') * scale[None, None, :, None], 2), (f'dw{block}/bias', bias, None)]
    scale, bias = fold(f'conv_pw_{block}_bn')
    tensors += [(f'pw{block}/kernel', w(f'conv_pw_{block}', 'kernel') * scale, 3), (f'pw{block}/bias', bias, None)]
tensors += [('dense/kernel', w('dense_1', 'kernel'), 1), ('dense/bias', w('dense_1', 'bias'), None)]

manifest, chunks, offset = [], [], 0
dequantized = {}
for name, array, axis in tensors:
    entry = {'name': name, 'shape': list(array.shape), 'offset': offset}
    if axis is None:  # 편향은 작아서 float32 그대로
        data = array.astype('<f4').tobytes(); entry['dtype'] = 'float32'; dequantized[name] = array
    else:  # 가중치는 float16 (uint8은 점수가 최대 0.29점 흔들려서 쓰지 않는다)
        half = array.astype('<f2')
        data = half.tobytes(); entry['dtype'] = 'float16'; dequantized[name] = half.astype(np.float32)
    entry['bytes'] = len(data); offset += len(data); chunks.append(data); manifest.append(entry)
open(f'{OUT_DIR}/nima-aesthetic.bin', 'wb').write(b''.join(chunks))
json.dump({'format': 'moment-select-nima-1', 'input': 224, 'tensors': manifest}, open(f'{OUT_DIR}/nima-aesthetic.json', 'w'), separators=(',', ':'))
print('bin bytes', offset)

# 접은 가중치(float)와 양자화 가중치로 각각 순전파해서 원래 모델과 비교
spec = importlib.util.spec_from_file_location('ref', 'nima/reference.py'); sys.argv = ['x', WEIGHTS]
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
STRIDES = ref.STRIDES
def forward_folded(image, W):
    x = image.astype(np.float32) / 127.5 - 1
    x = np.clip(ref.conv3x3(x, W['conv1/kernel'], 2) + W['conv1/bias'], 0, 6)
    for block, stride in enumerate(STRIDES, start=1):
        x = np.clip(ref.depthwise(x, W[f'dw{block}/kernel'], stride) + W[f'dw{block}/bias'], 0, 6)
        x = np.clip(x @ W[f'pw{block}/kernel'][0, 0] + W[f'pw{block}/bias'], 0, 6)
    logits = x.mean(axis=(0, 1)) @ W['dense/kernel'] + W['dense/bias']
    p = np.exp(logits - logits.max()); return p / p.sum()
mean_of = lambda p: float((p * np.arange(1, 11)).sum())
floatW = {name: array for name, array, _ in tensors}
for path in IMAGE_PATHS:
    image = np.asarray(Image.open(path).convert('RGB').resize((224, 224), Image.NEAREST))
    print(path.split('/')[-1], 'original %.3f' % mean_of(ref.forward(image)), 'folded %.3f' % mean_of(forward_folded(image, floatW)), 'float16 %.3f' % mean_of(forward_folded(image, dequantized)))
