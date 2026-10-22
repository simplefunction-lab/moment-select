/* =====================================================================
   모먼트 셀렉
   - 폴더의 원본을 그대로 두고 File System Access API로 읽고 쓴다
   - 얼굴 검출은 브라우저 안에서 face-api(TensorFlow.js)로 돌린다
   - 점수는 7개 항목의 로지스틱 모델. 처음엔 사진 종류별 기본값, 고르는 동안 사람마다 다시 학습
   ===================================================================== */

// ---------- 라이브러리 위치 (모두 앱과 함께 배포, 외부 요청 없음) ----------
const FACE_API_URL = './vendor/face-api/face-api.esm.js';
const FACE_MODEL_BASE = './vendor/face-api/model/';
const EXIFR_URL = './vendor/exifr/lite.esm.mjs';
// 통계 서버는 이 판에서 쓰지 않는다. 나중에 붙이려면 주소를 넣고 index.html의 CSP connect-src에도 추가해야 한다
const PRIOR_ENDPOINT = '';   // GET ?preset=wedding 에서 {weights:{항목:값}, bias} 를 받아 기본값으로 쓴다
const COLLECT_ENDPOINT = ''; // POST 로 익명 학습 데이터를 보낸다

// ---------- 실행 환경 ----------
const canPickFolder = 'showDirectoryPicker' in window;      // 폴더를 직접 읽고 쓸 수 있는 브라우저 (데스크톱 크롬, 엣지 등)
const isTouchDevice = matchMedia('(pointer: coarse)').matches; // 휴대폰, 태블릿
const canShareFiles = Boolean(navigator.canShare?.({ files: [new File([''], 'probe.jpg', { type: 'image/jpeg' })] })); // 공유 시트로 파일 저장 가능
const SHARE_BATCH_SIZE = 10; // 공유 시트 한 번에 넘길 장수

// ---------- 판정 기준값 ----------
const SUPPORTED_IMAGE = /\.(jpe?g|png|webp)$/i;
const UNSUPPORTED_IMAGE = /\.(cr2|cr3|nef|arw|raf|dng|orf|rw2|heic|heif)$/i;
const EXPORT_FOLDER_PREFIX = '모먼트셀렉_';
const ANALYSIS_LONG_SIDE = 1024;          // 분석용으로 줄일 긴 변
const THUMB_LONG_SIDE = 480;              // 썸네일 긴 변
const MAIN_FACE_MIN_HEIGHT = 0.035;       // 사진 높이 대비 이보다 작은 얼굴은 배경 인물
const EYE_CLOSED_EAR = 0.21;              // 두 눈 모두 이보다 작으면 눈 감음 의심
const EYE_CHECK_MIN_FACE_PX = 60;         // 얼굴이 이보다 작으면 눈을 판정하지 않음
const EYE_CHECK_MAX_YAW = 0.22;           // 고개가 이보다 돌아가 있으면 눈을 판정하지 않음 (옆얼굴)
const EYE_CHECK_MIN_CONFIDENCE = 0.6;     // 얼굴 검출 확신이 이보다 낮으면 눈을 판정하지 않음
const FOCUS_GRID = 4;                     // 초점 부분을 찾으려고 사진을 4x4 칸으로 나눈다
const FOCUS_TOP_TILES = 3;                // 가장 선명한 3칸 평균을 '초점 맞은 부분 선명도'로 쓴다
const SIMILAR_HASH_DISTANCE = 12;
const SIMILAR_TIME_GAP_MS = 10_000;
const GROUP_MAX_SIZE = 12;
const RECOMMEND_RATIO = 0.4;              // 점수 상위 40% 단위를 먼저 보여준다
const SCORE_SCALE = 8;                    // 가중합(0~1)을 로지스틱 모델로 바꿀 때의 기울기
const LEARN_MIN_DECISIONS = 20;           // 이만큼 고르면 학습을 시작
const LEARN_MIN_PER_CLASS = 5;            // 남김과 뺌이 각각 이만큼은 있어야 학습
const LEARN_EVERY = 10;                   // 새 결정이 이만큼 쌓일 때마다 다시 학습
const CACHE_VERSION = 'v3';
const NOISE_LAPLACIAN_GAIN = 0.40625;      // 노이즈 표준편차 σ가 흐린 라플라시안 분산에 더하는 양은 약 0.406σ² (커널 계수 제곱합)

// ---------- 점수 항목과 사진 종류별 기본 가중치 ----------
// 가중치는 출발값일 뿐이며 검증된 값이 아니다. 고르는 동안 사람마다 학습해서 바뀐다
const FEATURE_KEYS = ['subjectSharp', 'focusSharp', 'exposure', 'expression', 'smile', 'framing', 'clean'];
const FEATURE_LABELS = {
  subjectSharp: '인물 선명도',
  focusSharp: '초점 부분 선명도',
  exposure: '노출',
  expression: '찡그림 없음',
  smile: '웃음',
  framing: '얼굴 안 잘림',
  clean: '노이즈 적음',
};
const PRESETS = {
  wedding: { label: '웨딩과 스냅', hint: '차분한 포즈가 많아서 웃음보다 선명도와 찡그림 없음을 크게 봐요. 눈 감은 사진은 합성 요청 후보로 표시만 해요.', crowdMin: 5,
    weights: { subjectSharp: .34, focusSharp: .10, exposure: .20, expression: .16, smile: .05, framing: .10, clean: .05 } },
  travel: { label: '여행', hint: '풍경 비중이 커서 사진 전체의 초점과 노출을 크게 봐요.', crowdMin: 5,
    weights: { subjectSharp: .24, focusSharp: .22, exposure: .22, expression: .10, smile: .08, framing: .06, clean: .08 } },
  event: { label: '행사와 모임', hint: '단체 사진이 많아서 얼굴이 다 보이는지와 웃음을 크게 봐요.', crowdMin: 6,
    weights: { subjectSharp: .30, focusSharp: .08, exposure: .17, expression: .15, smile: .18, framing: .07, clean: .05 } },
  family: { label: '아이와 가족', hint: '움직임이 많아서 인물 선명도를 가장 크게 보고 웃음도 봐요.', crowdMin: 5,
    weights: { subjectSharp: .36, focusSharp: .08, exposure: .16, expression: .12, smile: .18, framing: .06, clean: .04 } },
  other: { label: '기타', hint: '항목을 고르게 봐요.', crowdMin: 5,
    weights: { subjectSharp: .28, focusSharp: .18, exposure: .20, expression: .12, smile: .08, framing: .07, clean: .07 } },
};
// 가중합 w·x 를 로지스틱 모델 sigmoid(bias + θ·x) 로 옮긴다. 순서는 그대로 유지된다
function modelFromWeights(weights, source) {
  return { theta: FEATURE_KEYS.map((key) => SCORE_SCALE * (weights[key] ?? 0)), bias: -SCORE_SCALE * 0.5, source, trainedOn: 0 };
}

// ---------- 전체 상태 ----------
const state = {
  folderName: '',
  photos: [],                    // 사진 목록 (분석 결과 포함)
  photoById: new Map(),          // 사진 id(폴더 안 상대 경로)로 찾기
  groups: new Map(),             // 묶음 id에서 사진 id 배열로
  units: [],                     // 1차 셀렉 단위 (낱장 또는 묶음)
  expandedGroups: new Set(),     // '한 장씩 보기'로 푼 묶음
  settings: { preset: 'wedding', target: 30, useCategories: true, crowdMin: 5, useScore: true, share: false },
  reviewers: { a: { name: '나', decisions: {} }, b: { name: '파트너', decisions: {} } }, // decisions: 사진 id에서 {status, at, addedAt, log}
  currentReviewer: 'a',
  priorModel: null,              // 사진 종류 기본값 또는 서버에서 받은 기본 모델
  models: { a: null, b: null },  // 사람마다 학습된 모델 (없으면 priorModel)
  rounds: { a: { distill: [], rescue: [] }, b: { distill: [], rescue: [] } }, // 이번 라운드에서 본 사진
  dividerSeen: { a: false, b: false },
  sentSampleCount: { a: 0, b: 0 }, // 서버로 이미 보낸 결정 수
  view: 'setup',
  categoryFilter: 'all',
  compareMode: false,
  undoStack: [],
  priorityIds: [],               // 되돌리거나 필름 띠에서 누른 사진을 맨 앞으로
  skippedFileCount: 0,
  analysisDone: false,
  skipFaceAnalysis: false,
};

// ---------- 작은 도우미 ----------
const $ = (selector, root = document) => root.querySelector(selector);
const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const reviewer = () => state.reviewers[state.currentReviewer];
const decisionOf = (photoId, reviewerKey = state.currentReviewer) => state.reviewers[reviewerKey].decisions[photoId];
const statusOf = (photoId, reviewerKey) => decisionOf(photoId, reviewerKey)?.status ?? null;
const clamp01 = (value) => Math.max(0, Math.min(1, value));
const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const preset = () => PRESETS[state.settings.preset];

function formatKoreanTime(ms) {
  const date = new Date(ms);
  return `${date.getMonth() + 1}월 ${date.getDate()}일 ${date.getHours()}시 ${String(date.getMinutes()).padStart(2, '0')}분`;
}

function showToast(html, actions = [], durationMs = 4000) {
  const toast = $('#toast');
  toast.innerHTML = `<span>${html}</span>`;
  for (const action of actions) {
    const button = document.createElement('button');
    button.className = 'button';
    button.textContent = action.label;
    button.onclick = () => { toast.classList.add('hidden'); action.run(); };
    toast.appendChild(button);
  }
  toast.classList.remove('hidden');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.add('hidden'), durationMs);
}

function cloneMark(kind) {
  return $(kind === 'keep' ? '#mark-template' : '#cross-template').content.firstElementChild.cloneNode(true);
}

/* =====================================================================
   저장 (IndexedDB)
   ===================================================================== */
const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open('moment-select', 1);
  request.onupgradeneeded = () => request.result.createObjectStore('kv');
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});
async function dbGet(key) {
  const db = await dbPromise;
  return new Promise((resolve) => {
    const request = db.transaction('kv').objectStore('kv').get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(undefined);
  });
}
async function dbSet(key, value) {
  const db = await dbPromise;
  return new Promise((resolve) => {
    const transaction = db.transaction('kv', 'readwrite');
    transaction.objectStore('kv').put(value, key);
    transaction.oncomplete = () => resolve(true);
    transaction.onerror = () => resolve(false);
  });
}
let saveTimer = null;
function saveProjectSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    if (!state.folderName) return;
    dbSet(`project:${state.folderName}`, {
      settings: state.settings, reviewers: state.reviewers, models: state.models, rounds: state.rounds,
      dividerSeen: state.dividerSeen, sentSampleCount: state.sentSampleCount, expandedGroups: [...state.expandedGroups],
    });
  }, 300);
}

/* =====================================================================
   시작 화면
   ===================================================================== */
function renderPresetOptions() {
  $('#preset-options').innerHTML = Object.entries(PRESETS).map(([key, value]) =>
    `<label><input type="radio" name="preset" value="${key}" ${key === state.settings.preset ? 'checked' : ''}>${value.label}</label>`).join('');
  $('#preset-hint').textContent = preset().hint;
}
$('#preset-options').onchange = (event) => {
  state.settings.preset = event.target.value;
  $('#crowd-input').value = preset().crowdMin;
  renderPresetOptions();
};
$('#category-input').onchange = (event) => $('#crowd-row').classList.toggle('hidden', !event.target.checked);
if (COLLECT_ENDPOINT) $('#share-field').classList.remove('hidden');
renderPresetOptions();

function applySettingsFromForm() {
  state.settings.target = Math.max(1, parseInt($('#target-input').value, 10) || 30);
  state.settings.useCategories = $('#category-input').checked;
  state.settings.crowdMin = Math.max(3, parseInt($('#crowd-input').value, 10) || 5);
  state.settings.useScore = $('#score-input').checked;
  state.settings.share = Boolean(COLLECT_ENDPOINT) && $('#share-input').checked;
  state.reviewers.a.name = $('#name-a').value.trim() || '나';
  state.reviewers.b.name = $('#name-b').value.trim() || '파트너';
}

/* =====================================================================
   1. 폴더 읽기
   ===================================================================== */
async function collectImageEntries(directoryHandle, pathPrefix = '') {
  const imageEntries = []; // {relPath, name, handle}
  for await (const [entryName, entryHandle] of directoryHandle.entries()) {
    if (entryName.startsWith('.')) continue;
    const relPath = pathPrefix ? `${pathPrefix}/${entryName}` : entryName;
    if (entryHandle.kind === 'directory') {
      if (entryName.startsWith(EXPORT_FOLDER_PREFIX)) continue; // 지난번에 내보낸 폴더는 다시 읽지 않음
      imageEntries.push(...await collectImageEntries(entryHandle, relPath));
    } else if (SUPPORTED_IMAGE.test(entryName)) {
      imageEntries.push({ relPath, name: entryName, handle: entryHandle });
    } else if (UNSUPPORTED_IMAGE.test(entryName)) {
      state.skippedFileCount++;
    }
  }
  return imageEntries;
}
const getPhotoFile = (photo) => (photo.handle ? photo.handle.getFile() : Promise.resolve(photo.file));

async function loadPriorModel() {
  const fallback = modelFromWeights(preset().weights, 'preset');
  if (!PRIOR_ENDPOINT) return fallback;
  try {
    const response = await fetch(`${PRIOR_ENDPOINT}?preset=${encodeURIComponent(state.settings.preset)}`);
    const body = await response.json();
    const theta = FEATURE_KEYS.map((key) => Number(body.weights?.[key]));
    if (theta.some((value) => !Number.isFinite(value)) || !Number.isFinite(Number(body.bias))) return fallback;
    return { theta, bias: Number(body.bias), source: 'server', trainedOn: Number(body.sampleCount) || 0 };
  } catch {
    return fallback;
  }
}

async function startWithEntries(folderName, imageEntries) {
  applySettingsFromForm();
  state.folderName = folderName;
  const savedProject = await dbGet(`project:${folderName}`);
  if (savedProject && savedProject.settings?.preset === state.settings.preset) {
    // 같은 폴더, 같은 사진 종류로 다시 열면 결정과 학습 결과를 이어서 쓴다
    state.reviewers.a.decisions = savedProject.reviewers?.a?.decisions ?? {};
    state.reviewers.b.decisions = savedProject.reviewers?.b?.decisions ?? {};
    state.models = savedProject.models ?? state.models;
    state.rounds = savedProject.rounds ?? state.rounds;
    state.dividerSeen = savedProject.dividerSeen ?? state.dividerSeen;
    state.sentSampleCount = savedProject.sentSampleCount ?? state.sentSampleCount;
    state.expandedGroups = new Set(savedProject.expandedGroups ?? []);
  }
  state.priorModel = await loadPriorModel();
  state.photos = imageEntries.map((entry) => ({
    id: entry.relPath, relPath: entry.relPath, name: entry.name,
    handle: entry.handle ?? null, file: entry.file ?? null,
    thumbUrl: '', takenAt: null, width: 0, height: 0,
    analysis: null,       // 분석 원값
    features: null,       // 점수 항목 7개 (0~1)
    score: 0,             // 0~100
    groupId: null, groupBest: false, recommended: false,
  }));
  state.photos.sort((left, right) => left.relPath.localeCompare(right.relPath, undefined, { numeric: true }));
  state.photoById = new Map(state.photos.map((photo) => [photo.id, photo]));
  if (!state.photos.length) { showToast('읽을 수 있는 사진(JPG, PNG, WEBP)을 찾지 못했어요. 다른 폴더나 사진을 골라주세요.'); return; }
  setView('analyze');
  await analyzeAll();
}

$('#pick-folder').onclick = async () => {
  if (!canPickFolder) { $('#fallback-input').click(); return; }
  try {
    const directoryHandle = await window.showDirectoryPicker({ id: 'moment-select-source', mode: 'read' });
    state.skippedFileCount = 0;
    $('#pick-note').textContent = '사진 목록을 읽는 중이에요.';
    const imageEntries = await collectImageEntries(directoryHandle);
    dbSet('lastHandle', directoryHandle);
    await startWithEntries(directoryHandle.name, imageEntries);
  } catch (error) {
    if (error.name !== 'AbortError') showToast(`폴더를 열지 못했어요. ${escapeHtml(error.message)}`);
  }
};

// File System Access API가 없는 브라우저: 폴더 업로드 입력으로 읽기만
$('#fallback-input').onchange = async (event) => {
  const files = [...event.target.files];
  state.skippedFileCount = files.filter((file) => UNSUPPORTED_IMAGE.test(file.name)).length;
  const imageEntries = files
    .filter((file) => SUPPORTED_IMAGE.test(file.name) && !file.webkitRelativePath.includes(`/${EXPORT_FOLDER_PREFIX}`))
    .map((file) => ({ relPath: file.webkitRelativePath.split('/').slice(1).join('/') || file.name, name: file.name, file }));
  await startWithEntries(files[0]?.webkitRelativePath.split('/')[0] || '선택한 폴더', imageEntries);
};

// 휴대폰과 폴더 선택이 안 되는 브라우저: 사진 보관함이나 갤러리에서 여러 장 고르기
$('#pick-photos').onclick = () => $('#photos-input').click();
$('#photos-input').onchange = async (event) => {
  const pickedFiles = [...event.target.files];
  const imageFiles = pickedFiles.filter((file) => file.type.startsWith('image/') || SUPPORTED_IMAGE.test(file.name));
  event.target.value = '';
  if (!imageFiles.length) return;
  state.skippedFileCount = pickedFiles.length - imageFiles.length;
  // 폴더 경로가 없으니 파일 이름을 id로 쓰고, 이름이 겹치면 번호를 붙인다
  const nameCounts = new Map();
  const imageEntries = imageFiles.map((file) => {
    const count = (nameCounts.get(file.name) ?? 0) + 1;
    nameCounts.set(file.name, count);
    return { relPath: count > 1 ? `${file.name} (${count})` : file.name, name: file.name, file };
  });
  await startWithEntries(`사진 ${imageFiles.length}장 (${imageFiles[0].name}부터)`, imageEntries);
};

// 환경에 맞게 시작 버튼을 배치한다
(function arrangeStartButtons() {
  const folderButton = $('#pick-folder'), photosButton = $('#pick-photos');
  if (isTouchDevice) {
    // 휴대폰: 사진 여러 장 선택이 기본. 폴더 선택은 지원하는 브라우저(안드로이드 크롬)에서만 보조로
    photosButton.classList.add('primary');
    folderButton.classList.remove('primary');
    folderButton.classList.toggle('hidden', !canPickFolder);
    photosButton.parentElement.prepend(photosButton);
    $('#pick-note').textContent = '사진 보관함이나 갤러리에서 고를 사진을 모두 선택하세요. 사진은 이 기기 밖으로 나가지 않아요.';
    $('#drive-note').classList.add('hidden');
  } else if (!canPickFolder) {
    $('#pick-note').textContent = '이 브라우저는 폴더에 바로 저장하지 못해서, 내보낼 때 한 장씩 내려받아요. 크롬이나 엣지에서 열면 새 폴더로 바로 복사돼요.';
  }
})();

(async function setupResume() {
  const lastHandle = await dbGet('lastHandle');
  if (!lastHandle?.queryPermission) return;
  const savedProject = await dbGet(`project:${lastHandle.name}`);
  $('#resume').classList.remove('hidden');
  $('#resume-name').textContent = `지난번 폴더 '${lastHandle.name}'`;
  if (savedProject?.settings) {
    Object.assign(state.settings, savedProject.settings);
    $('#target-input').value = state.settings.target;
    $('#category-input').checked = state.settings.useCategories;
    $('#crowd-input').value = state.settings.crowdMin;
    $('#score-input').checked = state.settings.useScore;
    $('#name-a').value = savedProject.reviewers?.a?.name ?? '나';
    $('#name-b').value = savedProject.reviewers?.b?.name ?? '파트너';
    renderPresetOptions();
  }
  $('#resume-button').onclick = async () => {
    if (await lastHandle.requestPermission({ mode: 'read' }) !== 'granted') { showToast('폴더를 읽으려면 권한을 허용해야 해요.'); return; }
    state.skippedFileCount = 0;
    await startWithEntries(lastHandle.name, await collectImageEntries(lastHandle));
  };
})();

/* =====================================================================
   2. 분석
   ===================================================================== */
const analysisCanvas = document.createElement('canvas');
const analysisContext = analysisCanvas.getContext('2d', { willReadFrequently: true });
const thumbCanvas = document.createElement('canvas');
const thumbContext = thumbCanvas.getContext('2d');
const hashCanvas = document.createElement('canvas');
hashCanvas.width = 9; hashCanvas.height = 8;
const hashContext = hashCanvas.getContext('2d', { willReadFrequently: true });

let faceApi = null;
let exifReader = null;

async function loadLibraries() {
  try { exifReader = (await import(EXIFR_URL)).default; } catch { exifReader = null; }
  if (state.skipFaceAnalysis) return;
  try {
    $('#analyze-text').textContent = '얼굴 분석 모델을 받는 중이에요.';
    const module = await import(FACE_API_URL);
    // WebGL(그래픽 가속)을 먼저 쓰고, 안 되면 CPU로. wasm 백엔드는 쓰지 않는다
    const webglReady = await module.tf.setBackend('webgl').catch(() => false);
    if (!webglReady) await module.tf.setBackend('cpu');
    await module.tf.ready();
    await Promise.all([
      module.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_BASE),
      module.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_BASE),
      module.nets.faceExpressionNet.loadFromUri(FACE_MODEL_BASE),
    ]);
    faceApi = module;
  } catch (error) {
    console.warn('face-api load failed', error);
    faceApi = null;
    showToast('얼굴 분석 모델을 불러오지 못해서 선명도와 노출만 보고 진행해요.');
  }
}

function drawScaled(bitmap, longSide, canvas, context) {
  const scale = Math.min(1, longSide / Math.max(bitmap.width, bitmap.height));
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
}

function toGrayscale(context, width, height) {
  const rgba = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height); // 픽셀 밝기 0~255
  for (let rgbaIndex = 0, pixelIndex = 0; rgbaIndex < rgba.length; rgbaIndex += 4, pixelIndex++) {
    gray[pixelIndex] = 0.299 * rgba[rgbaIndex] + 0.587 * rgba[rgbaIndex + 1] + 0.114 * rgba[rgbaIndex + 2];
  }
  return gray;
}

// 3x3 가우시안으로 살짝 흐리게: 픽셀 단위 노이즈는 줄고 실제 경계는 남는다
function gaussianSmooth(gray, width, height) {
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
function laplacianVariance(image, width, height, left = 0, top = 0, right = width, bottom = height) {
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
function focusSharpness(smooth, width, height, noiseFloor) {
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
function noiseSigma(gray, width, height) {
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

function luminanceStats(gray, width, left = 0, top = 0, right = width, bottom = gray.length / width) {
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

function differenceHash() {
  hashContext.drawImage(analysisCanvas, 0, 0, 9, 8);
  const gray = toGrayscale(hashContext, 9, 8);
  const hashWords = [0, 0];
  let bitIndex = 0;
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      if (gray[y * 9 + x] > gray[y * 9 + x + 1]) hashWords[bitIndex >> 5] |= (1 << (bitIndex & 31));
      bitIndex++;
    }
  }
  return hashWords.map((word) => word >>> 0);
}
function popCount(word) { word -= (word >>> 1) & 0x55555555; word = (word & 0x33333333) + ((word >>> 2) & 0x33333333); return (((word + (word >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; }
const hashDistance = (leftHash, rightHash) => popCount((leftHash[0] ^ rightHash[0]) >>> 0) + popCount((leftHash[1] ^ rightHash[1]) >>> 0);

function eyeAspectRatio(eyePoints) {
  const distance = (pointA, pointB) => Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
  return (distance(eyePoints[1], eyePoints[5]) + distance(eyePoints[2], eyePoints[4])) / (2 * distance(eyePoints[0], eyePoints[3]));
}

async function analyzeFaces(gray, smooth, width, height, noiseFloor) {
  if (!faceApi) return null;
  const detections = await faceApi
    .detectAllFaces(analysisCanvas, new faceApi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }))
    .withFaceLandmarks()
    .withFaceExpressions();
  return detections.map((detection) => {
    const box = detection.detection.box;
    const points = detection.landmarks.positions;
    // 고개 돌림: 코끝이 두 눈 바깥 끝의 가운데에서 얼마나 벗어났는지 (눈 사이 거리 대비)
    const eyeSpan = Math.abs(points[45].x - points[36].x) || 1;
    const yaw = (points[30].x - (points[36].x + points[45].x) / 2) / eyeSpan;
    const canJudgeEyes = box.height >= EYE_CHECK_MIN_FACE_PX && Math.abs(yaw) <= EYE_CHECK_MAX_YAW && detection.detection.score >= EYE_CHECK_MIN_CONFIDENCE;
    const leftEar = eyeAspectRatio(points.slice(36, 42));
    const rightEar = eyeAspectRatio(points.slice(42, 48));
    const { expressions } = detection;
    const faceLight = luminanceStats(gray, width, box.x, box.y, box.x + box.width, box.y + box.height);
    return {
      box: { x: box.x / width, y: box.y / height, w: box.width / width, h: box.height / height },
      isMainFace: box.height / height >= MAIN_FACE_MIN_HEIGHT,
      sharpness: Math.max(0, laplacianVariance(smooth, width, height, box.x, box.y, box.x + box.width, box.y + box.height) - noiseFloor), // 노이즈가 만든 가짜 선명도는 뺀다
      luminance: faceLight.mean,
      smile: expressions.happy ?? 0,
      negative: (expressions.sad ?? 0) + (expressions.angry ?? 0) + (expressions.fearful ?? 0) + (expressions.disgusted ?? 0),
      // 두 눈 모두 감겨 보여야 눈 감음 (한쪽만이면 윙크나 각도 문제일 가능성이 크다)
      eyesClosed: canJudgeEyes && leftEar < EYE_CLOSED_EAR && rightEar < EYE_CLOSED_EAR,
      eyesJudged: canJudgeEyes,
      cutAtEdge: box.x < 2 || box.y < 2 || box.x + box.width > width - 2 || box.y + box.height > height - 2,
    };
  });
}

async function analyzeOne(photo) {
  const file = await getPhotoFile(photo);
  const cacheKey = `cache:${CACHE_VERSION}:${photo.relPath}|${file.size}|${file.lastModified}|${faceApi ? 'face' : 'basic'}`;
  const cached = await dbGet(cacheKey);
  if (cached) {
    Object.assign(photo, { takenAt: cached.takenAt, width: cached.width, height: cached.height, analysis: cached.analysis });
    photo.thumbUrl = URL.createObjectURL(cached.thumbBlob);
    return;
  }
  let takenAt = null;
  if (exifReader) {
    try {
      const exif = await exifReader.parse(file, ['DateTimeOriginal', 'CreateDate']);
      const date = exif?.DateTimeOriginal ?? exif?.CreateDate;
      takenAt = date instanceof Date && !isNaN(date) ? date.getTime() : null;
    } catch { takenAt = null; }
  }
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  photo.width = bitmap.width; photo.height = bitmap.height;
  drawScaled(bitmap, THUMB_LONG_SIDE, thumbCanvas, thumbContext);
  const thumbBlob = await new Promise((resolve) => thumbCanvas.toBlob(resolve, 'image/jpeg', 0.82));
  drawScaled(bitmap, ANALYSIS_LONG_SIDE, analysisCanvas, analysisContext);
  bitmap.close();

  const { width, height } = analysisCanvas;
  const gray = toGrayscale(analysisContext, width, height);
  const smooth = gaussianSmooth(gray, width, height);
  const noise = noiseSigma(gray, width, height);
  const noiseFloor = NOISE_LAPLACIAN_GAIN * noise * noise; // 노이즈만으로 생기는 라플라시안 분산
  photo.analysis = {
    focusSharpness: focusSharpness(smooth, width, height, noiseFloor),
    noise,
    light: luminanceStats(gray, width),
    hash: differenceHash(),
    faces: await analyzeFaces(gray, smooth, width, height, noiseFloor), // null이면 얼굴 분석을 하지 않음
  };
  photo.takenAt = takenAt;
  photo.thumbUrl = URL.createObjectURL(thumbBlob);
  dbSet(cacheKey, { takenAt, width: photo.width, height: photo.height, analysis: photo.analysis, thumbBlob });
}

async function analyzeAll() {
  state.analysisDone = false;
  $('#analyze-meter').style.width = '0%';
  $('#analyze-sheet').innerHTML = '';
  await loadLibraries();
  const startedAt = performance.now();
  const totalCount = state.photos.length;
  let finishedCount = 0;
  for (const photo of state.photos) {
    try {
      await analyzeOne(photo);
    } catch (error) {
      console.warn('analyze failed', photo.relPath, error);
      photo.analysis = { focusSharpness: 0, noise: 0, light: { mean: 128, brightClip: 0, darkClip: 0 }, hash: [0, 0], faces: null, failed: true };
    }
    finishedCount++;
    const remainingSeconds = Math.round(((performance.now() - startedAt) / 1000 / finishedCount) * (totalCount - finishedCount));
    $('#analyze-meter').style.width = `${(finishedCount / totalCount) * 100}%`;
    $('#analyze-text').textContent = `${finishedCount} / ${totalCount}장, 남은 시간 약 ${remainingSeconds >= 60 ? `${Math.round(remainingSeconds / 60)}분` : `${remainingSeconds}초`}${faceApi ? '' : ', 얼굴 분석 꺼짐'}`;
    if (photo.thumbUrl) $('#analyze-sheet').insertAdjacentHTML('beforeend', `<img src="${photo.thumbUrl}" alt="">`);
  }
  // 브라우저가 열지 못한 사진(HEIC 등)은 고를 수도 없으니 목록에서 뺀다
  const unreadable = state.photos.filter((photo) => photo.analysis.failed);
  if (unreadable.length) {
    state.photos = state.photos.filter((photo) => !photo.analysis.failed);
    state.photoById = new Map(state.photos.map((photo) => [photo.id, photo]));
    state.skippedFileCount += unreadable.length;
  }
  if (!state.photos.length) { showToast('이 브라우저가 연 사진이 없어요. JPG로 바꾼 사진을 골라주세요.'); setView('setup'); return; }
  computeFeatures();
  rescoreAll();
  buildSimilarGroups();
  buildUnits();
  state.analysisDone = true;
  saveProjectSoon();
  const skipNote = state.skippedFileCount ? ` 열 수 없는 파일 ${state.skippedFileCount}개(RAW, HEIC 등)는 건너뛰었어요.` : '';
  showToast(`${state.photos.length}장을 분석했어요. 비슷한 컷 묶음 ${state.groups.size}개.${skipNote}`);
  setView('first');
}

$('#skip-faces').onclick = () => {
  state.skipFaceAnalysis = true;
  faceApi = null;
  $('#skip-faces').disabled = true;
  showToast('남은 사진은 선명도와 노출만 보고 빠르게 분석해요.');
};

/* =====================================================================
   점수 항목 계산 (모두 0~1, 클수록 좋음)
   - 선명도와 노이즈는 촬영마다 절대값이 달라서 이 폴더 안의 백분위로 바꾼다
   - 눈 감음은 항목에 넣지 않는다
   ===================================================================== */
function percentileRanker(values) {
  const sortedValues = [...values].sort((left, right) => left - right);
  return (value) => {
    if (sortedValues.length <= 1) return 0.5;
    let low = 0, high = sortedValues.length;
    while (low < high) { const middle = (low + high) >> 1; if (sortedValues[middle] < value) low = middle + 1; else high = middle; }
    return low / (sortedValues.length - 1);
  };
}
const mainFacesOf = (photo) => (photo.analysis?.faces ?? []).filter((face) => face.isMainFace);
const average = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

// 얼굴 밝기 90~200이면 1, 바깥은 30과 245에서 0이 되도록 줄인다
const faceExposureScore = (luminance) => (luminance < 90 ? clamp01((luminance - 30) / 60) : luminance > 200 ? clamp01((245 - luminance) / 45) : 1);
// 사진 전체는 고정 목표 밝기를 두지 않고 극단값과 클리핑만 깎는다 (일부러 어둡거나 밝게 찍은 사진 보호)
function globalExposureScore({ mean, brightClip, darkClip }) {
  return clamp01(1 - Math.max(0, brightClip - 0.06) * 4 - Math.max(0, darkClip - 0.2) * 2 - Math.max(0, 40 - mean) / 40 - Math.max(0, mean - 225) / 30);
}

function computeFeatures() {
  const usable = state.photos.filter((photo) => !photo.analysis.failed);
  const focusRank = percentileRanker(usable.map((photo) => Math.log1p(photo.analysis.focusSharpness)));
  const faceRank = percentileRanker(usable.flatMap((photo) => mainFacesOf(photo).map((face) => Math.log1p(face.sharpness))));
  const noiseRank = percentileRanker(usable.map((photo) => photo.analysis.noise));
  for (const photo of state.photos) {
    if (photo.analysis.failed) { photo.features = null; continue; }
    const faces = mainFacesOf(photo);
    const focusSharp = focusRank(Math.log1p(photo.analysis.focusSharpness));
    const globalExposure = globalExposureScore(photo.analysis.light);
    if (faces.length) {
      photo.features = {
        subjectSharp: average(faces.map((face) => faceRank(Math.log1p(face.sharpness)))),
        focusSharp,
        exposure: 0.6 * average(faces.map((face) => faceExposureScore(face.luminance))) + 0.4 * globalExposure,
        expression: clamp01(1 - average(faces.map((face) => face.negative))),
        smile: average(faces.map((face) => face.smile)),
        framing: faces.filter((face) => !face.cutAtEdge).length / faces.length,
        clean: 1 - noiseRank(photo.analysis.noise),
      };
    } else {
      // 인물이 없으면 인물 항목은 초점 부분으로 대신하고, 표정과 구도는 중립값 0.5
      photo.features = { subjectSharp: focusSharp, focusSharp, exposure: globalExposure, expression: 0.5, smile: 0.5, framing: 0.5, clean: 1 - noiseRank(photo.analysis.noise) };
    }
  }
}

const featureVector = (photo) => FEATURE_KEYS.map((key) => photo.features[key]);
const activeModel = (reviewerKey = state.currentReviewer) => state.models[reviewerKey] ?? state.priorModel;
function scoreWith(model, photo) {
  if (!photo.features) return 0;
  const vector = featureVector(photo);
  return Math.round(100 * sigmoid(model.bias + model.theta.reduce((sum, weight, index) => sum + weight * vector[index], 0)));
}
function rescoreAll() {
  const model = activeModel();
  for (const photo of state.photos) photo.score = scoreWith(model, photo);
  assignGroupBest();
}

/* =====================================================================
   취향 학습: 이 사람의 남김과 뺌으로 로지스틱 회귀를 다시 맞춘다
   - 기본 모델에서 너무 멀어지지 않게 L2로 붙잡아 둔다 (결정이 적을 때 튀지 않게)
   - '보류 전부 빼기'처럼 사진을 보지 않고 한 결정은 학습에서 뺀다
   ===================================================================== */
function trainingSamples(reviewerKey) {
  const samples = [];
  for (const [photoId, decision] of Object.entries(state.reviewers[reviewerKey].decisions)) {
    const photo = state.photoById.get(photoId);
    const lastStage = decision.log?.[decision.log.length - 1]?.stage;
    if (!photo?.features || lastStage === 'first-bulk') continue;
    samples.push({ vector: featureVector(photo), label: decision.status === 'keep' ? 1 : 0 });
  }
  return samples;
}

function trainModel(reviewerKey) {
  const samples = trainingSamples(reviewerKey);
  const keepCount = samples.filter((sample) => sample.label === 1).length;
  if (samples.length < LEARN_MIN_DECISIONS || keepCount < LEARN_MIN_PER_CLASS || samples.length - keepCount < LEARN_MIN_PER_CLASS) return false;
  const prior = state.priorModel;
  const theta = [...prior.theta];
  let bias = prior.bias;
  const regularization = 0.01, learningRate = 1.5;
  for (let iteration = 0; iteration < 1500; iteration++) {
    const thetaGradient = new Array(theta.length).fill(0);
    let biasGradient = 0;
    for (const { vector, label } of samples) {
      const error = sigmoid(bias + theta.reduce((sum, weight, index) => sum + weight * vector[index], 0)) - label;
      for (let index = 0; index < theta.length; index++) thetaGradient[index] += error * vector[index];
      biasGradient += error;
    }
    for (let index = 0; index < theta.length; index++) {
      theta[index] -= learningRate * (thetaGradient[index] / samples.length + 2 * regularization * (theta[index] - prior.theta[index]));
    }
    bias -= learningRate * (biasGradient / samples.length);
  }
  state.models[reviewerKey] = { theta, bias, source: 'learned', trainedOn: samples.length };
  return true;
}

function maybeRetrain() {
  if (!state.settings.useScore) return;
  const reviewerKey = state.currentReviewer;
  const sampleCount = trainingSamples(reviewerKey).length;
  const lastTrained = state.models[reviewerKey]?.trainedOn ?? 0;
  if (sampleCount - lastTrained < LEARN_EVERY && lastTrained !== 0) return;
  if (lastTrained === 0 && sampleCount < LEARN_MIN_DECISIONS) return;
  if (!trainModel(reviewerKey)) return;
  rescoreAll();
  buildUnits();
  if (lastTrained === 0) showToast(`결정 ${sampleCount}건으로 취향을 배웠어요. 남은 사진 순서를 다시 정했어요. 점수 기준에서 무엇을 중요하게 보는지 확인할 수 있어요.`, [], 5500);
  maybeSendStats();
}

/* ---------- 익명 통계 보내기 (서버 주소가 있고 동의했을 때만) ---------- */
function maybeSendStats(force = false) {
  if (!COLLECT_ENDPOINT || !state.settings.share) return;
  const reviewerKey = state.currentReviewer;
  const samples = trainingSamples(reviewerKey);
  if (!force && samples.length - state.sentSampleCount[reviewerKey] < 50) return;
  const payload = {
    app: 'moment-select', version: 2, preset: state.settings.preset,
    // 사진, 파일 이름, 얼굴 위치는 보내지 않는다. 항목 값과 남김 여부만
    samples: samples.slice(state.sentSampleCount[reviewerKey]).map((sample) => ({ x: sample.vector.map((value) => Math.round(value * 1000) / 1000), y: sample.label })),
    model: state.models[reviewerKey] ? { theta: state.models[reviewerKey].theta, bias: state.models[reviewerKey].bias } : null,
  };
  if (!payload.samples.length) return;
  const body = JSON.stringify(payload);
  const sent = navigator.sendBeacon?.(COLLECT_ENDPOINT, new Blob([body], { type: 'application/json' }));
  if (!sent) fetch(COLLECT_ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, keepalive: true }).catch(() => {});
  state.sentSampleCount[reviewerKey] = samples.length;
  saveProjectSoon();
}
addEventListener('pagehide', () => maybeSendStats(true));

/* =====================================================================
   인원 분류, 비슷한 컷 묶기, 1차 셀렉 단위
   ===================================================================== */
function categoryOf(photo) {
  if (!photo.analysis?.faces) return 'unknown';
  const faceCount = mainFacesOf(photo).length;
  if (faceCount === 0) return 'none';
  if (faceCount === 1) return 'solo';
  if (faceCount === 2) return 'pair';
  if (faceCount >= state.settings.crowdMin) return 'crowd';
  return 'small';
}
const CATEGORY_LABELS = { all: '전체', solo: '혼자', pair: '둘', small: '소그룹', crowd: '단체', none: '인물 없음', unknown: '분류 안 됨' };
const hasClosedEyes = (photo) => mainFacesOf(photo).some((face) => face.eyesClosed);

function buildSimilarGroups() {
  state.groups = new Map();
  const ordered = [...state.photos].sort((left, right) => {
    if (left.takenAt && right.takenAt && left.takenAt !== right.takenAt) return left.takenAt - right.takenAt;
    return left.relPath.localeCompare(right.relPath, undefined, { numeric: true });
  });
  let currentRun = [];
  const flushRun = () => {
    if (currentRun.length >= 2) {
      const groupId = `g${state.groups.size + 1}`;
      state.groups.set(groupId, currentRun.map((photo) => photo.id));
      for (const photo of currentRun) photo.groupId = groupId;
    }
    currentRun = [];
  };
  for (const photo of ordered) {
    photo.groupId = null;
    const previousPhoto = currentRun[currentRun.length - 1];
    if (previousPhoto) {
      const closeInTime = previousPhoto.takenAt && photo.takenAt ? Math.abs(photo.takenAt - previousPhoto.takenAt) <= SIMILAR_TIME_GAP_MS : true;
      const similarLook = !photo.analysis.failed && !previousPhoto.analysis.failed && hashDistance(photo.analysis.hash, previousPhoto.analysis.hash) <= SIMILAR_HASH_DISTANCE;
      if (!(closeInTime && similarLook) || currentRun.length >= GROUP_MAX_SIZE) flushRun();
    }
    currentRun.push(photo);
  }
  flushRun();
  assignGroupBest();
}

function assignGroupBest() {
  for (const memberIds of state.groups.values()) {
    const members = memberIds.map((id) => state.photoById.get(id));
    const best = members.reduce((top, photo) => (photo.score > top.score ? photo : top));
    for (const photo of members) photo.groupBest = photo === best;
  }
}

function buildUnits() {
  const units = [];
  const seenGroups = new Set();
  for (const photo of state.photos) {
    if (photo.groupId && !state.expandedGroups.has(photo.groupId)) {
      if (seenGroups.has(photo.groupId)) continue;
      seenGroups.add(photo.groupId);
      const memberIds = state.groups.get(photo.groupId);
      units.push({ key: photo.groupId, type: 'group', ids: memberIds, score: Math.max(...memberIds.map((id) => state.photoById.get(id).score)) });
    } else {
      units.push({ key: photo.id, type: 'single', ids: [photo.id], score: photo.score });
    }
  }
  const sortedScores = units.map((unit) => unit.score).sort((left, right) => right - left);
  const cutoff = sortedScores[Math.max(0, Math.ceil(sortedScores.length * RECOMMEND_RATIO) - 1)] ?? 0;
  for (const unit of units) {
    unit.recommended = state.settings.useScore && unit.score >= cutoff;
    for (const id of unit.ids) state.photoById.get(id).recommended = unit.recommended;
  }
  state.units = units;
}

/* =====================================================================
   결정 기록과 되돌리기
   ===================================================================== */
function emptyUndo() { return { reviewerKey: state.currentReviewer, previous: [], roundMarks: [] }; }

function recordDecisions(changes, stage) {
  const now = Date.now();
  const undoEntry = emptyUndo();
  for (const { id, status } of changes) {
    const previousDecision = decisionOf(id);
    undoEntry.previous.push({ id, decision: previousDecision ? structuredClone(previousDecision) : null });
    reviewer().decisions[id] = {
      status, at: now,
      // 뺐던 사진을 다시 남기면 그 시각을 기록한다
      addedAt: status === 'keep' ? (previousDecision?.status === 'drop' ? now : previousDecision?.addedAt) : undefined,
      log: [...(previousDecision?.log ?? []), { status, at: now, stage }],
    };
  }
  state.undoStack.push(undoEntry);
  if (state.undoStack.length > 300) state.undoStack.shift();
  saveProjectSoon();
  maybeRetrain();
  return undoEntry;
}

function markRoundSeen(roundName, ids, undoEntry) {
  const seenIds = state.rounds[state.currentReviewer][roundName];
  for (const id of ids) {
    if (!seenIds.includes(id)) { seenIds.push(id); undoEntry.roundMarks.push({ roundName, id }); }
  }
  saveProjectSoon();
}

function undoLast() {
  const undoEntry = state.undoStack.pop();
  if (!undoEntry) { showToast('되돌릴 결정이 없어요.', [], 1500); return; }
  const decisions = state.reviewers[undoEntry.reviewerKey].decisions;
  for (const { id, decision } of undoEntry.previous) {
    if (decision) decisions[id] = decision; else delete decisions[id];
  }
  for (const { roundName, id } of undoEntry.roundMarks) {
    const seenIds = state.rounds[undoEntry.reviewerKey][roundName];
    const index = seenIds.indexOf(id);
    if (index >= 0) seenIds.splice(index, 1);
  }
  state.priorityIds = [...undoEntry.previous.map((item) => item.id), ...undoEntry.roundMarks.map((mark) => mark.id)];
  saveProjectSoon();
  render();
}

/* =====================================================================
   볼 순서 (큐)
   ===================================================================== */
const matchesCategory = (photo) => !state.settings.useCategories || state.categoryFilter === 'all' || categoryOf(photo) === state.categoryFilter;
const keepIds = (reviewerKey = state.currentReviewer) => state.photos.filter((photo) => statusOf(photo.id, reviewerKey) === 'keep').map((photo) => photo.id);
const dropIds = (reviewerKey = state.currentReviewer) => state.photos.filter((photo) => statusOf(photo.id, reviewerKey) === 'drop').map((photo) => photo.id);
const undecidedCount = () => state.photos.filter((photo) => !statusOf(photo.id)).length;

function bringPriorityFirst(items, idsOf) {
  if (!state.priorityIds.length) return items;
  const priority = items.filter((item) => idsOf(item).some((id) => state.priorityIds.includes(id)));
  return [...priority, ...items.filter((item) => !priority.includes(item))];
}

function firstPassQueue() {
  const pending = state.units.filter((unit) => unit.ids.some((id) => !statusOf(id)) && unit.ids.some((id) => matchesCategory(state.photoById.get(id))));
  if (!state.settings.useScore) return bringPriorityFirst(pending, (unit) => unit.ids);
  const byScore = (left, right) => right.score - left.score;
  const recommended = pending.filter((unit) => unit.recommended).sort(byScore);
  const others = pending.filter((unit) => !unit.recommended).sort(byScore);
  const queue = [...recommended];
  if (others.length) {
    if (!state.dividerSeen[state.currentReviewer]) queue.push({ type: 'divider', key: 'divider', ids: [], count: others.length });
    queue.push(...others);
  }
  return bringPriorityFirst(queue, (unit) => unit.ids);
}
function distillQueue() {
  const seenIds = state.rounds[state.currentReviewer].distill;
  const queue = keepIds().map((id) => state.photoById.get(id))
    .filter((photo) => !seenIds.includes(photo.id) && matchesCategory(photo))
    .sort((left, right) => left.score - right.score);
  return bringPriorityFirst(queue, (photo) => [photo.id]);
}
function rescueQueue() {
  const seenIds = state.rounds[state.currentReviewer].rescue;
  const queue = dropIds().map((id) => state.photoById.get(id))
    .filter((photo) => !seenIds.includes(photo.id) && matchesCategory(photo))
    .sort((left, right) => right.score - left.score);
  return bringPriorityFirst(queue, (photo) => [photo.id]);
}
function comparePair() {
  const queue = distillQueue();
  if (!queue.length) return null;
  const weakest = queue[0];
  const candidates = keepIds().filter((id) => id !== weakest.id).map((id) => state.photoById.get(id));
  if (!candidates.length) return null;
  const closest = candidates.reduce((best, photo) => {
    const distance = hashDistance(weakest.analysis.hash, photo.analysis.hash) - (photo.groupId && photo.groupId === weakest.groupId ? 100 : 0);
    return distance < best.distance ? { photo, distance } : best;
  }, { photo: null, distance: Infinity }).photo;
  return [weakest, closest];
}

/* =====================================================================
   화면 전환과 상단 바
   ===================================================================== */
const PICK_VIEWS = ['first', 'distill', 'rescue'];
function setView(viewName) {
  state.view = viewName;
  state.priorityIds = [];
  for (const section of document.querySelectorAll('.view')) section.classList.add('hidden');
  $(`#view-${PICK_VIEWS.includes(viewName) ? 'pick' : viewName}`).classList.remove('hidden');
  render();
}
$('#tabs').onclick = (event) => {
  const button = event.target.closest('button[data-view]');
  if (button && !button.disabled) setView(button.dataset.view);
};
$('#reviewer-select').onchange = (event) => {
  state.currentReviewer = event.target.value;
  state.undoStack = [];
  rescoreAll();
  buildUnits();
  showToast(`이제 ${escapeHtml(reviewer().name)}의 결정으로 골라요. 결정과 학습은 사람마다 따로 저장돼요.`, [], 2600);
  render();
};

function renderTopbar() {
  const ready = state.analysisDone;
  for (const button of document.querySelectorAll('#tabs button')) {
    const isCurrent = button.dataset.view === state.view || (state.view === 'analyze' && button.dataset.view === 'setup');
    if (isCurrent) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    if (button.dataset.view !== 'setup') button.disabled = !ready;
  }
  const keepCount = ready ? keepIds().length : 0;
  const target = state.settings.target;
  $('#tab-first').textContent = ready ? undecidedCount() : '';
  $('#tab-keep').textContent = ready ? keepCount : '';
  $('#tab-drop').textContent = ready ? dropIds().length : '';
  $('#goal-count').textContent = keepCount;
  $('#goal-target').textContent = target;
  $('#goal').classList.toggle('over', keepCount > target);
  $('#goal-note').textContent = keepCount > target ? `${keepCount - target}장 많음` : '';
  $('#goal-meter').style.width = `${Math.min(100, (keepCount / target) * 100)}%`;
  $('#reviewer-select').innerHTML = Object.entries(state.reviewers)
    .map(([key, value]) => `<option value="${key}" ${key === state.currentReviewer ? 'selected' : ''}>${escapeHtml(value.name)}</option>`).join('');
}

function render() {
  renderTopbar();
  if (PICK_VIEWS.includes(state.view)) renderPickView();
  if (state.view === 'export') renderExportView();
}

/* =====================================================================
   고르기 화면
   ===================================================================== */
const fullImageUrls = new Map(); // 원본 objectURL (지금 볼 사진과 다음 두 장만 유지)
function fullImageUrl(photo) {
  if (!fullImageUrls.has(photo.id)) fullImageUrls.set(photo.id, getPhotoFile(photo).then((file) => URL.createObjectURL(file)));
  return fullImageUrls.get(photo.id);
}
function keepOnlyUrls(photoIdsToKeep) {
  for (const [id, urlPromise] of fullImageUrls) {
    if (!photoIdsToKeep.includes(id)) { urlPromise.then((url) => URL.revokeObjectURL(url)); fullImageUrls.delete(id); }
  }
}

function renderFilters(sourcePhotos) {
  const filters = $('#filters');
  if (!state.settings.useCategories) { filters.innerHTML = ''; return; }
  const counts = { all: sourcePhotos.length };
  for (const photo of sourcePhotos) { const category = categoryOf(photo); counts[category] = (counts[category] ?? 0) + 1; }
  filters.innerHTML = ['all', 'solo', 'pair', 'small', 'crowd', 'none', 'unknown']
    .filter((key) => key === 'all' || counts[key])
    .map((key) => `<button data-category="${key}" aria-pressed="${state.categoryFilter === key}">${CATEGORY_LABELS[key]}<span class="num">${counts[key] ?? 0}</span></button>`).join('');
}
$('#filters').onclick = (event) => {
  const button = event.target.closest('[data-category]');
  if (!button) return;
  state.categoryFilter = button.dataset.category;
  render();
};
$('#compare-input').onchange = (event) => { state.compareMode = event.target.checked; event.target.blur(); render(); };

function renderFilmstrip(entries) {
  // entries: [{ids, photo, count}] 지금 볼 것부터
  const filmstrip = $('#filmstrip');
  const visible = entries.slice(0, 16);
  filmstrip.innerHTML = visible.map((entry, index) => `
    <button class="${index === 0 ? 'current' : ''}" data-ids="${escapeHtml(JSON.stringify(entry.ids))}" title="${escapeHtml(entry.photo.relPath)}" aria-label="${index === 0 ? '지금 보는 사진' : `${index + 1}번째로 볼 사진`}">
      <img src="${entry.photo.thumbUrl}" alt="" loading="lazy">${entry.count > 1 ? `<span class="count num">${entry.count}</span>` : ''}
    </button>`).join('') + (entries.length > visible.length ? `<span class="more num">${entries.length - visible.length}장 더</span>` : '');
}
$('#filmstrip').onclick = (event) => {
  const button = event.target.closest('button[data-ids]');
  if (!button || button.classList.contains('current')) return;
  state.priorityIds = JSON.parse(button.dataset.ids);
  render();
};

function setNotice(html, alert = false) {
  const notice = $('#notice');
  notice.innerHTML = html;
  notice.classList.toggle('alert', alert);
}

function photoTagsHtml(photo, { showAdded }) {
  const tags = [];
  if (state.settings.useScore) tags.push(`<span class="tag num">점수 ${photo.score}</span>`);
  if (state.settings.useCategories) tags.push(`<span class="tag">${CATEGORY_LABELS[categoryOf(photo)]}</span>`);
  if (photo.groupId && state.view !== 'first') tags.push(`<span class="tag num">비슷한 컷 ${state.groups.get(photo.groupId).length}장 중 하나</span>`);
  if (hasClosedEyes(photo)) tags.push(`<span class="tag flag"><i class="ph ph-eye-closed"></i> 눈 감음 의심${state.settings.preset === 'wedding' ? ', 합성 요청 후보' : ''}</span>`);
  const decision = decisionOf(photo.id);
  if (showAdded && decision?.addedAt) tags.push(`<span class="tag added">${formatKoreanTime(decision.addedAt)}에 다시 남김</span>`);
  return tags.join('');
}

function renderPickView() {
  const mode = state.view;
  $('#compare-wrap').classList.toggle('hidden', mode !== 'distill');
  $('#criteria-button').classList.toggle('hidden', !state.settings.useScore);
  $('#compare-input').checked = state.compareMode;
  const stage = $('#stage');
  const keepCount = keepIds().length;
  const target = state.settings.target;

  if (mode === 'first') {
    renderFilters(state.photos);
    const queue = firstPassQueue();
    const remaining = queue.reduce((sum, unit) => sum + unit.ids.filter((id) => !statusOf(id)).length, 0);
    setNotice(`남은 사진 <b class="num">${remaining}</b>장${state.settings.useScore ? '. 점수 높은 사진부터 보여드려요.' : ''}`);
    renderFilmstrip(queue.filter((unit) => unit.type !== 'divider').map((unit) => {
      const members = unit.ids.map((id) => state.photoById.get(id));
      return { ids: unit.ids, photo: members.find((photo) => photo.groupBest) ?? members[0], count: unit.ids.length };
    }));
    const head = queue[0];
    if (!head) {
      return renderEmpty(stage, '1차 셀렉을 마쳤어요', `남긴 사진 ${keepCount}장, 목표 ${target}장. ${keepCount > target ? '줄이기에서 목표 장수까지 줄여보세요.' : '뺀 사진 중 아까운 컷이 있는지 다시 볼 수 있어요.'}`,
        [{ label: '줄이기', primary: keepCount > target, run: () => setView('distill') }, { label: '뺀 사진 다시 보기', run: () => setView('rescue') }, { label: '내보내기', primary: keepCount <= target, run: () => setView('export') }]);
    }
    if (head.type === 'divider') return renderDivider(stage, head);
    if (head.type === 'group') return renderBurst(stage, head);
    return renderSingle(stage, state.photoById.get(head.ids[0]), queue.slice(1).flatMap((unit) => unit.ids).slice(0, 2), 'first');
  }

  if (mode === 'distill') {
    renderFilters(keepIds().map((id) => state.photoById.get(id)));
    const queue = distillQueue();
    setNotice(keepCount > target
      ? `목표까지 <b class="num">${keepCount - target}장</b> 더 빼야 해요. 점수 낮은 사진부터 보여드려요.`
      : `목표 ${target}장을 맞췄어요. 더 줄이려면 계속 넘기세요.`, keepCount > target);
    renderFilmstrip(queue.map((photo) => ({ ids: [photo.id], photo, count: 1 })));
    if (state.compareMode) {
      const pair = comparePair();
      if (pair) return renderCompare(stage, pair);
    }
    const head = queue[0];
    if (!head) {
      return renderEmpty(stage, '이번 줄이기를 다 봤어요', `남긴 사진은 ${keepCount}장이에요. 한 번 더 돌면 남은 사진을 다시 점수 낮은 순으로 보여줘요.`,
        [{ label: '한 번 더 줄이기', primary: keepCount > target, run: () => { state.rounds[state.currentReviewer].distill = []; saveProjectSoon(); render(); } },
         { label: '내보내기', primary: keepCount <= target, run: () => setView('export') }]);
    }
    return renderSingle(stage, head, queue.slice(1, 3).map((photo) => photo.id), 'distill');
  }

  renderFilters(dropIds().map((id) => state.photoById.get(id)));
  const queue = rescueQueue();
  setNotice('뺀 사진을 점수 높은 순으로 다시 봐요. 남기면 다시 남긴 시각이 기록돼요.');
  renderFilmstrip(queue.map((photo) => ({ ids: [photo.id], photo, count: 1 })));
  const head = queue[0];
  if (!head) {
    return renderEmpty(stage, '뺀 사진을 다 다시 봤어요', `남긴 사진 ${keepCount}장, 목표 ${target}장.`,
      [{ label: '처음부터 다시 보기', run: () => { state.rounds[state.currentReviewer].rescue = []; saveProjectSoon(); render(); } },
       { label: keepCount > target ? '줄이기' : '내보내기', primary: true, run: () => setView(keepCount > target ? 'distill' : 'export') }]);
  }
  return renderSingle(stage, head, queue.slice(1, 3).map((photo) => photo.id), 'rescue');
}

let currentKeys = null;   // 지금 화면의 키보드 동작 {left, right, down, zoom, faces, number}
let stagePhotoId = null;  // 무대에 낱장으로 떠 있는 사진 id (점수 기준 패널에서 사용)

function renderEmpty(stage, title, body, buttons) {
  keepOnlyUrls([]);
  stage.innerHTML = `<div class="empty"><h2>${title}</h2><p>${body}</p><div class="actions"></div></div>`;
  for (const spec of buttons) {
    const button = document.createElement('button');
    button.className = `button${spec.primary ? ' primary' : ''}`;
    button.textContent = spec.label;
    button.onclick = spec.run;
    $('.actions', stage).appendChild(button);
  }
  $('#controls').innerHTML = '';
  currentKeys = null;
}

function renderDivider(stage, divider) {
  keepOnlyUrls([]);
  stage.innerHTML = `<div class="empty"><h2>점수 높은 사진을 다 봤어요</h2>
    <p>점수가 낮은 ${divider.count}개 묶음과 낱장이 남았어요. 점수가 낮을 뿐 볼만한 컷이 섞여 있을 수 있어요. 지금 모두 빼도 나중에 '다시 보기'에서 볼 수 있어요.</p>
    <div class="actions"><button class="button primary" id="divider-continue">남은 사진도 보기</button><button class="button" id="divider-drop">남은 사진 모두 빼기</button></div></div>`;
  $('#divider-continue').onclick = () => { state.dividerSeen[state.currentReviewer] = true; saveProjectSoon(); render(); };
  $('#divider-drop').onclick = () => {
    const pendingIds = state.units.filter((unit) => !unit.recommended).flatMap((unit) => unit.ids).filter((id) => !statusOf(id) && matchesCategory(state.photoById.get(id)));
    recordDecisions(pendingIds.map((id) => ({ id, status: 'drop' })), 'first-bulk');
    state.dividerSeen[state.currentReviewer] = true;
    showToast(`남은 ${pendingIds.length}장을 뺐어요.`, [{ label: '되돌리기', run: undoLast }]);
    render();
  };
  $('#controls').innerHTML = '';
  currentKeys = null;
}

// ---------- 낱장 ----------
function renderSingle(stage, photo, preloadIds, stageName) {
  stagePhotoId = photo.id;
  keepOnlyUrls([photo.id, ...preloadIds]);
  for (const id of preloadIds) fullImageUrl(state.photoById.get(id));
  const ratio = photo.width && photo.height ? photo.width / photo.height : 1.5;
  const keepWord = stageName === 'distill' ? '남김 유지' : '남기기';
  stage.innerHTML = `
    <div class="frame" style="--ratio:${ratio}">
      <div class="photo"><img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}"></div>
      ${mainFacesOf(photo).map((face) => `<div class="facebox ${face.eyesClosed ? 'closed' : ''}" style="left:${face.box.x * 100}%;top:${face.box.y * 100}%;width:${face.box.w * 100}%;height:${face.box.h * 100}%"></div>`).join('')}
      <div class="tags">${photoTagsHtml(photo, { showAdded: stageName === 'distill' })}<span class="tag path">${escapeHtml(photo.relPath)}</span></div>
    </div>`;
  const frame = $('.frame', stage);
  const keepMark = cloneMark('keep'), dropMark = cloneMark('drop');
  frame.append(keepMark, dropMark);
  const image = $('.photo img', frame);
  fullImageUrl(photo).then((url) => { if (image.isConnected) image.src = url; });

  const decide = (status) => {
    if (stageName === 'distill') {
      const undoEntry = status === 'drop' ? recordDecisions([{ id: photo.id, status: 'drop' }], 'distill') : emptyUndo();
      if (status === 'keep') state.undoStack.push(undoEntry);
      markRoundSeen('distill', [photo.id], undoEntry);
    } else if (stageName === 'rescue') {
      const undoEntry = status === 'keep' ? recordDecisions([{ id: photo.id, status: 'keep' }], 'rescue') : emptyUndo();
      if (status === 'drop') state.undoStack.push(undoEntry);
      markRoundSeen('rescue', [photo.id], undoEntry);
      const keepCount = keepIds().length;
      if (status === 'keep' && keepCount > state.settings.target) {
        showToast(`다시 남겼어요. 이제 목표보다 ${keepCount - state.settings.target}장 많아요. 줄이기에서 점수 낮은 사진부터 빼주세요.`, [{ label: '줄이기로 가기', run: () => setView('distill') }], 5500);
      }
    } else {
      recordDecisions([{ id: photo.id, status }], 'first');
    }
    state.priorityIds = [];
    render();
  };
  attachSwipe(frame, keepMark, dropMark, decide);

  let zoomed = false;
  const largestFace = [...mainFacesOf(photo)].sort((left, right) => right.box.h - left.box.h)[0];
  const toggleZoom = () => {
    zoomed = !zoomed;
    frame.classList.toggle('zoomed', zoomed);
    image.style.transformOrigin = largestFace ? `${(largestFace.box.x + largestFace.box.w / 2) * 100}% ${(largestFace.box.y + largestFace.box.h / 2) * 100}%` : '50% 50%';
    image.style.transform = zoomed ? 'scale(3)' : '';
  };
  frame.addEventListener('mousemove', (event) => {
    if (!zoomed) return;
    const rect = frame.getBoundingClientRect();
    image.style.transformOrigin = `${((event.clientX - rect.left) / rect.width) * 100}% ${((event.clientY - rect.top) / rect.height) * 100}%`;
  });
  frame.addEventListener('dblclick', toggleZoom);

  $('#controls').innerHTML = `
    <button class="button decide" id="act-drop"><i class="ph ph-x"></i>빼기<kbd>←</kbd></button>
    <button class="button icon-only" id="act-undo" title="되돌리기 (Z)" aria-label="되돌리기"><i class="ph ph-arrow-counter-clockwise"></i></button>
    <button class="button icon-only" id="act-zoom" title="얼굴 크게 보기 (Space)" aria-label="얼굴 크게 보기"><i class="ph ph-magnifying-glass-plus"></i></button>
    <button class="button icon-only" id="act-faces" title="얼굴 표시 (F)" aria-label="얼굴 표시"><i class="ph ph-user-focus"></i></button>
    <button class="button decide keep" id="act-keep"><i class="ph ph-circle"></i>${keepWord}<kbd>→</kbd></button>`;
  const leave = (status) => leaveFrame(frame, status === 'keep' ? keepMark : dropMark, status, decide);
  $('#act-drop').onclick = () => leave('drop');
  $('#act-keep').onclick = () => leave('keep');
  $('#act-undo').onclick = undoLast;
  $('#act-zoom').onclick = toggleZoom;
  $('#act-faces').onclick = () => frame.classList.toggle('show-faces');
  currentKeys = { left: () => leave('drop'), right: () => leave('keep'), zoom: toggleZoom, faces: () => frame.classList.toggle('show-faces') };
}

// 결정하면 그리스 펜슬 표시를 그린 뒤 사진을 옆으로 보낸다
function leaveFrame(frame, mark, status, onDone) {
  if (frame.dataset.leaving) return;
  frame.dataset.leaving = '1';
  mark.style.opacity = '';
  mark.classList.add('drawn');
  const direction = status === 'keep' ? 1 : -1;
  setTimeout(() => {
    frame.classList.add('leave');
    frame.style.transform = `translateX(${direction * 70}vw) rotate(${direction * 6}deg)`;
    frame.style.opacity = '0';
  }, 170);
  setTimeout(() => onDone(status), 400);
}

function attachSwipe(frame, keepMark, dropMark, onDecide) {
  let startX = 0, deltaX = 0, dragging = false;
  frame.addEventListener('pointerdown', (event) => {
    if (frame.classList.contains('zoomed') || event.button !== 0) return;
    dragging = true; startX = event.clientX; deltaX = 0;
    frame.setPointerCapture(event.pointerId);
    frame.classList.add('dragging'); frame.classList.remove('settle');
  });
  frame.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    deltaX = event.clientX - startX;
    frame.style.transform = `translateX(${deltaX}px) rotate(${deltaX / 40}deg)`;
    keepMark.style.opacity = clamp01(deltaX / 140);
    dropMark.style.opacity = clamp01(-deltaX / 140);
  });
  const finish = () => {
    if (!dragging) return;
    dragging = false;
    frame.classList.remove('dragging');
    if (Math.abs(deltaX) > 120) { leaveFrame(frame, deltaX > 0 ? keepMark : dropMark, deltaX > 0 ? 'keep' : 'drop', onDecide); return; }
    frame.classList.add('settle');
    frame.style.transform = '';
    keepMark.style.opacity = ''; dropMark.style.opacity = '';
  };
  frame.addEventListener('pointerup', finish);
  frame.addEventListener('pointercancel', finish);
}

// ---------- 비슷한 컷 묶음 ----------
let burstSelection = { key: null, selected: new Set() };

function renderBurst(stage, unit) {
  keepOnlyUrls([]);
  const members = unit.ids.map((id) => state.photoById.get(id)).sort((left, right) => right.score - left.score);
  if (burstSelection.key !== unit.key) {
    burstSelection = { key: unit.key, selected: new Set(state.settings.useScore ? members.filter((photo) => photo.groupBest).map((photo) => photo.id) : []) };
  }
  // 좁은 화면(휴대폰 세로)에서는 사진을 크게 보이게 열 수를 줄인다
  const isNarrow = matchMedia('(max-width: 760px)').matches;
  const columns = isNarrow ? (members.length <= 3 ? 1 : 2) : members.length <= 4 ? 2 : members.length <= 9 ? 3 : 4;
  stage.innerHTML = `
    <div class="burst">
      <div class="burst-head">
        <h2>비슷한 컷 ${members.length}장</h2>
        <span class="hint">남길 사진을 누르세요. 안 누른 사진은 빠져요.${state.settings.useScore ? ' 점수가 가장 높은 한 장을 미리 골라뒀어요.' : ''}</span>
        <span class="spacer"></span>
        <button class="button" id="burst-expand">한 장씩 보기</button>
      </div>
      <div class="burst-grid" style="--cols:${columns};--rows:${Math.min(3, Math.ceil(members.length / columns))}">
        ${members.map((photo, index) => `
          <button class="shot" data-id="${escapeHtml(photo.id)}" aria-pressed="${burstSelection.selected.has(photo.id)}" title="${escapeHtml(photo.relPath)}. 더블클릭하면 크게 봐요">
            <img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}" loading="lazy">
            ${index < 9 ? `<span class="index num">${index + 1}</span>` : ''}
            <span class="tags">${state.settings.useScore ? `<span class="tag num">점수 ${photo.score}</span>` : ''}${hasClosedEyes(photo) ? '<span class="tag flag"><i class="ph ph-eye-closed"></i> 눈 감음</span>' : ''}</span>
          </button>`).join('')}
      </div>
    </div>`;
  for (const shot of stage.querySelectorAll('.shot')) shot.append(cloneMark('keep'));
  const updateConfirm = () => { $('#act-confirm').innerHTML = `<i class="ph ph-circle"></i>${burstSelection.selected.size}장 남기기<kbd>→</kbd>`; };
  const toggleMember = (photoId) => {
    if (burstSelection.selected.has(photoId)) burstSelection.selected.delete(photoId); else burstSelection.selected.add(photoId);
    const shot = stage.querySelector(`.shot[data-id="${CSS.escape(photoId)}"]`);
    shot.setAttribute('aria-pressed', burstSelection.selected.has(photoId));
    $('.mark', shot).classList.toggle('drawn', burstSelection.selected.has(photoId));
    updateConfirm();
  };
  for (const shot of stage.querySelectorAll('.shot')) {
    shot.onclick = () => toggleMember(shot.dataset.id);
    shot.ondblclick = (event) => { event.preventDefault(); openViewer(state.photoById.get(shot.dataset.id)); };
  }
  $('#burst-expand').onclick = () => { state.expandedGroups.add(unit.key); buildUnits(); saveProjectSoon(); render(); };
  const confirm = () => {
    recordDecisions(members.map((photo) => ({ id: photo.id, status: burstSelection.selected.has(photo.id) ? 'keep' : 'drop' })), 'first-burst');
    burstSelection = { key: null, selected: new Set() };
    render();
  };
  const dropAll = () => { burstSelection.selected.clear(); confirm(); };
  $('#controls').innerHTML = `
    <button class="button decide" id="act-dropall"><i class="ph ph-x"></i>모두 빼기<kbd>←</kbd></button>
    <button class="button icon-only" id="act-undo" title="되돌리기 (Z)" aria-label="되돌리기"><i class="ph ph-arrow-counter-clockwise"></i></button>
    <button class="button decide keep" id="act-confirm"></button>`;
  updateConfirm();
  $('#act-dropall').onclick = dropAll;
  $('#act-confirm').onclick = confirm;
  $('#act-undo').onclick = undoLast;
  currentKeys = { left: dropAll, right: confirm, number: (digit) => { const photo = members[digit - 1]; if (photo) toggleMember(photo.id); } };
}

// ---------- 두 장 비교 ----------
function renderCompare(stage, [leftPhoto, rightPhoto]) {
  keepOnlyUrls([leftPhoto.id, rightPhoto.id]);
  stage.innerHTML = `<div class="compare">${[leftPhoto, rightPhoto].map((photo, index) => `
    <button class="shot" data-side="${index}" aria-label="${index === 0 ? '왼쪽' : '오른쪽'} 사진 남기기">
      <img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}">
      <span class="choose-key"><kbd>${index === 0 ? '←' : '→'}</kbd></span>
      <span class="tags">${photoTagsHtml(photo, { showAdded: true })}</span>
    </button>`).join('')}</div>`;
  const shots = stage.querySelectorAll('.shot');
  shots.forEach((shot, index) => {
    shot.append(cloneMark('keep'));
    fullImageUrl(index === 0 ? leftPhoto : rightPhoto).then((url) => { const image = $('img', shot); if (image.isConnected) image.src = url; });
  });
  const keepOne = (winnerIndex) => {
    const winner = winnerIndex === 0 ? leftPhoto : rightPhoto;
    const loser = winnerIndex === 0 ? rightPhoto : leftPhoto;
    $('.mark', shots[winnerIndex]).classList.add('drawn');
    setTimeout(() => {
      const undoEntry = recordDecisions([{ id: loser.id, status: 'drop' }], 'distill-compare');
      markRoundSeen('distill', [winner.id, loser.id], undoEntry);
      render();
    }, 260);
  };
  const keepBoth = () => {
    const undoEntry = emptyUndo();
    state.undoStack.push(undoEntry);
    markRoundSeen('distill', [leftPhoto.id], undoEntry);
    render();
  };
  shots[0].onclick = () => keepOne(0);
  shots[1].onclick = () => keepOne(1);
  $('#controls').innerHTML = `
    <button class="button decide" id="act-left">왼쪽 남기기<kbd>←</kbd></button>
    <button class="button decide" id="act-both">둘 다 남기기<kbd>↓</kbd></button>
    <button class="button icon-only" id="act-undo" title="되돌리기 (Z)" aria-label="되돌리기"><i class="ph ph-arrow-counter-clockwise"></i></button>
    <button class="button decide" id="act-right">오른쪽 남기기<kbd>→</kbd></button>`;
  $('#act-left').onclick = () => keepOne(0);
  $('#act-right').onclick = () => keepOne(1);
  $('#act-both').onclick = keepBoth;
  $('#act-undo').onclick = undoLast;
  currentKeys = { left: () => keepOne(0), right: () => keepOne(1), down: keepBoth };
}

// ---------- 점수 기준 패널 ----------
$('#criteria').addEventListener('toggle', (event) => {
  if (event.newState !== 'open') return;
  const model = activeModel();
  const totalPositive = model.theta.reduce((sum, weight) => sum + Math.max(0, weight), 0) || 1;
  const currentPhoto = currentPhotoOnStage();
  const sourceText = model.source === 'learned'
    ? `${escapeHtml(reviewer().name)}의 결정 ${model.trainedOn}건으로 배운 기준이에요. 결정이 ${LEARN_EVERY}건 쌓일 때마다 다시 배워요.`
    : model.source === 'server'
      ? `다른 사용자들의 ${preset().label} 사진 결정으로 만든 기본 기준이에요. 결정이 ${LEARN_MIN_DECISIONS}건 쌓이면 내 취향으로 다시 맞춰요.`
      : `${preset().label} 기본 기준이에요. 남김과 뺌이 각각 ${LEARN_MIN_PER_CLASS}건 이상, 합쳐서 ${LEARN_MIN_DECISIONS}건이 쌓이면 내 취향으로 다시 맞춰요.`;
  $('#criteria').innerHTML = `
    <h2>점수 기준</h2>
    <p class="hint source">${sourceText}</p>
    <table class="num">
      <tr><td class="hint">항목</td><td class="hint">비중</td>${currentPhoto ? '<td class="hint">이 사진</td>' : ''}</tr>
      ${FEATURE_KEYS.map((key, index) => `<tr><td>${FEATURE_LABELS[key]}</td><td>${Math.round((Math.max(0, model.theta[index]) / totalPositive) * 100)}%</td>${currentPhoto?.features ? `<td>${Math.round(currentPhoto.features[key] * 100)}</td>` : ''}</tr>`).join('')}
    </table>
    <p class="hint source">눈 감음은 점수에 넣지 않고 표시만 해요. 선명도와 노이즈는 이 폴더 사진들 사이의 순위로 계산해요.</p>`;
});
function currentPhotoOnStage() {
  return $('#stage .frame') && stagePhotoId ? state.photoById.get(stagePhotoId) : null;
}

// ---------- 크게 보기 ----------
async function openViewer(photo) {
  const viewer = $('#viewer');
  const image = $('img', viewer) ?? viewer.insertAdjacentElement('afterbegin', document.createElement('img'));
  image.alt = photo.name;
  image.src = photo.thumbUrl;
  viewer.showModal();
  const url = URL.createObjectURL(await getPhotoFile(photo));
  image.src = url;
  viewer.dataset.url = url;
}
$('#viewer').addEventListener('close', () => { const viewer = $('#viewer'); if (viewer.dataset.url) URL.revokeObjectURL(viewer.dataset.url); viewer.dataset.url = ''; });
$('#viewer').onclick = () => $('#viewer').close();

// ---------- 키보드 ----------
document.addEventListener('keydown', (event) => {
  if (event.target.matches('input:not([type=checkbox]):not([type=radio]), textarea, select')) return;
  if ($('#viewer').open || !PICK_VIEWS.includes(state.view)) return;
  if (event.key === 'z' || event.key === 'Z') { event.preventDefault(); undoLast(); return; }
  if (!currentKeys) return;
  const actions = { ArrowLeft: currentKeys.left, ArrowRight: currentKeys.right, ArrowDown: currentKeys.down, ' ': currentKeys.zoom, f: currentKeys.faces, F: currentKeys.faces };
  if (actions[event.key]) { event.preventDefault(); actions[event.key](); return; }
  if (/^[1-9]$/.test(event.key)) currentKeys.number?.(Number(event.key));
});

/* =====================================================================
   내보내기
   ===================================================================== */
let exportBasis = 'current'; // current | both | either

function selectionIds(basis) {
  const keepA = new Set(keepIds('a')), keepB = new Set(keepIds('b'));
  if (basis === 'both') return state.photos.filter((photo) => keepA.has(photo.id) && keepB.has(photo.id)).map((photo) => photo.id);
  if (basis === 'either') return state.photos.filter((photo) => keepA.has(photo.id) || keepB.has(photo.id)).map((photo) => photo.id);
  return keepIds();
}

function selectionListText(ids) {
  const photos = ids.map((id) => state.photoById.get(id));
  const closedEyePhotos = photos.filter(hasClosedEyes);
  const lines = [`모먼트 셀렉 목록, ${photos.length}장`, `폴더: ${state.folderName}`, `만든 시각: ${formatKoreanTime(Date.now())}`, '', ...photos.map((photo) => photo.relPath)];
  if (closedEyePhotos.length) lines.push('', `눈 감음 의심 ${closedEyePhotos.length}장${state.settings.preset === 'wedding' ? ' (합성 요청 후보)' : ''}`, ...closedEyePhotos.map((photo) => photo.relPath));
  return lines.join('\n');
}

function renderExportView() {
  const wrap = $('#export');
  const keepA = keepIds('a'), keepB = keepIds('b');
  const keepASet = new Set(keepA), keepBSet = new Set(keepB);
  const bothCount = keepA.filter((id) => keepBSet.has(id)).length;
  const onlyA = keepA.length - bothCount, onlyB = keepB.filter((id) => !keepASet.has(id)).length;
  const bothDecided = Object.keys(state.reviewers.a.decisions).length > 0 && Object.keys(state.reviewers.b.decisions).length > 0;
  if (!bothDecided && exportBasis !== 'current') exportBasis = 'current';
  const ids = selectionIds(exportBasis);
  const closedEyeCount = ids.filter((id) => hasClosedEyes(state.photoById.get(id))).length;
  const nameA = escapeHtml(state.reviewers.a.name), nameB = escapeHtml(state.reviewers.b.name);
  const target = state.settings.target;

  wrap.innerHTML = `
    <h1>내보내기</h1>
    <p class="summary num">${bothDecided
      ? `${nameA} <b>${keepA.length}장</b>, ${nameB} <b>${keepB.length}장</b>을 남겼어요. 둘 다 남긴 사진은 <b>${bothCount}장</b>, 한 명만 남긴 사진은 ${onlyA + onlyB}장(${nameA} ${onlyA}, ${nameB} ${onlyB})이에요.`
      : `${nameA} 기준으로 <b>${keepA.length}장</b>을 남겼어요.`}</p>

    <section>
      <h2>원본 저장</h2>
      <p class="hint">${canPickFolder
        ? '고른 사진의 원본을 새 폴더로 복사하고, 같은 폴더에 파일 목록을 남겨요. 원래 폴더는 건드리지 않아요.'
        : canShareFiles
          ? `공유 창을 ${SHARE_BATCH_SIZE}장씩 열어요. 공유 창에서 이미지 저장을 누르면 사진 보관함에 저장돼요.`
          : '고른 사진을 한 장씩 내려받아요.'}</p>
      <div class="row">
        ${bothDecided ? `<div class="segmented" id="basis-options" role="radiogroup" aria-label="내보낼 사진">
          <label><input type="radio" name="basis" value="current" ${exportBasis === 'current' ? 'checked' : ''}>${escapeHtml(reviewer().name)} 기준</label>
          <label><input type="radio" name="basis" value="both" ${exportBasis === 'both' ? 'checked' : ''}>둘 다 남긴 사진</label>
          <label><input type="radio" name="basis" value="either" ${exportBasis === 'either' ? 'checked' : ''}>한 명이라도 남긴 사진</label>
        </div>` : ''}
      </div>
      <div class="row">
        ${canPickFolder ? `<button class="button primary" id="export-copy" ${ids.length ? '' : 'disabled'}><i class="ph ph-folder-simple-plus"></i>${ids.length}장 복사할 폴더 선택</button>` : ''}
        ${!canPickFolder && canShareFiles ? `<button class="button primary" id="export-share" ${ids.length ? '' : 'disabled'}><i class="ph ph-share-network"></i>${shareButtonLabel(ids)}</button>` : ''}
        ${!canPickFolder ? `<button class="button${canShareFiles ? '' : ' primary'}" id="export-download" ${ids.length ? '' : 'disabled'}><i class="ph ph-download-simple"></i>한 장씩 내려받기</button>` : ''}
        <button class="button" id="export-list" ${ids.length ? '' : 'disabled'}><i class="ph ph-copy"></i>파일 목록 복사</button>
        <span class="hint num" id="export-progress"></span>
      </div>
      <div class="row">
        <label for="export-target" class="hint">남길 장수</label><input type="number" id="export-target" min="1" value="${target}">
        <span class="hint">${ids.length > target ? `지금 ${ids.length - target}장 많아요.` : ids.length < target ? `${target - ids.length}장 더 남길 수 있어요.` : '딱 맞아요.'}${closedEyeCount ? ` 눈 감음 의심 ${closedEyeCount}장은 목록 파일에 따로 적어둬요.` : ''}</span>
      </div>
    </section>

    <section>
      <h2>둘이 고른 결과 합치기</h2>
      <p class="hint">같은 기기라면 위쪽에서 사람을 바꿔 고르면 돼요. 서로 다른 기기에서 골랐다면 한 사람이 결정 파일을 저장해 보내고, 다른 사람이 같은 사진을 연 상태에서 불러오면 됩니다. 결정은 파일 이름으로 맞추니 같은 원본을 써야 해요.</p>
      <div class="row">
        <input type="text" id="rename-a" value="${nameA}" style="width:120px" aria-label="첫 번째 사람 이름">
        <input type="text" id="rename-b" value="${nameB}" style="width:120px" aria-label="두 번째 사람 이름">
        <button class="button" id="decisions-save"><i class="ph ph-download-simple"></i>${escapeHtml(reviewer().name)} 결정 파일 저장</button>
        <button class="button" id="decisions-load"><i class="ph ph-upload-simple"></i>다른 사람 결정 파일 불러오기</button>
        <input type="file" id="decisions-file" accept="application/json" hidden>
      </div>
    </section>

    <section>
      <h2>고른 사진 <span class="hint num">${ids.length}장</span></h2>
      ${ids.length ? '' : '<p class="hint">아직 남긴 사진이 없어요. 1차 셀렉에서 오른쪽으로 넘기면 여기에 모여요.</p>'}
      <div class="contact">
        ${ids.map((id) => { const photo = state.photoById.get(id); const decision = decisionOf(id); return `
          <figure data-id="${escapeHtml(id)}">
            <button class="thumb" aria-label="${escapeHtml(photo.name)} 크게 보기"><img src="${photo.thumbUrl}" alt="" loading="lazy"></button>
            <figcaption><span title="${escapeHtml(photo.relPath)}">${escapeHtml(photo.name)}</span>${hasClosedEyes(photo) ? '<i class="ph ph-eye-closed flagged" title="눈 감음 의심"></i>' : ''}${decision?.addedAt ? '<i class="ph ph-arrow-u-up-left" title="다시 남긴 사진"></i>' : ''}${exportBasis === 'current' ? '<button class="remove" title="빼기" aria-label="빼기"><i class="ph ph-x"></i></button>' : ''}</figcaption>
          </figure>`; }).join('')}
      </div>
    </section>`;

  if ($('#basis-options')) $('#basis-options').onchange = (event) => { exportBasis = event.target.value; render(); };
  $('#export-target').onchange = (event) => { state.settings.target = Math.max(1, parseInt(event.target.value, 10) || 1); saveProjectSoon(); render(); };
  $('#export-list').onclick = async () => { await navigator.clipboard.writeText(selectionListText(ids)); showToast('파일 목록을 복사했어요.'); };
  if ($('#export-copy')) $('#export-copy').onclick = () => copySelection(ids);
  if ($('#export-download')) $('#export-download').onclick = () => downloadSelection(ids);
  if ($('#export-share')) $('#export-share').onclick = () => shareNextBatch(ids);
  for (const figure of wrap.querySelectorAll('.contact figure')) {
    $('.thumb', figure).onclick = () => openViewer(state.photoById.get(figure.dataset.id));
    const removeButton = $('.remove', figure);
    if (removeButton) removeButton.onclick = () => { recordDecisions([{ id: figure.dataset.id, status: 'drop' }], 'export-remove'); render(); };
  }
  $('#rename-a').onchange = (event) => { state.reviewers.a.name = event.target.value.trim() || '나'; saveProjectSoon(); render(); };
  $('#rename-b').onchange = (event) => { state.reviewers.b.name = event.target.value.trim() || '파트너'; saveProjectSoon(); render(); };
  $('#decisions-save').onclick = () => {
    const payload = { app: 'moment-select', version: 2, folder: state.folderName, reviewer: reviewer().name, exportedAt: Date.now(), decisions: reviewer().decisions };
    downloadBlob(new Blob([JSON.stringify(payload)], { type: 'application/json' }), `셀렉결정_${reviewer().name}.json`);
  };
  $('#decisions-load').onclick = () => $('#decisions-file').click();
  $('#decisions-file').onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (payload.app !== 'moment-select') throw new Error('모먼트 셀렉에서 저장한 결정 파일이 아니에요.');
      const otherKey = state.currentReviewer === 'a' ? 'b' : 'a';
      const matched = Object.fromEntries(Object.entries(payload.decisions).filter(([id]) => state.photoById.has(id)));
      state.reviewers[otherKey].decisions = matched;
      if (payload.reviewer) state.reviewers[otherKey].name = payload.reviewer;
      saveProjectSoon();
      showToast(`${escapeHtml(state.reviewers[otherKey].name)}의 결정 ${Object.keys(matched).length}건을 불러왔어요.${Object.keys(matched).length < Object.keys(payload.decisions).length ? ' 이 폴더에 없는 사진의 결정은 건너뛰었어요.' : ''}`);
      render();
    } catch (error) {
      showToast(`결정 파일을 불러오지 못했어요. ${escapeHtml(error.message)}`);
    }
  };
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = Object.assign(document.createElement('a'), { href: url, download: fileName });
  document.body.appendChild(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
function timestampForFolder() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
}

async function copySelection(ids) {
  const progress = $('#export-progress');
  const photos = ids.map((id) => state.photoById.get(id));
  const outputName = uniqueOutputNames(photos); // 이름이 겹치면 폴더 경로를 붙인다
  maybeSendStats(true);

  try {
    const destination = await window.showDirectoryPicker({ id: 'moment-select-export', mode: 'readwrite' });
    const folder = await destination.getDirectoryHandle(`${EXPORT_FOLDER_PREFIX}${timestampForFolder()}`, { create: true });
    for (const [index, photo] of photos.entries()) {
      const source = await getPhotoFile(photo);
      const writable = await (await folder.getFileHandle(outputName(photo), { create: true })).createWritable();
      await source.stream().pipeTo(writable); // 큰 원본도 메모리에 다 올리지 않고 흘려서 복사
      progress.textContent = `${index + 1} / ${photos.length}장 복사 중`;
    }
    const listWritable = await (await folder.getFileHandle('셀렉목록.txt', { create: true })).createWritable();
    await listWritable.write(selectionListText(ids));
    await listWritable.close();
    progress.textContent = '';
    showToast(`${photos.length}장을 '${escapeHtml(destination.name)}/${escapeHtml(folder.name)}' 폴더에 복사했어요.`, [], 6000);
  } catch (error) {
    progress.textContent = '';
    if (error.name !== 'AbortError') showToast(`복사하지 못했어요. ${escapeHtml(error.message)} 저장 공간과 폴더 권한을 확인해주세요.`);
  }
}

// ---------- 폴더에 쓸 수 없는 환경의 저장 ----------
function uniqueOutputNames(photos) {
  const nameCounts = {};
  for (const photo of photos) nameCounts[photo.name] = (nameCounts[photo.name] ?? 0) + 1;
  return (photo) => (nameCounts[photo.name] > 1 ? photo.relPath.replaceAll('/', '_').replace(/ \((\d+)\)$/, '_$1') : photo.name);
}

async function downloadSelection(ids) {
  const progress = $('#export-progress');
  const photos = ids.map((id) => state.photoById.get(id));
  const outputName = uniqueOutputNames(photos);
  for (const [index, photo] of photos.entries()) {
    downloadBlob(await getPhotoFile(photo), outputName(photo));
    progress.textContent = `${index + 1} / ${photos.length}장 받는 중`;
    await new Promise((resolve) => setTimeout(resolve, 350)); // 브라우저가 연속 다운로드를 막지 않게 간격을 둔다
  }
  downloadBlob(new Blob([selectionListText(ids)], { type: 'text/plain' }), '셀렉목록.txt');
  progress.textContent = '';
}

// 공유 시트 저장: 공유 창은 버튼을 누를 때마다 한 번만 열 수 있어서 묶음 단위로 이어서 연다
let shareProgress = { signature: '', nextIndex: 0 }; // 지금 저장 중인 목록과 다음에 저장할 위치
function shareButtonLabel(ids) {
  const signature = ids.join('|');
  if (shareProgress.signature !== signature) shareProgress = { signature, nextIndex: 0 };
  if (shareProgress.nextIndex >= ids.length) return '모두 저장했어요. 처음부터 다시 저장';
  const end = Math.min(ids.length, shareProgress.nextIndex + SHARE_BATCH_SIZE);
  return `${shareProgress.nextIndex + 1}~${end}번째 사진 저장 (전체 ${ids.length}장)`;
}
async function shareNextBatch(ids) {
  shareButtonLabel(ids);
  if (shareProgress.nextIndex >= ids.length) shareProgress.nextIndex = 0;
  const batch = ids.slice(shareProgress.nextIndex, shareProgress.nextIndex + SHARE_BATCH_SIZE).map((id) => state.photoById.get(id));
  const outputName = uniqueOutputNames(ids.map((id) => state.photoById.get(id)));
  try {
    const files = await Promise.all(batch.map(async (photo) => {
      const file = await getPhotoFile(photo);
      return new File([file], outputName(photo), { type: file.type || 'image/jpeg' });
    }));
    await navigator.share({ files });
    shareProgress.nextIndex += batch.length;
  } catch (error) {
    if (error.name !== 'AbortError') showToast(`공유 창을 열지 못했어요. ${escapeHtml(error.message)} 한 장씩 내려받기를 써주세요.`);
  }
  render();
}

// 설치형 웹앱: 한 번 열고 나면 인터넷 없이도 동작하게 앱 파일을 기기에 보관한다
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  navigator.serviceWorker.register('./sw.js').catch((error) => console.warn('service worker', error));
}
render();
