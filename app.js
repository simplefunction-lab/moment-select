/* =====================================================================
   모먼트 셀렉
   - 폴더의 원본을 그대로 두고 File System Access API로 읽고 쓴다
   - 얼굴 검출은 브라우저 안에서 face-api(TensorFlow.js)로 돌린다
   - 점수는 10개 항목(선명도, 노출, 표정, 눈 뜸, 미적 점수, 주인공 등)의 로지스틱 모델
   - 처음엔 사진 종류별 기본값, 고르는 동안 남김/뺌과 비슷한 컷 비교로 사람마다 다시 학습
   ===================================================================== */

import { t, num, getLanguage, setLanguage, applyStaticText, presetText, featureLabel, categoryLabel, allExportPrefixes, isDefaultName } from './i18n.js';
import { buildAestheticModel, scoreAesthetic, AESTHETIC_INPUT } from './aesthetic.js';
// 스와이프 물리(스프링)와 목표 장수 축하. 둘 다 앱과 함께 배포한다 (vendor/LICENSES.txt)
import { animate } from './vendor/motion/motion-animate.mjs';
import { create as createConfetti } from './vendor/confetti/confetti.module.mjs';
// 움직임 줄이기 설정이면 JS 애니메이션도 바로 끝낸다 (CSS 쪽은 styles.css의 prefers-reduced-motion 규칙)
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const motion = (element, keyframes, options) => animate(element, keyframes, reducedMotion.matches ? { duration: 0 } : options);

// ---------- 라이브러리 위치 (모두 앱과 함께 배포, 외부 요청 없음) ----------
const FACE_API_URL = './vendor/face-api/face-api.esm.js';
const FACE_MODEL_BASE = './vendor/face-api/model/';
const NIMA_MANIFEST_URL = './vendor/nima/nima-aesthetic.json';
const NIMA_WEIGHTS_URL = './vendor/nima/nima-aesthetic.bin';
const EXIFR_URL = './vendor/exifr/lite.esm.mjs';
// MediaPipe Face Landmarker: 눈 감음과 웃음을 정밀하게 본다 (눈 감은 사진 감점을 켰을 때만 받는다)
const MEDIAPIPE_URL = './vendor/mediapipe/vision_bundle.mjs';
const MEDIAPIPE_WASM_LOADER_URL = './vendor/mediapipe/wasm/vision_wasm_internal.js';
const MEDIAPIPE_WASM_BINARY_URL = './vendor/mediapipe/wasm/vision_wasm_internal.wasm';
const FACE_LANDMARKER_URL = './vendor/mediapipe/face_landmarker.task';
// 분석 전에 받는 모델. 사용자가 무엇을 받는지 알 수 있게 용도(purposeKey)를 함께 보여준다.
// bytes는 진행 막대 계산용 파일 크기 (모델 파일을 바꾸면 같이 고친다)
// onlyWhen: 이 설정이 켜져 있을 때만 받는다
const MODEL_DOWNLOADS = [
  { purposeKey: 'modelPurposeFaces', files: [
    { url: FACE_API_URL, bytes: 1333462 },
    { url: `${FACE_MODEL_BASE}tiny_face_detector_model-weights_manifest.json`, bytes: 3219 },
    { url: `${FACE_MODEL_BASE}tiny_face_detector_model.bin`, bytes: 193321 },
    { url: `${FACE_MODEL_BASE}face_landmark_68_model-weights_manifest.json`, bytes: 8485 },
    { url: `${FACE_MODEL_BASE}face_landmark_68_model.bin`, bytes: 356840 },
    { url: `${FACE_MODEL_BASE}face_expression_model-weights_manifest.json`, bytes: 6980 },
    { url: `${FACE_MODEL_BASE}face_expression_model.bin`, bytes: 329468 },
  ] },
  { purposeKey: 'modelPurposePeople', files: [
    { url: `${FACE_MODEL_BASE}face_recognition_model-weights_manifest.json`, bytes: 19615 },
    { url: `${FACE_MODEL_BASE}face_recognition_model.bin`, bytes: 6444032 },
  ] },
  { purposeKey: 'modelPurposeAesthetic', files: [
    { url: NIMA_MANIFEST_URL, bytes: 4873 },
    { url: NIMA_WEIGHTS_URL, bytes: 6434472 },
  ] },
  { purposeKey: 'modelPurposeEyes', onlyWhen: 'penalizeClosedEyes', files: [
    { url: MEDIAPIPE_URL, bytes: 156139 },
    { url: MEDIAPIPE_WASM_LOADER_URL, bytes: 334128 },
    { url: MEDIAPIPE_WASM_BINARY_URL, bytes: 13519922 },
    { url: FACE_LANDMARKER_URL, bytes: 3758596 },
  ] },
];
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
const ANALYSIS_LONG_SIDE = 1024;          // 분석용으로 줄일 긴 변
const THUMB_LONG_SIDE = 480;              // 썸네일 긴 변
const MAIN_FACE_MIN_HEIGHT = 0.035;       // 사진 높이 대비 이보다 작은 얼굴은 배경 인물
const EYE_CLOSED_EAR = 0.21;              // 두 눈 모두 이보다 작으면 눈 감음 의심
const EYE_CHECK_MIN_FACE_PX = 60;         // 얼굴이 이보다 작으면 눈을 판정하지 않음
const EYE_CHECK_MAX_YAW = 0.22;           // 고개가 이보다 돌아가 있으면 눈을 판정하지 않음 (옆얼굴)
const EYE_CHECK_MIN_CONFIDENCE = 0.6;     // 얼굴 검출 확신이 이보다 낮으면 눈을 판정하지 않음
const EYE_BLINK_CLOSED = 0.5;             // 정밀 모델의 eyeBlinkLeft, eyeBlinkRight가 둘 다 이 이상이면 눈 감음
const EYE_MESH_MIN_FACE_PX = 40;          // 정밀 모델로 볼 얼굴의 최소 높이 (분석용 크기 기준)
const EYE_MESH_MAX_FACES = 6;             // 사진마다 정밀 모델로 볼 큰 얼굴 수
const EYE_MESH_CROP = 1.8;                // 얼굴 상자의 이 배수만큼 잘라서 넘긴다
const FOCUS_GRID = 4;                     // 초점 부분을 찾으려고 사진을 4x4 칸으로 나눈다
const FOCUS_TOP_TILES = 3;                // 가장 선명한 3칸 평균을 '초점 맞은 부분 선명도'로 쓴다
const SIMILAR_HASH_DISTANCE = 12;       // dHash 차이가 이 이하면 바로 비슷한 컷
const SIMILAR_HASH_LOOSE = 22;          // 이 이하이면서 작은 흑백 이미지 상관도가 높으면 비슷한 컷 (구도가 조금 바뀐 경우)
const SIMILAR_CORRELATION = 0.6;        // 16x12 흑백 이미지 상관도 기준
const SIMILAR_TIME_GAP_MS = 90_000;     // 촬영 시각이 이 간격 안이어야 같은 묶음 (시각 정보가 있을 때)
const SIMILAR_WINDOW = 8;               // 촬영 순서로 앞쪽 몇 장까지 비교할지
const GROUP_MAX_SIZE = 12;
const RECOMMEND_RATIO = 0.4;              // 점수 상위 40% 단위를 먼저 보여준다
const SCORE_SCALE = 8;                    // 가중합(0~1)을 로지스틱 모델로 바꿀 때의 기울기
const LEARN_MIN_DECISIONS = 20;           // 이만큼 고르면 학습을 시작
const LEARN_MIN_PER_CLASS = 5;            // 남김과 뺌이 각각 이만큼은 있어야 학습
const LEARN_EVERY = 10;                   // 새 결정이 이만큼 쌓일 때마다 다시 학습
const CACHE_VERSION = 'v5';             // 분석 방식이 바뀌면 올린다 (v5: 같은 사람 알아보기, 미적 점수)
const APP_VERSION = 'v13';              // 앱 버전. 배포할 때마다 sw.js의 APP_VERSION과 같이 올린다
const ANALYZE_SHEET_MAX = 120;           // 분석 화면에 동시에 보여줄 썸네일 수 (1,000장 넘게 불러와도 화면 요소가 계속 늘지 않게)
const ETA_WARMUP_PHOTOS = 3;             // 이만큼 분석하기 전에는 남은 시간을 보여주지 않는다 (첫 장은 모델 준비로 느리다)
const CROSS_CHECK_SAMPLES = 6;           // WebGL과 CPU 결과를 비교해 볼 사진 수
const DARK_RETRY_LUMINANCE = 100;        // 평균 밝기가 이보다 어두운데 얼굴을 못 찾으면 밝게 보정해서 한 번 더 찾는다
const DARK_RETRY_THRESHOLD = 0.35;       // 다시 찾을 때의 검출 기준 (기본 0.45보다 조금 너그럽게)
const NOISE_LAPLACIAN_GAIN = 0.40625;      // 노이즈 표준편차 σ가 흐린 라플라시안 분산에 더하는 양은 약 0.406σ² (커널 계수 제곱합)
// 같은 사람 알아보기 (주인공)
const DESCRIPTOR_MAX_FACES = 4;          // 사진마다 큰 얼굴 몇 개까지 '누구인지' 계산할지 (얼굴 인식은 무거워서)
const SAME_PERSON_DISTANCE = 0.55;       // 얼굴 특징 거리(유클리드)가 이보다 가까우면 같은 사람 (face-api 권장 0.6보다 조금 엄격)
const CAST_MIN_SHARE = 0.08;             // 얼굴 있는 사진 중 이 비율 이상 나와야 주인공 후보로 자동 선택
const CAST_MIN_PHOTOS = 3;               // 그리고 최소 이만큼은 나와야 한다
const CAST_CANDIDATES = 8;               // 주인공 고르기 창에 보여줄 사람 수
// 장면 (줄이기 단계에서 장면이 통째로 사라지지 않게)
const SCENE_GAP_MS = 10 * 60_000;        // 촬영 시각이 이만큼 벌어지면 다른 장면
// 취향을 빨리 배우려고 애매한 사진을 섞어 보여주기 (능동 학습)
const PROBE_EVERY = 5;                   // 1차 셀렉에서 이 순번마다 한 번 (5번째, 10번째, ...)
const PROBE_UNTIL_ACTIONS = 40;          // 1차 셀렉에서 이만큼 결정하면 그만 섞는다 (비교 쌍은 묶음 하나에 여러 개가 쌓여 기준으로 쓰지 않는다)
const LEARN_MIN_PAIRS = 8;               // 비슷한 컷 비교 쌍만으로 학습을 시작할 수 있는 최소 쌍 수

// ---------- 점수 항목과 사진 종류별 기본 가중치 ----------
// 가중치는 출발값일 뿐이며 검증된 값이 아니다. 고르는 동안 사람마다 학습해서 바뀐다
// aesthetic: 미적 점수 (NIMA, 이 폴더 안 백분위), cast: 주인공이 나온 정도
const FEATURE_KEYS = ['subjectSharp', 'focusSharp', 'exposure', 'expression', 'smile', 'framing', 'clean', 'eyesOpen', 'aesthetic', 'cast'];
const EYES_OPEN_WEIGHT = 0.15; // '눈 감은 사진 감점'을 켰을 때 눈 뜸 항목의 기본 비중 (나머지 항목과 합쳐 1로 맞춘다)
// 사진 종류 이름과 설명 문구는 i18n.js에 있다. castCount: 주인공을 자동으로 몇 명 고를지
const PRESETS = {
  wedding: { crowdMin: 5, castCount: 2, target: 30,
    weights: { subjectSharp: .34, focusSharp: .10, exposure: .20, expression: .16, smile: .05, framing: .10, clean: .05, aesthetic: .14, cast: .14 } },
  travel: { crowdMin: 5, castCount: 1, target: 20, // 인스타그램 캐러셀 한 번에 올릴 수 있는 최대 장수
    weights: { subjectSharp: .24, focusSharp: .22, exposure: .22, expression: .10, smile: .08, framing: .06, clean: .08, aesthetic: .20, cast: .06 } },
  event: { crowdMin: 6, castCount: 1, target: 30,
    weights: { subjectSharp: .30, focusSharp: .08, exposure: .17, expression: .15, smile: .18, framing: .07, clean: .05, aesthetic: .12, cast: .10 } },
  family: { crowdMin: 5, castCount: 2, target: 30,
    weights: { subjectSharp: .36, focusSharp: .08, exposure: .16, expression: .12, smile: .18, framing: .06, clean: .04, aesthetic: .12, cast: .14 } },
  other: { crowdMin: 5, castCount: 1, target: 30,
    weights: { subjectSharp: .28, focusSharp: .18, exposure: .20, expression: .12, smile: .08, framing: .07, clean: .07, aesthetic: .16, cast: .06 } },
};
// 시작 화면의 사진 종류 순서와 기본값. 해외에서는 원본 수백 장에서 고객이 직접 고르는 웨딩이 드물어서 여행을 앞에 둔다
const PRESET_ORDER = { en: ['travel', 'event', 'family', 'wedding', 'other'], ko: ['wedding', 'travel', 'event', 'family', 'other'] };
const presetOrder = () => PRESET_ORDER[getLanguage()] ?? PRESET_ORDER.en;
// 가중합 w·x 를 로지스틱 모델 sigmoid(bias + θ·x) 로 옮긴다. 순서는 그대로 유지된다
function modelFromWeights(weights, source, penalizeClosedEyes) {
  // 눈 감음 감점을 끄면 눈 뜸 항목 비중은 0. 켜면 기본 비중을 더하고 전체 합을 1로 다시 맞춘다
  const mergedWeights = { ...weights, eyesOpen: penalizeClosedEyes ? EYES_OPEN_WEIGHT : 0 };
  const totalWeight = Object.values(mergedWeights).reduce((sum, weight) => sum + weight, 0) || 1;
  return { theta: FEATURE_KEYS.map((key) => SCORE_SCALE * ((mergedWeights[key] ?? 0) / totalWeight)), bias: -SCORE_SCALE * 0.5, source, trainedOn: 0 };
}

// ---------- 전체 상태 ----------
const state = {
  folderName: '',                // 화면과 파일 목록에 보이는 폴더 이름
  projectKey: '',                // 결정을 저장하는 이름 (언어와 상관없다)
  photos: [],                    // 사진 목록 (분석 결과 포함)
  photoById: new Map(),          // 사진 id(폴더 안 상대 경로)로 찾기
  groups: new Map(),             // 묶음 id에서 사진 id 배열로
  units: [],                     // 1차 셀렉 단위 (낱장 또는 묶음)
  expandedGroups: new Set(),     // '한 장씩 보기'로 푼 묶음
  settings: { preset: 'wedding', target: 30, useCategories: true, crowdMin: 5, useScore: true, penalizeClosedEyes: true, share: false },
  reviewers: { a: { name: t('defaultNameA'), decisions: {} }, b: { name: t('defaultNameB'), decisions: {} } }, // decisions: 사진 id에서 {status, at, addedAt, log}
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
  people: [],                    // 같은 사람끼리 묶은 얼굴 [{key, faces:[{photoId, faceIndex}], photoIds:Set}] 많이 나온 순
  castKeys: null,                // 주인공으로 고른 사람 key 목록 (null이면 아직 안 정해서 자동으로 고른다)
  scenes: new Map(),             // 사진 id에서 장면 번호로 (촬영 시각 간격으로 나눔)
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

const formatTime = (ms) => t('formatTime', ms); // 언어에 맞는 날짜와 시각 표기
const isExportFolder = (name) => allExportPrefixes().some((prefix) => name.startsWith(prefix)); // 지난번에 내보낸 폴더인지

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
  const mark = $(kind === 'keep' ? '#mark-template' : '#cross-template').content.firstElementChild.cloneNode(true);
  $('.word', mark).textContent = t(kind === 'keep' ? 'markKeep' : 'markDrop'); // 템플릿 안은 자동 번역이 안 되니 직접 넣는다
  return mark;
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
function saveProjectNow() {
  clearTimeout(saveTimer);
  if (!state.projectKey) return Promise.resolve(false);
  return dbSet(`project:${state.projectKey}`, {
    settings: state.settings, reviewers: state.reviewers, models: state.models, rounds: state.rounds,
    dividerSeen: state.dividerSeen, sentSampleCount: state.sentSampleCount, expandedGroups: [...state.expandedGroups],
    castKeys: state.castKeys,
  });
}
function saveProjectSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveProjectNow, 300);
}

/* =====================================================================
   시작 화면
   ===================================================================== */
let presetTouched = false; // 사용자가 사진 종류를 직접 골랐는지 (안 골랐으면 언어를 바꿀 때 기본값도 바꾼다)
let targetTouched = false; // 사용자가 남길 장수를 직접 바꿨는지 (안 바꿨으면 사진 종류마다 기본 장수를 쓴다)
function renderPresetOptions() {
  $('#preset-options').innerHTML = presetOrder().map((key) =>
    `<label><input type="radio" name="preset" value="${key}" ${key === state.settings.preset ? 'checked' : ''}>${presetText(key).label}</label>`).join('');
  $('#preset-hint').textContent = presetText(state.settings.preset).hint;
  renderTargetHint();
}
// 여행에서 20장이면 '인스타그램 캐러셀 한 번 분량'이라고 알려준다
function renderTargetHint() {
  const carousel = state.settings.preset === 'travel' && parseInt($('#target-input').value, 10) === PRESETS.travel.target;
  $('#target-hint').textContent = t(carousel ? 'targetHintCarousel' : 'targetHint');
}
function choosePreset(presetKey) {
  state.settings.preset = presetKey;
  $('#crowd-input').value = preset().crowdMin;
  if (!targetTouched) $('#target-input').value = preset().target;
  renderPresetOptions();
}
$('#preset-options').onchange = (event) => { presetTouched = true; choosePreset(event.target.value); };
$('#target-input').addEventListener('input', () => { targetTouched = true; renderTargetHint(); });
choosePreset(presetOrder()[0]);
$('#category-input').onchange = (event) => $('#crowd-row').classList.toggle('hidden', !event.target.checked);
if (COLLECT_ENDPOINT) $('#share-field').classList.remove('hidden');

function applySettingsFromForm() {
  state.settings.target = Math.max(1, parseInt($('#target-input').value, 10) || 30);
  state.settings.useCategories = $('#category-input').checked;
  state.settings.crowdMin = Math.max(3, parseInt($('#crowd-input').value, 10) || 5);
  state.settings.useScore = $('#score-input').checked;
  state.settings.penalizeClosedEyes = $('#eyes-input').checked;
  state.settings.share = Boolean(COLLECT_ENDPOINT) && $('#share-input').checked;
  state.reviewers.a.name = $('#name-a').value.trim() || t('defaultNameA');
  state.reviewers.b.name = $('#name-b').value.trim() || t('defaultNameB');
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
      if (isExportFolder(entryName)) continue; // 지난번에 내보낸 폴더는 다시 읽지 않음
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
  const fallback = modelFromWeights(preset().weights, 'preset', state.settings.penalizeClosedEyes);
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

// source: 다음에 열 때 이어서 고를 수 있게 기억할 정보 {kind: handle | folder | photos, key, count?, first?}
// key는 저장 이름이라 언어와 상관없어야 한다 (화면에 보이는 folderName은 언어마다 다를 수 있다)
async function startWithEntries(folderName, imageEntries, source = { kind: 'handle', key: folderName }) {
  applySettingsFromForm();
  state.folderName = folderName;
  state.projectKey = source.key;
  // 예전 판은 사진 고르기를 화면 문구(folderName)로 저장했다
  const savedProject = await dbGet(`project:${source.key}`) ?? (source.key !== folderName ? await dbGet(`project:${folderName}`) : undefined);
  dbSet('lastProject', source);
  state.castKeys = null;
  if (savedProject && savedProject.settings?.preset === state.settings.preset) {
    // 같은 폴더, 같은 사진 종류로 다시 열면 결정과 학습 결과를 이어서 쓴다
    state.reviewers.a.decisions = savedProject.reviewers?.a?.decisions ?? {};
    state.reviewers.b.decisions = savedProject.reviewers?.b?.decisions ?? {};
    state.models = savedProject.models ?? state.models;
    // 점수 항목 수가 바뀐 예전 학습 결과는 쓸 수 없으니 버리고 다시 배운다
    for (const reviewerKey of Object.keys(state.models)) if (state.models[reviewerKey]?.theta?.length !== FEATURE_KEYS.length) state.models[reviewerKey] = null;
    state.rounds = savedProject.rounds ?? state.rounds;
    state.dividerSeen = savedProject.dividerSeen ?? state.dividerSeen;
    state.sentSampleCount = savedProject.sentSampleCount ?? state.sentSampleCount;
    state.expandedGroups = new Set(savedProject.expandedGroups ?? []);
    state.castKeys = savedProject.castKeys ?? null;
  }
  state.priorModel = await loadPriorModel();
  // 같은 화면에서 사진을 새로 고르면 지난 사진의 썸네일과 원본 주소를 풀어 메모리를 돌려준다
  for (const photo of state.photos) if (photo.thumbUrl) URL.revokeObjectURL(photo.thumbUrl);
  keepOnlyUrls([]);
  state.photos = imageEntries.map((entry) => ({
    id: entry.relPath, relPath: entry.relPath, name: entry.name,
    handle: entry.handle ?? null, file: entry.file ?? null,
    thumbUrl: '', takenAt: null, width: 0, height: 0,
    analysis: null,       // 분석 원값
    features: null,       // 점수 항목 10개 (0~1, FEATURE_KEYS 순서)
    score: 0,             // 0~100
    groupId: null, groupBest: false, recommended: false,
  }));
  state.photos.sort((left, right) => left.relPath.localeCompare(right.relPath, undefined, { numeric: true }));
  state.photoById = new Map(state.photos.map((photo) => [photo.id, photo]));
  if (!state.photos.length) { showToast(t('noPhotosFound')); return; }
  setView('analyze');
  await analyzeAll();
}

$('#pick-folder').onclick = async () => {
  if (!canPickFolder) { $('#fallback-input').click(); return; }
  try {
    const directoryHandle = await window.showDirectoryPicker({ id: 'moment-select-source', mode: 'read' });
    state.skippedFileCount = 0;
    $('#pick-note').textContent = t('pickNoteReading');
    const imageEntries = await collectImageEntries(directoryHandle);
    dbSet('lastHandle', directoryHandle);
    await startWithEntries(directoryHandle.name, imageEntries);
  } catch (error) {
    if (error.name !== 'AbortError') showToast(t('folderOpenError', escapeHtml(error.message)));
  }
};

// File System Access API가 없는 브라우저: 폴더 업로드 입력으로 읽기만
$('#fallback-input').onchange = async (event) => {
  const files = [...event.target.files];
  state.skippedFileCount = files.filter((file) => UNSUPPORTED_IMAGE.test(file.name)).length;
  const imageEntries = files
    .filter((file) => SUPPORTED_IMAGE.test(file.name) && !file.webkitRelativePath.split('/').some(isExportFolder))
    .map((file) => ({ relPath: file.webkitRelativePath.split('/').slice(1).join('/') || file.name, name: file.name, file }));
  const folderName = files[0]?.webkitRelativePath.split('/')[0] || t('fallbackFolderName');
  await startWithEntries(folderName, imageEntries, { kind: 'folder', key: folderName });
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
  const count = imageFiles.length, first = imageFiles[0].name;
  await startWithEntries(t('pickedSourceName', { count, first }), imageEntries, { kind: 'photos', key: `picked:${count}:${first}`, count, first });
};

// 환경에 맞게 시작 버튼을 배치한다
let pickNoteKey = 'pickNoteDefault'; // 시작 화면 안내 문구 (환경에 따라 바뀐다)
(function arrangeStartButtons() {
  const folderButton = $('#pick-folder'), photosButton = $('#pick-photos');
  if (isTouchDevice) {
    // 휴대폰: 사진 여러 장 선택이 기본. 폴더 선택은 지원하는 브라우저(안드로이드 크롬)에서만 보조로
    photosButton.classList.add('primary');
    folderButton.classList.remove('primary');
    folderButton.classList.toggle('hidden', !canPickFolder);
    photosButton.parentElement.prepend(photosButton);
    pickNoteKey = 'pickNoteTouch';
    $('#drive-note').classList.add('hidden');
  } else if (!canPickFolder) {
    pickNoteKey = 'pickNoteNoFolder';
  }
})();

let resumeText = null; // 이어하기 안내 문구를 만드는 함수 (언어를 바꿀 때 다시 쓰려고 기억)
// 지난번 설정을 시작 화면에 되살린다. 사진 종류가 같아야 결정을 이어 쓰므로, 같은 사진을 다시 고르면 그대로 이어진다
function restoreSettingsForm(savedProject) {
  Object.assign(state.settings, savedProject.settings);
  presetTouched = true; targetTouched = true;
  $('#target-input').value = state.settings.target;
  $('#category-input').checked = state.settings.useCategories;
  $('#crowd-row').classList.toggle('hidden', !state.settings.useCategories);
  $('#crowd-input').value = state.settings.crowdMin;
  $('#score-input').checked = state.settings.useScore;
  $('#eyes-input').checked = state.settings.penalizeClosedEyes !== false;
  $('#name-a').value = savedProject.reviewers?.a?.name ?? t('defaultNameA');
  $('#name-b').value = savedProject.reviewers?.b?.name ?? t('defaultNameB');
  renderPresetOptions();
  renderTopbar(); // 위쪽 '남긴 사진 0 / 목표'도 되살린 목표로
}
function showResume(makeText, buttonKey, onClick) {
  resumeText = makeText;
  $('#resume-name').textContent = makeText();
  $('#resume-button').dataset.i18n = buttonKey;
  $('#resume-button').textContent = t(buttonKey);
  $('#resume-button').onclick = onClick;
  $('#resume').classList.remove('hidden');
}
(async function setupResume() {
  const [lastHandle, lastProject] = await Promise.all([dbGet('lastHandle'), dbGet('lastProject')]);
  const handleUsable = Boolean(lastHandle?.queryPermission);
  const key = lastProject?.key ?? (handleUsable ? lastHandle.name : null);
  if (!key) return;
  const savedProject = await dbGet(`project:${key}`);
  if (savedProject?.settings) restoreSettingsForm(savedProject);
  if (handleUsable && (!lastProject || (lastProject.kind === 'handle' && lastProject.key === lastHandle.name))) {
    showResume(() => t('resumeName', lastHandle.name), 'resumeButton', async () => {
      if (await lastHandle.requestPermission({ mode: 'read' }) !== 'granted') { showToast(t('permissionNeeded')); return; }
      state.skippedFileCount = 0;
      await startWithEntries(lastHandle.name, await collectImageEntries(lastHandle));
    });
  } else if (savedProject && lastProject?.kind === 'photos') {
    // 사진 보관함에서 고른 사진은 다시 열 권한이 없어서, 같은 사진을 다시 고르도록 안내한다
    showResume(() => t('resumePicked', { count: lastProject.count, first: lastProject.first }), 'resumeRepick', () => $('#photos-input').click());
  } else if (savedProject && lastProject?.kind === 'folder') {
    showResume(() => t('resumeName', lastProject.key), 'resumeRepick', () => $('#fallback-input').click());
  }
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
// 얼굴 검출 엔진 상태 (점수 기준 패널의 분석 정보에 보여준다)
const faceEngine = {
  backend: 'none',     // 실제로 쓰는 계산 방식: webgl(그래픽 가속) | cpu | none
  float32: null,       // WebGL이 32비트 실수 렌더링을 지원하는지 (휴대폰 GPU 정밀도 문제를 짚어보는 단서)
  crossCheck: null,    // WebGL과 CPU 비교 결과 {samples, webglFaces, cpuFaces, switched}
  darkRetries: 0,      // 어두워서 밝게 보정해 다시 찾은 횟수
  darkRetryHits: 0,    // 그렇게 해서 얼굴을 찾은 횟수
};
// 진단용: 주소 뒤에 ?faceBackend=cpu 를 붙이면 처음부터 CPU로 계산한다
const forcedFaceBackend = new URLSearchParams(location.search).get('faceBackend');
const retryCanvas = document.createElement('canvas'); // 어두운 사진을 밝게 보정한 사본
const retryContext = retryCanvas.getContext('2d', { willReadFrequently: true });

// 미적 점수 모델 상태 (분석 정보에 보여준다)
let aestheticModel = null;
const aestheticEngine = { status: 'off', scored: 0 }; // status: ready | failed | off, scored: 점수를 낸 사진 수
const aestheticCanvas = document.createElement('canvas'); // 모델 입력 224x224 사본
aestheticCanvas.width = AESTHETIC_INPUT; aestheticCanvas.height = AESTHETIC_INPUT;
const aestheticContext = aestheticCanvas.getContext('2d');

// 눈 감음 정밀 모델(MediaPipe Face Landmarker) 상태 (분석 정보에 보여준다)
let eyeLandmarker = null;
const eyeEngine = { status: 'off', faces: 0, ms: 0 }; // status: ready | failed | off, faces: 이번에 정밀 모델에 넘긴 얼굴 수, ms: 그 시간 합
const eyeCropCanvas = document.createElement('canvas'); // 얼굴 둘레를 잘라낸 사본
const eyeCropContext = eyeCropCanvas.getContext('2d');

const megabytes = (bytes) => (bytes / 1_000_000).toFixed(1);
// 분석 전에 모델 파일을 받는다. "로딩 중"이 아니라 무엇을 위한 모델인지, 얼마나 받았는지 보여준다.
// 받은 파일은 서비스 워커가 기기에 보관해서 다음부터는 바로 넘어간다.
async function downloadModels(groups) {
  const totalBytes = groups.flatMap((group) => group.files).reduce((sum, file) => sum + file.bytes, 0);
  let finishedBytes = 0; // 다 받은 파일들의 크기 합
  $('#analyze-title').textContent = t('preparingTitle');
  $('#analyze-note').textContent = t('modelNote');
  $('#analyze-note').classList.remove('hidden');
  let lastPaintAt = 0; // 진행 막대를 마지막으로 그린 시각
  for (const group of groups) {
    const showProgress = (currentBytes, force = false) => {
      // 조각마다 그리면 막대의 CSS 애니메이션이 계속 처음부터 다시 시작돼 실제보다 뒤처진다. 0.1초에 한 번만 그린다
      if (!force && performance.now() - lastPaintAt < 100) return;
      lastPaintAt = performance.now();
      $('#analyze-meter').style.width = `${Math.min(100, (currentBytes / totalBytes) * 100)}%`;
      $('#analyze-text').textContent = t('modelProgress', { purpose: t(group.purposeKey), loaded: megabytes(currentBytes), total: megabytes(totalBytes) });
    };
    for (const file of group.files) {
      showProgress(finishedBytes, true);
      const response = await fetch(file.url);
      if (!response.ok) throw new Error(`${file.url} ${response.status}`);
      if (response.body?.getReader) {
        const reader = response.body.getReader();
        let fileBytes = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          fileBytes += value.length;
          showProgress(finishedBytes + Math.min(fileBytes, file.bytes));
        }
      } else {
        await response.arrayBuffer();
      }
      finishedBytes += file.bytes;
    }
  }
  $('#analyze-meter').style.width = '100%';
}

async function loadLibraries() {
  try { exifReader = (await import(EXIFR_URL)).default; } catch { exifReader = null; }
  aestheticModel = null;
  aestheticEngine.status = 'off';
  eyeLandmarker?.close();
  eyeLandmarker = null;
  eyeEngine.status = 'off';
  if (state.skipFaceAnalysis) return;
  try {
    await downloadModels(MODEL_DOWNLOADS.filter((group) => !group.onlyWhen || state.settings[group.onlyWhen]));
  } catch (error) {
    // 미리 받기가 실패해도 아래에서 각 모델을 직접 불러 본다
    console.warn('model download failed', error);
  }
  $('#analyze-title').textContent = t('analyzeTitle');
  $('#analyze-note').classList.add('hidden');
  $('#analyze-meter').style.width = '0%';
  try {
    $('#analyze-text').textContent = t('loadingModel');
    const module = await import(FACE_API_URL);
    // WebGL(그래픽 가속)을 먼저 쓰고, 안 되면 CPU로. wasm 백엔드는 쓰지 않는다
    // 진단용: ?faceF16=1 이면 휴대폰 GPU처럼 16비트 정밀도로 계산하게 해서 문제를 재현한다
    if (new URLSearchParams(location.search).get('faceF16') === '1') module.tf.env().set('WEBGL_FORCE_F16_TEXTURES', true);
    const webglReady = forcedFaceBackend === 'cpu' ? false : await module.tf.setBackend('webgl').catch(() => false);
    if (!webglReady) await module.tf.setBackend('cpu');
    await module.tf.ready();
    faceEngine.backend = module.tf.getBackend();
    try { faceEngine.float32 = faceEngine.backend === 'webgl' ? module.tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE') : null; } catch { faceEngine.float32 = null; }
    await Promise.all([
      module.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_BASE),
      module.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_BASE),
      module.nets.faceExpressionNet.loadFromUri(FACE_MODEL_BASE),
    ]);
    faceApi = module;
  } catch (error) {
    console.warn('face-api load failed', error);
    faceApi = null;
    showToast(t('modelFailed'));
    return;
  }
  // 같은 사람 알아보기 모델: 실패해도 나머지 분석은 그대로 한다
  try { await faceApi.nets.faceRecognitionNet.loadFromUri(FACE_MODEL_BASE); } catch (error) { console.warn('face recognition load failed', error); }
  // 미적 점수 모델 (face-api 안의 TensorFlow.js로 돌린다)
  try {
    const [manifest, weightBuffer] = await Promise.all([
      fetch(NIMA_MANIFEST_URL).then((response) => response.json()),
      fetch(NIMA_WEIGHTS_URL).then((response) => response.arrayBuffer()),
    ]);
    aestheticModel = buildAestheticModel(faceApi.tf, manifest, weightBuffer);
    aestheticEngine.status = 'ready';
  } catch (error) {
    console.warn('aesthetic model load failed', error);
    aestheticEngine.status = 'failed';
  }
  // 눈 감음 정밀 모델: 눈 감은 사진 감점을 켰을 때만. 실패하면 face-api 랜드마크로 눈 모양을 본다
  if (state.settings.penalizeClosedEyes) {
    try {
      quietMediaPipeInfoLogs();
      const { FaceLandmarker } = await import(MEDIAPIPE_URL);
      eyeLandmarker = await FaceLandmarker.createFromOptions({
        wasmLoaderPath: new URL(MEDIAPIPE_WASM_LOADER_URL, location.href).href,
        wasmBinaryPath: new URL(MEDIAPIPE_WASM_BINARY_URL, location.href).href,
      }, {
        baseOptions: { modelAssetPath: new URL(FACE_LANDMARKER_URL, location.href).href, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numFaces: 1,
        outputFaceBlendshapes: true,
      });
      eyeEngine.status = 'ready';
    } catch (error) {
      console.warn('face landmarker load failed', error);
      eyeLandmarker = null;
      eyeEngine.status = 'failed';
    }
  }
}
// MediaPipe(wasm)는 "INFO: Created TensorFlow Lite XNNPACK delegate" 같은 정보 로그도 console.error로 찍는다.
// 에러처럼 보이지 않게 "INFO:"로 시작하는 줄만 걸러 낸다 (경고와 진짜 에러는 그대로)
let mediaPipeLogsQuieted = false;
function quietMediaPipeInfoLogs() {
  if (mediaPipeLogsQuieted) return;
  mediaPipeLogsQuieted = true;
  const consoleError = console.error.bind(console);
  console.error = (...args) => {
    if (typeof args[0] === 'string' && args[0].startsWith('INFO:')) return;
    consoleError(...args);
  };
}
// 미적 점수를 지금 낼 수 있는지: 그래픽 가속(WebGL)이 없으면 장당 1~2초라 건너뛴다
const canScoreAesthetic = () => Boolean(aestheticModel && faceApi && faceApi.tf.getBackend() === 'webgl');

async function aestheticOfAnalysisCanvas() {
  if (!canScoreAesthetic()) return null;
  // 학습 때(Keras 기본)처럼 정사각형으로 늘려 최근접 방식으로 줄인다. 부드럽게 줄이면 점수가 1점 넘게 달라진다
  aestheticContext.imageSmoothingEnabled = false;
  aestheticContext.drawImage(analysisCanvas, 0, 0, AESTHETIC_INPUT, AESTHETIC_INPUT);
  try {
    const { mean, std } = await scoreAesthetic(aestheticModel, aestheticCanvas);
    aestheticEngine.scored++;
    return { mean: Math.round(mean * 1000) / 1000, std: Math.round(std * 1000) / 1000 };
  } catch (error) {
    console.warn('aesthetic scoring failed', error);
    return null;
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

// 1024px를 한 번에 9x8이나 16x12로 줄이면 픽셀 몇 개만 집어오는 앨리어싱이 생겨 비교값이 흔들린다.
// 반씩 여러 번 줄여서 64px 근처의 부드러운 사본을 만든 뒤 그것을 작은 크기로 줄인다.
const smallCanvas = document.createElement('canvas');
const smallContext = smallCanvas.getContext('2d');
function prepareSmallCopy() {
  let source = analysisCanvas;
  let width = analysisCanvas.width, height = analysisCanvas.height;
  const stepCanvas = document.createElement('canvas');
  const stepContext = stepCanvas.getContext('2d');
  while (Math.max(width, height) > 128) {
    width = Math.max(1, Math.round(width / 2)); height = Math.max(1, Math.round(height / 2));
    stepCanvas.width = width; stepCanvas.height = height;
    stepContext.imageSmoothingQuality = 'high';
    stepContext.drawImage(source, 0, 0, width, height);
    smallCanvas.width = width; smallCanvas.height = height;
    smallContext.drawImage(stepCanvas, 0, 0);
    source = smallCanvas;
  }
}

function differenceHash() {
  hashContext.imageSmoothingQuality = 'high';
  hashContext.drawImage(smallCanvas, 0, 0, 9, 8);
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
// 16x12로 줄인 흑백 이미지를 평균 0, 길이 1로 맞춘 벡터. 두 벡터의 내적이 상관도(-1~1)
const tinyCanvas = document.createElement('canvas');
tinyCanvas.width = 16; tinyCanvas.height = 12;
const tinyContext = tinyCanvas.getContext('2d', { willReadFrequently: true });
function tinyLumaVector() {
  tinyContext.imageSmoothingQuality = 'high';
  tinyContext.drawImage(smallCanvas, 0, 0, 16, 12);
  const gray = toGrayscale(tinyContext, 16, 12);
  const mean = gray.reduce((sum, value) => sum + value, 0) / gray.length;
  const centered = Array.from(gray, (value) => value - mean);
  const length = Math.hypot(...centered) || 1;
  return centered.map((value) => Math.round((value / length) * 10000) / 10000);
}
const correlation = (leftVector, rightVector) => leftVector.reduce((sum, value, index) => sum + value * rightVector[index], 0);

function popCount(word) { word -= (word >>> 1) & 0x55555555; word = (word & 0x33333333) + ((word >>> 2) & 0x33333333); return (((word + (word >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; }
const hashDistance = (leftHash, rightHash) => popCount((leftHash[0] ^ rightHash[0]) >>> 0) + popCount((leftHash[1] ^ rightHash[1]) >>> 0);

function eyeAspectRatio(eyePoints) {
  const distance = (pointA, pointB) => Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
  return (distance(eyePoints[1], eyePoints[5]) + distance(eyePoints[2], eyePoints[4])) / (2 * distance(eyePoints[0], eyePoints[3]));
}

function detectFaces(canvas, scoreThreshold = 0.45) {
  return faceApi
    .detectAllFaces(canvas, new faceApi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold }))
    .withFaceLandmarks()
    .withFaceExpressions();
}

// 어두운 사진을 감마 보정으로 밝혀 retryCanvas에 그린다 (평균 밝기를 120 근처로)
function brightenIntoRetryCanvas(meanLuminance) {
  const { width, height } = analysisCanvas;
  retryCanvas.width = width; retryCanvas.height = height;
  const gamma = Math.min(1, Math.max(0.3, Math.log(120 / 255) / Math.log(Math.max(1, meanLuminance) / 255)));
  const lookup = new Uint8ClampedArray(256); // 밝기 0~255를 보정 후 값으로 바꾸는 표
  for (let value = 0; value < 256; value++) lookup[value] = 255 * Math.pow(value / 255, gamma);
  const image = analysisContext.getImageData(0, 0, width, height);
  for (let index = 0; index < image.data.length; index += 4) {
    image.data[index] = lookup[image.data[index]];
    image.data[index + 1] = lookup[image.data[index + 1]];
    image.data[index + 2] = lookup[image.data[index + 2]];
  }
  retryContext.putImageData(image, 0, 0);
}

async function analyzeFaces(gray, smooth, width, height, noiseFloor, meanLuminance) {
  if (!faceApi) return null;
  let detectionCanvas = analysisCanvas; // 얼굴을 찾은 그림 (어두우면 밝힌 사본)
  let detections = await detectFaces(analysisCanvas);
  if (!detections.length && meanLuminance < DARK_RETRY_LUMINANCE) {
    // 밤 사진처럼 어두우면 얼굴을 놓치기 쉬워서, 밝게 보정한 사본으로 한 번 더 찾는다
    faceEngine.darkRetries++;
    brightenIntoRetryCanvas(meanLuminance);
    detections = await detectFaces(retryCanvas, DARK_RETRY_THRESHOLD);
    if (detections.length) { faceEngine.darkRetryHits++; detectionCanvas = retryCanvas; }
  }
  const descriptors = await faceDescriptors(detectionCanvas, detections, height);
  const blendshapes = faceBlendshapes(detectionCanvas, detections);
  return detections.map((detection, detectionIndex) => {
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
    // 정밀 모델이 얼굴을 찾았으면 눈 감음과 웃음을 그것으로 본다. 못 찾았으면 눈 모양(EAR)으로
    const mesh = blendshapes.get(detectionIndex);
    const eyesJudged = Boolean(mesh) || canJudgeEyes;
    return {
      box: { x: box.x / width, y: box.y / height, w: box.width / width, h: box.height / height },
      isMainFace: box.height / height >= MAIN_FACE_MIN_HEIGHT,
      sharpness: Math.max(0, laplacianVariance(smooth, width, height, box.x, box.y, box.x + box.width, box.y + box.height) - noiseFloor), // 노이즈가 만든 가짜 선명도는 뺀다
      luminance: faceLight.mean,
      smile: mesh ? ((expressions.happy ?? 0) + mesh.smile) / 2 : expressions.happy ?? 0,
      negative: (expressions.sad ?? 0) + (expressions.angry ?? 0) + (expressions.fearful ?? 0) + (expressions.disgusted ?? 0),
      // 두 눈 모두 감겨 보여야 눈 감음 (한쪽만이면 윙크나 각도 문제일 가능성이 크다)
      eyesClosed: mesh
        ? mesh.blinkLeft >= EYE_BLINK_CLOSED && mesh.blinkRight >= EYE_BLINK_CLOSED
        : canJudgeEyes && leftEar < EYE_CLOSED_EAR && rightEar < EYE_CLOSED_EAR,
      eyesJudged,
      eyeMethod: mesh ? 'mesh' : eyesJudged ? 'ear' : null, // 눈을 무엇으로 판정했는지 (분석 정보용)
      cutAtEdge: box.x < 2 || box.y < 2 || box.x + box.width > width - 2 || box.y + box.height > height - 2,
      descriptor: descriptors.get(detectionIndex) ?? null, // 누구인지 나타내는 128개 숫자 (큰 얼굴 몇 개만)
    };
  });
}

// 큰 얼굴 몇 개마다 둘레를 잘라 정밀 모델에 넘기고 눈 감음, 웃음 값을 받는다.
// 반환: 검출 순번에서 {blinkLeft, blinkRight, smile}로 (정밀 모델이 없거나 얼굴을 못 찾으면 빠진다)
function faceBlendshapes(sourceCanvas, detections) {
  const result = new Map();
  if (!eyeLandmarker) return result;
  const chosen = detections
    .map((detection, index) => ({ box: detection.detection.box, index }))
    .filter(({ box }) => box.height >= EYE_MESH_MIN_FACE_PX)
    .sort((left, right) => right.box.height - left.box.height)
    .slice(0, EYE_MESH_MAX_FACES);
  for (const { box, index } of chosen) {
    const left = Math.max(0, Math.floor(box.x + box.width / 2 - (box.width * EYE_MESH_CROP) / 2));
    const top = Math.max(0, Math.floor(box.y + box.height / 2 - (box.height * EYE_MESH_CROP) / 2));
    const right = Math.min(sourceCanvas.width, Math.ceil(box.x + box.width / 2 + (box.width * EYE_MESH_CROP) / 2));
    const bottom = Math.min(sourceCanvas.height, Math.ceil(box.y + box.height / 2 + (box.height * EYE_MESH_CROP) / 2));
    eyeCropCanvas.width = right - left; eyeCropCanvas.height = bottom - top;
    eyeCropContext.drawImage(sourceCanvas, left, top, right - left, bottom - top, 0, 0, right - left, bottom - top);
    try {
      const startedAt = performance.now();
      const detected = eyeLandmarker.detect(eyeCropCanvas);
      eyeEngine.ms += performance.now() - startedAt;
      eyeEngine.faces++;
      const categories = detected.faceBlendshapes?.[0]?.categories;
      const nose = detected.faceLandmarks?.[0]?.[1]; // 코끝
      if (!categories || !nose) continue;
      // 잘라낸 영역에 다른 사람 얼굴이 함께 들어가 그쪽을 찾았으면 쓰지 않는다
      const noseX = left + nose.x * eyeCropCanvas.width, noseY = top + nose.y * eyeCropCanvas.height;
      if (noseX < box.x || noseX > box.x + box.width || noseY < box.y || noseY > box.y + box.height) continue;
      const score = (name) => categories.find((category) => category.categoryName === name)?.score ?? 0;
      result.set(index, {
        blinkLeft: score('eyeBlinkLeft'),
        blinkRight: score('eyeBlinkRight'),
        smile: (score('mouthSmileLeft') + score('mouthSmileRight')) / 2,
      });
    } catch (error) {
      console.warn('face landmarker failed', error);
    }
  }
  return result;
}

// 큰 얼굴 몇 개만 골라 '누구인지' 특징(128차원)을 계산한다. 반환: 검출 순번에서 특징 배열로
async function faceDescriptors(sourceCanvas, detections, imageHeight) {
  const result = new Map();
  if (!faceApi?.nets.faceRecognitionNet.isLoaded || !detections.length) return result;
  const chosen = detections
    .map((detection, index) => ({ detection, index }))
    .filter(({ detection }) => detection.detection.box.height / imageHeight >= MAIN_FACE_MIN_HEIGHT)
    .sort((left, right) => right.detection.detection.box.height - left.detection.detection.box.height)
    .slice(0, DESCRIPTOR_MAX_FACES);
  if (!chosen.length) return result;
  try {
    // 랜드마크로 기울기를 바로잡은 얼굴 영역을 잘라서 계산한다 (face-api가 내부에서 하는 방식과 같다)
    const faceCanvases = await faceApi.extractFaces(sourceCanvas, chosen.map(({ detection }) => detection.alignedRect));
    for (let order = 0; order < chosen.length; order++) {
      const descriptor = await faceApi.computeFaceDescriptor(faceCanvases[order]);
      result.set(chosen[order].index, Array.from(descriptor, (value) => Math.round(value * 1000) / 1000));
    }
  } catch (error) {
    console.warn('face descriptor failed', error);
  }
  return result;
}

async function analyzeOne(photo) {
  const file = await getPhotoFile(photo);
  // 얼굴 분석, 미적 점수, 눈 감음 정밀 모델을 썼는지에 따라 따로 보관한다 (나중에 더 할 수 있게 되면 다시 분석)
  const cacheKey = `cache:${CACHE_VERSION}:${photo.relPath}|${file.size}|${file.lastModified}|${faceApi ? 'face' : 'basic'}${canScoreAesthetic() ? '+aesthetic' : ''}${eyeLandmarker ? '+eyes' : ''}`;
  const cached = await dbGet(cacheKey);
  if (cached) {
    Object.assign(photo, { takenAt: cached.takenAt, width: cached.width, height: cached.height, analysis: cached.analysis });
    photo.thumbUrl = URL.createObjectURL(cached.thumbBlob);
    return;
  }
  let takenAt = null;
  if (exifReader) {
    try {
      // exifr lite는 태그 이름 배열만 넘기면 오류가 나서(그동안 촬영 시각을 못 읽었다) 블록별로 지정한다.
      // 촬영 시각이 없으면 IFD0의 수정 시각(ModifyDate)을 대신 쓴다
      const exif = await exifReader.parse(file, { ifd0: ['ModifyDate'], exif: ['DateTimeOriginal', 'CreateDate'], gps: false, interop: false, ifd1: false });
      const date = exif?.DateTimeOriginal ?? exif?.CreateDate ?? exif?.ModifyDate;
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
  const light = luminanceStats(gray, width);
  photo.analysis = {
    focusSharpness: focusSharpness(smooth, width, height, noiseFloor),
    noise,
    light,
    hash: (prepareSmallCopy(), differenceHash()),
    tiny: tinyLumaVector(),
    faces: await analyzeFaces(gray, smooth, width, height, noiseFloor, light.mean), // null이면 얼굴 분석을 하지 않음
    aesthetic: await aestheticOfAnalysisCanvas(), // {mean, std} 1~10점, null이면 계산하지 않음
  };
  photo.takenAt = takenAt;
  photo.thumbUrl = URL.createObjectURL(thumbBlob);
  dbSet(cacheKey, { takenAt, width: photo.width, height: photo.height, analysis: photo.analysis, thumbBlob });
}

// 휴대폰 GPU에 따라 WebGL 계산이 부정확해 얼굴을 못 찾는 경우가 있다.
// 처음 몇 장을 WebGL과 CPU로 각각 돌려 보고, WebGL이 눈에 띄게 적게 찾으면 CPU로 계속한다.
async function crossCheckFaceBackends() {
  if (!faceApi || faceEngine.backend !== 'webgl' || forcedFaceBackend === 'webgl') return;
  $('#analyze-text').textContent = t('checkingBackend');
  const sampleCanvases = []; // 비교에 쓸 분석용 크기 사본
  for (const photo of state.photos.slice(0, CROSS_CHECK_SAMPLES)) {
    try {
      const bitmap = await createImageBitmap(await getPhotoFile(photo), { imageOrientation: 'from-image' });
      const canvas = document.createElement('canvas');
      drawScaled(bitmap, ANALYSIS_LONG_SIDE, canvas, canvas.getContext('2d'));
      bitmap.close();
      sampleCanvases.push(canvas);
    } catch { /* 열지 못한 사진은 비교에서 뺀다 */ }
  }
  if (!sampleCanvases.length) return;
  const countFaces = async () => {
    let total = 0;
    for (const canvas of sampleCanvases) total += (await faceApi.detectAllFaces(canvas, new faceApi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.45 }))).length;
    return total;
  };
  try {
    const webglFaces = await countFaces();
    await faceApi.tf.setBackend('cpu');
    await faceApi.tf.ready();
    const cpuFaces = await countFaces();
    const switched = cpuFaces >= 2 && webglFaces < cpuFaces * 0.6;
    if (!switched) { await faceApi.tf.setBackend('webgl'); await faceApi.tf.ready(); }
    faceEngine.crossCheck = { samples: sampleCanvases.length, webglFaces, cpuFaces, switched };
  } catch (error) {
    console.warn('face backend cross-check failed', error);
  }
  faceEngine.backend = faceApi.tf.getBackend();
  console.info('face engine', JSON.stringify(faceEngine));
}

async function analyzeAll() {
  state.analysisDone = false;
  $('#analyze-meter').style.width = '0%';
  $('#analyze-sheet').innerHTML = '';
  await loadLibraries();
  await crossCheckFaceBackends();
  const totalCount = state.photos.length;
  let finishedCount = 0;
  let rateStartedAt = 0; // 남은 시간 계산 기준 (첫 장은 모델 준비 시간이 섞여 있어서 둘째 장부터 잰다)
  for (const photo of state.photos) {
    try {
      await analyzeOne(photo);
    } catch (error) {
      console.warn('analyze failed', photo.relPath, error);
      photo.analysis = { focusSharpness: 0, noise: 0, light: { mean: 128, brightClip: 0, darkClip: 0 }, hash: [0, 0], faces: null, failed: true };
    }
    finishedCount++;
    if (finishedCount === 1) rateStartedAt = performance.now();
    const remainingSeconds = finishedCount <= ETA_WARMUP_PHOTOS ? null
      : Math.round(((performance.now() - rateStartedAt) / 1000 / (finishedCount - 1)) * (totalCount - finishedCount));
    $('#analyze-meter').style.width = `${(finishedCount / totalCount) * 100}%`;
    $('#analyze-text').textContent = t('analyzeProgress', { done: finishedCount, total: totalCount, seconds: remainingSeconds, faceOff: !faceApi });
    if (photo.thumbUrl) {
      const sheet = $('#analyze-sheet');
      sheet.insertAdjacentHTML('beforeend', `<img src="${photo.thumbUrl}" alt="">`);
      if (sheet.childElementCount > ANALYZE_SHEET_MAX) sheet.firstElementChild.remove();
    }
  }
  // 브라우저가 열지 못한 사진(HEIC 등)은 고를 수도 없으니 목록에서 뺀다
  const unreadable = state.photos.filter((photo) => photo.analysis.failed);
  if (unreadable.length) {
    state.photos = state.photos.filter((photo) => !photo.analysis.failed);
    state.photoById = new Map(state.photos.map((photo) => [photo.id, photo]));
    state.skippedFileCount += unreadable.length;
  }
  if (!state.photos.length) { showToast(t('noReadable')); setView('setup'); return; }
  buildPeople();
  buildScenes();
  computeFeatures();
  rescoreAll();
  buildSimilarGroups();
  buildUnits();
  state.analysisDone = true;
  saveProjectSoon();
  const skipNote = state.skippedFileCount ? t('skipNote', state.skippedFileCount) : '';
  showToast(t('analyzedToast', { count: state.photos.length, groups: state.groups.size, skipNote }));
  setView('first');
}

$('#skip-faces').onclick = () => {
  state.skipFaceAnalysis = true;
  faceApi = null;
  $('#skip-faces').disabled = true;
  showToast(t('skipFacesToast'));
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
  // 미적 점수도 촬영마다 수준이 달라서 이 폴더 안의 순위로 쓴다. 점수가 없는 사진은 중립값 0.5
  const aestheticRank = percentileRanker(usable.filter((photo) => photo.analysis.aesthetic).map((photo) => photo.analysis.aesthetic.mean));
  const castKeys = new Set(effectiveCastKeys());
  for (const photo of state.photos) {
    if (photo.analysis.failed) { photo.features = null; continue; }
    const faces = mainFacesOf(photo);
    const focusSharp = focusRank(Math.log1p(photo.analysis.focusSharpness));
    const aesthetic = photo.analysis.aesthetic ? aestheticRank(photo.analysis.aesthetic.mean) : 0.5;
    // 주인공: 고른 사람 중 몇 명이 이 사진에 나왔는지 (주인공을 안 골랐으면 모두 0.5라 순서에 영향 없음)
    const cast = castKeys.size ? castPresentIn(photo, castKeys).size / castKeys.size : 0.5;
    // 눈 뜸: 눈을 판정할 수 있었던 얼굴 중 눈 뜬 비율. 감점을 끄면 모든 사진이 1이라 순서에 영향이 없다
    const judgedFaces = faces.filter((face) => face.eyesJudged);
    const eyesOpen = state.settings.penalizeClosedEyes && judgedFaces.length ? 1 - judgedFaces.filter((face) => face.eyesClosed).length / judgedFaces.length : 1;
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
        eyesOpen,
        aesthetic,
        cast,
      };
    } else {
      // 인물이 없으면 인물 항목은 초점 부분으로 대신하고, 표정과 구도는 중립값 0.5
      photo.features = { subjectSharp: focusSharp, focusSharp, exposure: globalExposure, expression: 0.5, smile: 0.5, framing: 0.5, clean: 1 - noiseRank(photo.analysis.noise), eyesOpen: 1, aesthetic, cast };
    }
  }
}

/* =====================================================================
   같은 사람 알아보기 (주인공)
   - 얼굴 특징(128차원)이 가까운 얼굴끼리 한 사람으로 묶는다 (중심점과의 거리, 탐욕 방식)
   - 한 사진 안의 두 얼굴은 같은 사람으로 묶지 않는다
   - 사람 key는 그 사람의 첫 얼굴(사진 id#얼굴 순번)이라 같은 폴더를 다시 열어도 같다
   ===================================================================== */
const personByFace = new Map(); // `${사진 id}#${얼굴 순번}`에서 사람 key로
let autoCastKeys = [];          // 자동으로 고른 주인공 (사용자가 아직 안 골랐을 때 쓴다)
const effectiveCastKeys = () => state.castKeys ?? autoCastKeys;
const faceRef = (photoId, faceIndex) => `${photoId}#${faceIndex}`;
const euclidean = (left, right) => { let sum = 0; for (let index = 0; index < left.length; index++) sum += (left[index] - right[index]) ** 2; return Math.sqrt(sum); };

function buildPeople() {
  personByFace.clear();
  const people = []; // {key, centroid, faces:[{photoId, faceIndex}], photoIds:Set}
  const orderedPhotos = [...state.photos].sort((left, right) => left.id.localeCompare(right.id, undefined, { numeric: true }));
  for (const photo of orderedPhotos) {
    const faces = photo.analysis?.faces ?? [];
    const usedInPhoto = new Set(); // 이 사진에서 이미 배정된 사람
    faces.forEach((face, faceIndex) => {
      if (!face.descriptor) return;
      let nearest = null, nearestDistance = Infinity;
      for (const person of people) {
        if (usedInPhoto.has(person)) continue;
        const distance = euclidean(face.descriptor, person.centroid);
        if (distance < nearestDistance) { nearest = person; nearestDistance = distance; }
      }
      if (!nearest || nearestDistance >= SAME_PERSON_DISTANCE) {
        nearest = { key: faceRef(photo.id, faceIndex), centroid: Float64Array.from(face.descriptor), faces: [], photoIds: new Set() };
        people.push(nearest);
      } else {
        // 중심점을 새 얼굴까지 포함한 평균으로 옮긴다
        const count = nearest.faces.length;
        for (let index = 0; index < nearest.centroid.length; index++) nearest.centroid[index] = (nearest.centroid[index] * count + face.descriptor[index]) / (count + 1);
      }
      nearest.faces.push({ photoId: photo.id, faceIndex });
      nearest.photoIds.add(photo.id);
      usedInPhoto.add(nearest);
      personByFace.set(faceRef(photo.id, faceIndex), nearest.key);
    });
  }
  state.people = people.sort((left, right) => right.photoIds.size - left.photoIds.size || left.key.localeCompare(right.key));
  // 사용자가 고른 주인공 중 이번에 없는 사람은 뺀다 (모두 없어지면 자동 선택으로 돌아간다)
  if (state.castKeys) {
    const existing = new Set(state.people.map((person) => person.key));
    state.castKeys = state.castKeys.filter((key) => existing.has(key));
    if (!state.castKeys.length) state.castKeys = null;
  }
  // 자동 선택: 얼굴 있는 사진 중 충분히 자주 나온 사람을 사진 종류별 인원만큼
  const photosWithDescriptors = state.photos.filter((photo) => (photo.analysis?.faces ?? []).some((face) => face.descriptor)).length;
  const minimumPhotos = Math.max(CAST_MIN_PHOTOS, Math.ceil(photosWithDescriptors * CAST_MIN_SHARE));
  autoCastKeys = state.people.filter((person) => person.photoIds.size >= minimumPhotos).slice(0, preset().castCount).map((person) => person.key);
}

// 이 사진에 나온 주인공 key 집합
function castPresentIn(photo, castKeys) {
  const present = new Set();
  (photo.analysis?.faces ?? []).forEach((face, faceIndex) => {
    const personKey = personByFace.get(faceRef(photo.id, faceIndex));
    if (personKey && castKeys.has(personKey)) present.add(personKey);
  });
  return present;
}

/* =====================================================================
   장면: 촬영 시각이 SCENE_GAP_MS 넘게 벌어지면 다른 장면으로 본다
   - 줄이기 단계에서 어떤 장면의 마지막 남은 사진은 뒤로 미뤄서 장면이 통째로 사라지지 않게 한다
   - 촬영 시각이 없는 사진이 많으면(메신저로 받은 사진 등) 장면을 나누지 않는다
   ===================================================================== */
function buildScenes() {
  state.scenes = new Map();
  const timed = state.photos.filter((photo) => photo.takenAt).sort((left, right) => left.takenAt - right.takenAt);
  if (timed.length < state.photos.length * 0.8) return;
  let sceneNumber = 0;
  timed.forEach((photo, index) => {
    if (index && photo.takenAt - timed[index - 1].takenAt > SCENE_GAP_MS) sceneNumber++;
    state.scenes.set(photo.id, sceneNumber);
  });
}
const sceneCount = () => new Set(state.scenes.values()).size;
// 장면마다 지금 남긴 사진 수
function keptCountByScene() {
  const counts = new Map();
  for (const id of keepIds()) {
    const scene = state.scenes.get(id);
    if (scene !== undefined) counts.set(scene, (counts.get(scene) ?? 0) + 1);
  }
  return counts;
}
// 이 사진이 자기 장면에서 남은 마지막 한 장인지 (장면이 하나뿐이면 의미가 없어서 false)
function isLastKeptOfScene(photo, counts = keptCountByScene()) {
  if (sceneCount() < 2) return false;
  const scene = state.scenes.get(photo.id);
  return scene !== undefined && statusOf(photo.id) === 'keep' && counts.get(scene) === 1;
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

// 비슷한 컷끼리 고른 결과: "이긴 사진이 진 사진보다 낫다"는 쌍. 장면과 인물이 같아서 품질과 취향 차이만 남는다
// 진 사진의 마지막 결정 기록(lostTo)에서 꺼낸다. 되돌리면 기록이 사라지므로 쌍도 함께 사라진다
function trainingPairs(reviewerKey) {
  const pairs = [];
  for (const [photoId, decision] of Object.entries(state.reviewers[reviewerKey].decisions)) {
    const lastEntry = decision.log?.[decision.log.length - 1];
    if (decision.status !== 'drop' || !lastEntry?.lostTo) continue;
    const loser = state.photoById.get(photoId);
    if (!loser?.features) continue;
    const loserVector = featureVector(loser);
    for (const { id: winnerId } of lastEntry.lostTo) {
      const winner = state.photoById.get(winnerId);
      if (!winner?.features) continue;
      pairs.push({ difference: featureVector(winner).map((value, index) => value - loserVector[index]) });
    }
  }
  return pairs;
}
const learningEvidence = (reviewerKey) => trainingSamples(reviewerKey).length + trainingPairs(reviewerKey).length;

// 낱장 결정(남김/뺌)은 로지스틱 회귀, 비교 쌍은 Bradley-Terry(이긴 쪽 점수가 높을 확률)로 함께 맞춘다
function trainModel(reviewerKey) {
  const samples = trainingSamples(reviewerKey);
  const pairs = trainingPairs(reviewerKey);
  const keepCount = samples.filter((sample) => sample.label === 1).length;
  const samplesReady = samples.length >= LEARN_MIN_DECISIONS && keepCount >= LEARN_MIN_PER_CLASS && samples.length - keepCount >= LEARN_MIN_PER_CLASS;
  const pairsReady = pairs.length >= LEARN_MIN_PAIRS;
  if (!samplesReady && !pairsReady) return false;
  const usedSamples = samplesReady ? samples : []; // 남김/뺌이 한쪽으로 쏠려 있으면 낱장 결정은 쓰지 않는다
  const evidenceCount = usedSamples.length + pairs.length;
  const prior = state.priorModel;
  const theta = [...prior.theta];
  let bias = prior.bias;
  const regularization = 0.01, learningRate = 1.5;
  const dot = (vector) => theta.reduce((sum, weight, index) => sum + weight * vector[index], 0);
  for (let iteration = 0; iteration < 1500; iteration++) {
    const thetaGradient = new Array(theta.length).fill(0);
    let biasGradient = 0;
    for (const { vector, label } of usedSamples) {
      const error = sigmoid(bias + dot(vector)) - label;
      for (let index = 0; index < theta.length; index++) thetaGradient[index] += error * vector[index];
      biasGradient += error;
    }
    for (const { difference } of pairs) {
      const error = sigmoid(dot(difference)) - 1; // 이긴 쪽이 이길 확률을 1에 가깝게 (기준점 bias는 쌍에서 상쇄된다)
      for (let index = 0; index < theta.length; index++) thetaGradient[index] += error * difference[index];
    }
    for (let index = 0; index < theta.length; index++) {
      theta[index] -= learningRate * (thetaGradient[index] / evidenceCount + 2 * regularization * (theta[index] - prior.theta[index]));
    }
    if (usedSamples.length) bias -= learningRate * (biasGradient / usedSamples.length);
  }
  state.models[reviewerKey] = { theta, bias, source: 'learned', trainedOn: evidenceCount, decisions: usedSamples.length, pairs: pairs.length };
  return true;
}

function maybeRetrain() {
  if (!state.settings.useScore) return;
  const reviewerKey = state.currentReviewer;
  const evidence = learningEvidence(reviewerKey);
  const lastTrained = state.models[reviewerKey]?.trainedOn ?? 0;
  if (lastTrained !== 0 && evidence - lastTrained < LEARN_EVERY) return;
  if (!trainModel(reviewerKey)) return;
  rescoreAll();
  buildUnits();
  if (lastTrained === 0) showToast(t('learnedToast', evidence), [], 5500);
  maybeSendStats();
}

// 점수가 실제 결정을 얼마나 맞혔는지 (결정할 때 보이던 점수로 계산하므로 학습에 쓴 데이터로 자기 채점하지 않는다)
//  auc: 남긴 사진이 뺀 사진보다 점수가 높았던 비율 (0.5면 무작위, 1이면 완벽)
//  pairHits/pairTotal: 비슷한 컷 비교에서 고른 사진의 점수가 더 높았던 횟수
function scoreAccuracy(reviewerKey) {
  const keptScores = [], droppedScores = [];
  let pairHits = 0, pairTotal = 0;
  for (const decision of Object.values(state.reviewers[reviewerKey].decisions)) {
    const lastEntry = decision.log?.[decision.log.length - 1];
    if (!lastEntry || typeof lastEntry.score !== 'number' || lastEntry.stage === 'first-bulk') continue;
    (decision.status === 'keep' ? keptScores : droppedScores).push(lastEntry.score);
    for (const winner of lastEntry.lostTo ?? []) {
      if (typeof winner.score !== 'number') continue;
      pairTotal++;
      pairHits += winner.score > lastEntry.score ? 1 : winner.score === lastEntry.score ? 0.5 : 0;
    }
  }
  let concordant = 0;
  for (const keptScore of keptScores) for (const droppedScore of droppedScores) concordant += keptScore > droppedScore ? 1 : keptScore === droppedScore ? 0.5 : 0;
  const auc = keptScores.length && droppedScores.length ? concordant / (keptScores.length * droppedScores.length) : null;
  return { auc, decisions: keptScores.length + droppedScores.length, pairHits, pairTotal };
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
const hasClosedEyes = (photo) => mainFacesOf(photo).some((face) => face.eyesClosed);

// 두 사진이 같은 순간의 비슷한 컷인지: 해시가 아주 가깝거나, 조금 멀어도 전체 명암 배치가 닮았으면
function looksSimilar(leftPhoto, rightPhoto) {
  if (leftPhoto.analysis.failed || rightPhoto.analysis.failed) return false;
  const distance = hashDistance(leftPhoto.analysis.hash, rightPhoto.analysis.hash);
  if (distance <= SIMILAR_HASH_DISTANCE) return true;
  if (distance > SIMILAR_HASH_LOOSE || !leftPhoto.analysis.tiny || !rightPhoto.analysis.tiny) return false;
  return correlation(leftPhoto.analysis.tiny, rightPhoto.analysis.tiny) >= SIMILAR_CORRELATION;
}

// 비슷한 컷 묶기: 촬영 순서로 앞쪽 여러 장과 비교해서 이어지는 것끼리 한 묶음으로 합친다 (union-find)
// 바로 옆 사진만 비교하던 방식은 사이에 다른 사진이 끼거나 몇십 초 뒤에 다시 찍은 컷을 놓쳤다
function buildSimilarGroups() {
  state.groups = new Map();
  const ordered = [...state.photos].sort((left, right) => {
    if (left.takenAt && right.takenAt && left.takenAt !== right.takenAt) return left.takenAt - right.takenAt;
    return left.relPath.localeCompare(right.relPath, undefined, { numeric: true });
  });
  const parentIndex = ordered.map((_, index) => index); // union-find 부모 표
  const findRoot = (index) => { while (parentIndex[index] !== index) { parentIndex[index] = parentIndex[parentIndex[index]]; index = parentIndex[index]; } return index; };
  for (let current = 0; current < ordered.length; current++) {
    for (let previous = current - 1; previous >= Math.max(0, current - SIMILAR_WINDOW); previous--) {
      const currentPhoto = ordered[current], previousPhoto = ordered[previous];
      if (currentPhoto.takenAt && previousPhoto.takenAt && currentPhoto.takenAt - previousPhoto.takenAt > SIMILAR_TIME_GAP_MS) break;
      if (looksSimilar(currentPhoto, previousPhoto)) parentIndex[findRoot(current)] = findRoot(previous);
    }
  }
  const clusters = new Map(); // 대표 번호에서 사진 목록으로 (촬영 순서 유지)
  ordered.forEach((photo, index) => {
    photo.groupId = null;
    const root = findRoot(index);
    if (!clusters.has(root)) clusters.set(root, []);
    clusters.get(root).push(photo);
  });
  for (const members of clusters.values()) {
    if (members.length < 2) continue;
    // 너무 크면 화면에서 비교하기 어려워서 촬영 순서대로 나눈다
    for (let offset = 0; offset < members.length; offset += GROUP_MAX_SIZE) {
      const chunk = members.slice(offset, offset + GROUP_MAX_SIZE);
      if (chunk.length < 2) continue;
      const groupId = `g:${chunk[0].id}`; // 첫 사진 기준 id라 다시 열어도 같은 묶음은 같은 id
      state.groups.set(groupId, chunk.map((photo) => photo.id));
      for (const photo of chunk) photo.groupId = groupId;
    }
  }
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

// changes: [{id, status}], preferences: 비슷한 컷끼리 고른 결과 [{winnerId, loserId}] (쌍 비교 학습용)
// 되돌리기 버튼: 넓은 화면에서는 'Undo' 글자를 붙이고, 방금 결정했으면 잠깐 강조해서 되돌릴 수 있다는 걸 알린다
let lastDecisionAt = 0;
let undoNudges = 0;
const UNDO_NUDGE_MS = 600;   // 결정 뒤 이 시간 안에 그린 조작 줄이면 되돌리기 버튼을 강조한다
const UNDO_NUDGE_LIMIT = 3;  // 처음 이만큼만 (되돌릴 수 있다는 걸 알리는 용도, 그 뒤로는 방해만 된다)
const shouldNudgeUndo = () => performance.now() - lastDecisionAt < UNDO_NUDGE_MS && undoNudges < UNDO_NUDGE_LIMIT && ++undoNudges > 0;
const undoButtonHtml = () => `<button class="button icon-only undo${shouldNudgeUndo() ? ' nudge' : ''}" id="act-undo" title="${t('undoTitle')}" aria-label="${t('undo')}"><i class="ph ph-arrow-counter-clockwise"></i><span class="undo-label">${t('undo')}</span></button>`;

function recordDecisions(changes, stage, preferences = []) {
  lastDecisionAt = performance.now();
  const now = Date.now();
  const undoEntry = emptyUndo();
  const winnersByLoser = new Map(); // 진 사진 id에서 이긴 사진 [{id, score}]로
  for (const { winnerId, loserId } of preferences) {
    if (!winnersByLoser.has(loserId)) winnersByLoser.set(loserId, []);
    winnersByLoser.get(loserId).push({ id: winnerId, score: state.photoById.get(winnerId)?.score });
  }
  for (const { id, status } of changes) {
    const previousDecision = decisionOf(id);
    undoEntry.previous.push({ id, decision: previousDecision ? structuredClone(previousDecision) : null });
    // score: 결정할 때 보이던 점수 (점수 적중률 측정용), lostTo: 이 사진 대신 고른 비슷한 사진
    const logEntry = { status, at: now, stage, score: state.photoById.get(id)?.score };
    if (status === 'drop' && winnersByLoser.has(id)) logEntry.lostTo = winnersByLoser.get(id);
    reviewer().decisions[id] = {
      status, at: now,
      // 뺐던 사진을 다시 남기면 그 시각을 기록한다
      addedAt: status === 'keep' ? (previousDecision?.status === 'drop' ? now : previousDecision?.addedAt) : undefined,
      log: [...(previousDecision?.log ?? []), logEntry],
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
  if (!undoEntry) { showToast(t('nothingToUndo'), [], 1500); return; }
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

// 1차 셀렉에서 사용자가 결정한 횟수 (한 번에 여러 장을 고른 묶음도 1회)
function firstPassActionCount() {
  const actionTimes = new Set();
  for (const decision of Object.values(reviewer().decisions)) {
    for (const entry of decision.log ?? []) if (entry.stage?.startsWith('first') && entry.stage !== 'first-bulk') actionTimes.add(entry.at);
  }
  return actionTimes.size;
}
// 취향 확인용 사진: 학습 초반에 PROBE_EVERY번째마다, 점수가 가운데쯤이라 모델이 가장 헷갈리는 사진(낱장 또는 묶음)을 끼워 보여준다.
// 점수 높은 사진만 보면 "무엇을 빼는지"를 배울 수 없어서 학습이 느리다
let probeUnitKey = null;
function pickProbeUnit(pendingUnits) {
  probeUnitKey = null;
  if (!state.settings.useScore) return null;
  const actionCount = firstPassActionCount();
  if (actionCount >= PROBE_UNTIL_ACTIONS || (actionCount + 1) % PROBE_EVERY !== 0) return null;
  if (pendingUnits.length < 2) return null;
  const sortedScores = state.units.map((unit) => unit.score).sort((left, right) => left - right);
  const medianScore = sortedScores[Math.floor(sortedScores.length / 2)];
  const probe = pendingUnits.reduce((best, unit) => {
    const gap = Math.abs(unit.score - medianScore), bestGap = Math.abs(best.score - medianScore);
    return gap < bestGap || (gap === bestGap && unit.key < best.key) ? unit : best;
  });
  probeUnitKey = probe.key;
  return probe;
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
  const probe = recommended.length ? pickProbeUnit(pending) : null; // 추천 사진을 다 본 뒤에는 섞지 않는다
  if (probe && queue[0] !== probe) queue.splice(queue.indexOf(probe), 1), queue.unshift(probe);
  return bringPriorityFirst(queue, (unit) => unit.ids);
}
function distillQueue() {
  const seenIds = state.rounds[state.currentReviewer].distill;
  const counts = keptCountByScene();
  const sceneProtected = new Set(keepIds().filter((id) => isLastKeptOfScene(state.photoById.get(id), counts))); // 장면의 마지막 한 장
  // 점수 낮은 순, 단 장면의 마지막 남은 사진은 맨 뒤로 미룬다
  const queue = keepIds().map((id) => state.photoById.get(id))
    .filter((photo) => !seenIds.includes(photo.id) && matchesCategory(photo))
    .sort((left, right) => (sceneProtected.has(left.id) - sceneProtected.has(right.id)) || left.score - right.score);
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
  const immersive = PICK_VIEWS.includes(viewName);
  document.body.classList.toggle('immersive', immersive);
  $('meta[name="theme-color"]').content = immersive ? '#24282A' : '#F8F9F7';
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
  showToast(t('switchedReviewer', escapeHtml(reviewer().name)), [], 2600);
  render();
};

let previousKeepCount = null; // 직전에 그린 남긴 장수 (목표를 맞추는 순간을 알아보려고)
function renderTopbar() {
  const ready = state.analysisDone;
  for (const button of document.querySelectorAll('#tabs button')) {
    const isCurrent = button.dataset.view === state.view || (state.view === 'analyze' && button.dataset.view === 'setup');
    if (isCurrent) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    if (button.dataset.view !== 'setup') button.disabled = !ready;
  }
  const keepCount = ready ? keepIds().length : 0;
  const target = state.settings.target;
  $('#tab-first').textContent = ready ? num(undecidedCount()) : '';
  $('#tab-keep').textContent = ready ? num(keepCount) : '';
  $('#tab-drop').textContent = ready ? num(dropIds().length) : '';
  $('#goal-count').textContent = num(keepCount);
  $('#goal-target').textContent = num(target);
  $('#goal').classList.toggle('hidden', !ready); // 사진을 불러오기 전 시작 화면에서는 '남긴 사진 0 / 목표'를 숨긴다
  $('#goal').classList.toggle('over', keepCount > target);
  $('#goal').classList.toggle('hit', ready && keepCount === target);
  // 고르는 중에 남긴 장수가 목표와 딱 맞게 바뀌는 순간만 (처음 열 때나 다른 화면에서는 아님)
  if (ready && PICK_VIEWS.includes(state.view) && previousKeepCount !== null && previousKeepCount !== keepCount && keepCount === target) celebrate();
  previousKeepCount = ready ? keepCount : null;
  $('#goal-note').textContent = keepCount > target ? t('goalOver', keepCount - target) : '';
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
    .map((key) => `<button data-category="${key}" aria-pressed="${state.categoryFilter === key}">${categoryLabel(key)}<span class="num">${num(counts[key] ?? 0)}</span></button>`).join('');
}
$('#filters').onclick = (event) => {
  const button = event.target.closest('[data-category]');
  if (!button) return;
  state.categoryFilter = button.dataset.category;
  render();
};
$('#compare-input').onchange = (event) => { state.compareMode = event.target.checked; event.target.blur(); render(); };

// 필름 띠에서 누른 사진(priorityIds)을 뺀 원래 순서로 큐를 만든다.
// 띠는 이 순서로 고정해서, 썸네일을 눌러도 띠 전체가 다시 섞이지 않게 한다.
function naturalQueue(buildQueue) {
  const savedPriorityIds = state.priorityIds;
  state.priorityIds = [];
  try { return buildQueue(); } finally { state.priorityIds = savedPriorityIds; }
}

const FILMSTRIP_VISIBLE = 16;
function renderFilmstrip(entries, currentIds = []) {
  // entries: [{ids, photo, count}] 원래 볼 순서. currentIds: 지금 무대에 떠 있는 사진
  const filmstrip = $('#filmstrip');
  const isCurrent = (entry) => entry.ids.some((id) => currentIds.includes(id));
  const currentIndex = entries.findIndex(isCurrent);
  // 지금 보는 사진이 앞쪽 16장 밖이면 그 근처부터 보여준다
  const start = currentIndex >= FILMSTRIP_VISIBLE ? currentIndex - 3 : 0;
  const visible = entries.slice(start, start + FILMSTRIP_VISIBLE);
  filmstrip.innerHTML = visible.map((entry, offset) => {
    const order = start + offset;
    const current = order === currentIndex;
    return `
    <button class="${current ? 'current' : ''}" data-order="${order}" data-ids="${escapeHtml(JSON.stringify(entry.ids))}" title="${escapeHtml(entry.photo.relPath)}" aria-label="${current ? t('filmCurrent') : t('filmNth', order + 1)}" ${current ? 'aria-current="true"' : ''}>
      <img src="${entry.photo.thumbUrl}" alt="" loading="lazy">${entry.count > 1 ? `<span class="count num">${entry.count}</span>` : ''}
    </button>`;
  }).join('') + (entries.length > start + visible.length ? `<span class="more num">${t('filmMore', entries.length - start - visible.length)}</span>` : '');
  // 지금 보는 썸네일이 띠 밖으로 밀려나 있으면 띠만 옆으로 넘긴다 (페이지는 움직이지 않게 scrollLeft로)
  const currentButton = $('button.current', filmstrip);
  if (currentButton) {
    const left = currentButton.getBoundingClientRect().left - filmstrip.getBoundingClientRect().left + filmstrip.scrollLeft;
    if (left < filmstrip.scrollLeft || left + currentButton.offsetWidth > filmstrip.scrollLeft + filmstrip.clientWidth) {
      filmstrip.scrollLeft = Math.max(0, left - 16);
    }
  }
}
$('#filmstrip').onclick = (event) => {
  const button = event.target.closest('button[data-ids]');
  if (!button || button.classList.contains('current')) return;
  // 맨 앞(원래 볼 차례) 사진을 누르면 따로 당겨 온 사진을 풀고 원래 순서로 돌아간다
  state.priorityIds = button.dataset.order === '0' ? [] : JSON.parse(button.dataset.ids);
  render();
};

function setNotice(html, alert = false) {
  const notice = $('#notice');
  notice.innerHTML = html;
  notice.classList.toggle('alert', alert);
}

function photoTagsHtml(photo, { showAdded }) {
  const tags = [];
  if (state.settings.useScore) tags.push(`<span class="tag num">${t('tagScore', photo.score)}</span>`);
  if (state.settings.useCategories) tags.push(`<span class="tag">${categoryLabel(categoryOf(photo))}</span>`);
  if (photo.groupId && state.view !== 'first') tags.push(`<span class="tag num">${t('tagSimilar', state.groups.get(photo.groupId).length)}</span>`);
  if (hasClosedEyes(photo)) tags.push(`<span class="tag flag"><i class="ph ph-eye-closed"></i> ${t(state.settings.preset === 'wedding' && !state.settings.penalizeClosedEyes ? 'tagEyesWedding' : 'tagEyes')}</span>`);
  const decision = decisionOf(photo.id);
  if (showAdded && decision?.addedAt) tags.push(`<span class="tag added">${t('tagReAdded', formatTime(decision.addedAt))}</span>`);
  // 취향 확인용으로 끼워 넣은 사진: 점수가 낮은 사진이 왜 일찍 나왔는지 알려준다
  if (state.view === 'first' && probeUnitKey === photo.id) tags.push(`<span class="tag">${t('tagProbe')}</span>`);
  if (state.view === 'distill' && isLastKeptOfScene(photo)) tags.push(`<span class="tag">${t('tagLastInScene')}</span>`);
  return tags.join('');
}

function renderPickView() {
  const mode = state.view;
  $('#compare-wrap').classList.toggle('hidden', mode !== 'distill');
  $('#criteria-button').classList.toggle('hidden', !state.settings.useScore);
  $('#cast-button').classList.toggle('hidden', !state.settings.useScore || !state.people.length);
  $('#compare-input').checked = state.compareMode;
  const stage = $('#stage');
  const keepCount = keepIds().length;
  const target = state.settings.target;

  if (mode === 'first') {
    renderFilters(state.photos);
    const queue = firstPassQueue();
    const remaining = queue.reduce((sum, unit) => sum + unit.ids.filter((id) => !statusOf(id)).length, 0);
    setNotice(t('noticeFirst', { remaining, scored: state.settings.useScore }));
    const head = queue[0];
    renderFilmstrip(naturalQueue(firstPassQueue).filter((unit) => unit.type !== 'divider').map((unit) => {
      const members = unit.ids.map((id) => state.photoById.get(id));
      return { ids: unit.ids, photo: members.find((photo) => photo.groupBest) ?? members[0], count: unit.ids.length };
    }), head?.ids ?? []);
    if (!head) {
      return renderEmpty(stage, t('firstDoneTitle'), t('firstDoneBody', { keep: keepCount, target }),
        [{ label: t('btnDistill'), primary: keepCount > target, run: () => setView('distill') }, { label: t('btnRescueDropped'), run: () => setView('rescue') }, { label: t('btnExport'), primary: keepCount <= target, run: () => setView('export') }]);
    }
    if (head.type === 'divider') return renderDivider(stage, head);
    if (head.type === 'group') return renderBurst(stage, head);
    const headPhoto = state.photoById.get(head.ids[0]);
    const firstKeptLookalikes = singleViewIds.has(headPhoto.id) ? [] : similarKeptPhotos(headPhoto);
    if (firstKeptLookalikes.length) return renderKeptCompare(stage, headPhoto, firstKeptLookalikes, 'first');
    return renderSingle(stage, headPhoto, queue.slice(1).flatMap((unit) => unit.ids).slice(0, 2), 'first');
  }

  if (mode === 'distill') {
    renderFilters(keepIds().map((id) => state.photoById.get(id)));
    const queue = distillQueue();
    setNotice(keepCount > target
      ? t('noticeDistillOver', keepCount - target)
      : keepCount < target
        ? t('noticeDistillUnder', { target, count: target - keepCount })
        : t('noticeDistillOk', target), keepCount > target);
    renderFilmstrip(naturalQueue(distillQueue).map((photo) => ({ ids: [photo.id], photo, count: 1 })), queue[0] ? [queue[0].id] : []);
    if (state.compareMode) {
      const pair = comparePair();
      if (pair) return renderCompare(stage, pair);
    }
    const head = queue[0];
    if (!head) {
      return renderEmpty(stage, t('distillDoneTitle'), t('distillDoneBody', keepCount),
        [{ label: t('btnDistillAgain'), primary: keepCount > target, run: () => { state.rounds[state.currentReviewer].distill = []; saveProjectSoon(); render(); } },
         { label: t('btnExport'), primary: keepCount <= target, run: () => setView('export') }]);
    }
    return renderSingle(stage, head, queue.slice(1, 3).map((photo) => photo.id), 'distill');
  }

  renderFilters(dropIds().map((id) => state.photoById.get(id)));
  const queue = rescueQueue();
  setNotice(t('noticeRescue'));
  const head = queue[0];
  renderFilmstrip(naturalQueue(rescueQueue).map((photo) => ({ ids: [photo.id], photo, count: 1 })), head ? [head.id] : []);
  if (!head) {
    return renderEmpty(stage, t('rescueDoneTitle'), t('rescueDoneBody', { keep: keepCount, target }),
      [{ label: t('btnRescueAgain'), run: () => { state.rounds[state.currentReviewer].rescue = []; saveProjectSoon(); render(); } },
       { label: keepCount > target ? t('btnDistill') : t('btnExport'), primary: true, run: () => setView(keepCount > target ? 'distill' : 'export') }]);
  }
  const rescueKeptLookalikes = singleViewIds.has(head.id) ? [] : similarKeptPhotos(head);
  if (rescueKeptLookalikes.length) return renderKeptCompare(stage, head, rescueKeptLookalikes, 'rescue');
  return renderSingle(stage, head, queue.slice(1, 3).map((photo) => photo.id), 'rescue');
}

// ---------- 앰비언트 배경: 지금 사진을 크게 흐려 화면 전체에 깐다 ----------
let ambientUrl = '';
function setAmbient(url) {
  if (!url || url === ambientUrl) return;
  ambientUrl = url;
  const [first, second] = document.querySelectorAll('#ambient .layer');
  const [showing, next] = first.classList.contains('on') ? [first, second] : [second, first];
  next.style.backgroundImage = `url("${url}")`;
  next.classList.add('on');
  showing.classList.remove('on');
}

// ---------- 목표 장수를 딱 맞췄을 때 축하 ----------
// 그리스 펜슬 빨강과 인화지 색만 쓴다. 움직임 줄이기 설정이면 건너뛴다
let confettiShooter = null;
let lastCelebratedAt = 0;
function celebrate() {
  if (performance.now() - lastCelebratedAt < 4000) return;
  lastCelebratedAt = performance.now();
  confettiShooter ??= createConfetti($('#celebrate'), { resize: true, useWorker: false, disableForReducedMotion: true });
  const colors = ['#C8261C', '#E8473C', '#F6D5D0', '#F8F9F7', '#F1B84B'];
  for (const [x, angle] of [[0.08, 60], [0.92, 120]]) {
    confettiShooter({ particleCount: 70, spread: 62, startVelocity: 52, angle, origin: { x, y: 0.95 }, colors, scalar: 0.95, ticks: 220 });
  }
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
  stage.innerHTML = `<div class="empty"><h2>${t('dividerTitle')}</h2>
    <p>${t('dividerBody', divider.count)}</p>
    <div class="actions"><button class="button primary" id="divider-continue">${t('dividerContinue')}</button><button class="button" id="divider-drop">${t('dividerDrop')}</button></div></div>`;
  $('#divider-continue').onclick = () => { state.dividerSeen[state.currentReviewer] = true; saveProjectSoon(); render(); };
  $('#divider-drop').onclick = () => {
    const pendingIds = state.units.filter((unit) => !unit.recommended).flatMap((unit) => unit.ids).filter((id) => !statusOf(id) && matchesCategory(state.photoById.get(id)));
    recordDecisions(pendingIds.map((id) => ({ id, status: 'drop' })), 'first-bulk');
    state.dividerSeen[state.currentReviewer] = true;
    showToast(t('droppedRest', pendingIds.length), [{ label: t('undo'), run: undoLast }]);
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
  const keepWord = stageName === 'distill' ? t('keepStill') : t('keep');
  const nextPhoto = preloadIds.length ? state.photoById.get(preloadIds[0]) : null;
  const nextRatio = nextPhoto?.width && nextPhoto?.height ? nextPhoto.width / nextPhoto.height : 1.5;
  stage.innerHTML = `
    <div class="deck">
      ${nextPhoto ? `<div class="frame peek" style="--ratio:${nextRatio}" aria-hidden="true"><div class="photo"><img src="${nextPhoto.thumbUrl}" alt=""></div></div>` : ''}
      <div class="frame current" style="--ratio:${ratio}">
        <div class="photo"><img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}"></div>
        ${mainFacesOf(photo).map((face) => `<div class="facebox ${face.eyesClosed ? 'closed' : ''}" style="left:${face.box.x * 100}%;top:${face.box.y * 100}%;width:${face.box.w * 100}%;height:${face.box.h * 100}%"></div>`).join('')}
        <div class="tags">${photoTagsHtml(photo, { showAdded: stageName === 'distill' })}<span class="tag path">${escapeHtml(photo.relPath)}</span></div>
      </div>
    </div>`;
  const frame = $('.frame.current', stage);
  const peek = $('.frame.peek', stage);
  if (nextPhoto) frame.dataset.nextId = nextPhoto.id;
  setAmbient(photo.thumbUrl);
  // 뒤에 있던 카드가 앞으로 나온 경우는 이미 제자리라 그대로 두고, 아니면 살짝 튀어 오르며 나타난다
  if (promotedPhotoId !== photo.id) motion(frame, { opacity: [0, 1], transform: ['translateY(14px) scale(.95)', 'translateY(0px) scale(1)'] }, { duration: 0.26, ease: [0.16, 1, 0.3, 1] });
  promotedPhotoId = null;
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
        showToast(t('rescuedOver', keepCount - state.settings.target), [{ label: t('goDistill'), run: () => setView('distill') }], 5500);
      }
    } else {
      recordDecisions([{ id: photo.id, status }], 'first');
    }
    state.priorityIds = [];
    render();
  };
  attachSwipe(frame, keepMark, dropMark, decide, peek);

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
    <button class="button decide" id="act-drop"><i class="ph ph-x"></i>${t('drop')}<kbd>←</kbd></button>
    ${undoButtonHtml()}
    <button class="button icon-only" id="act-zoom" title="${t('zoomTitle')}" aria-label="${t('zoomTitle')}"><i class="ph ph-magnifying-glass-plus"></i></button>
    <button class="button icon-only" id="act-faces" title="${t('facesTitle')}" aria-label="${t('facesTitle')}"><i class="ph ph-user-focus"></i></button>
    <button class="button decide keep" id="act-keep"><i class="ph ph-circle"></i>${keepWord}<kbd>→</kbd></button>`;
  const leave = (status) => leaveFrame(frame, status === 'keep' ? keepMark : dropMark, status, decide, { peek, nextId: nextPhoto?.id });
  $('#act-drop').onclick = () => leave('drop');
  $('#act-keep').onclick = () => leave('keep');
  $('#act-undo').onclick = undoLast;
  $('#act-zoom').onclick = toggleZoom;
  $('#act-faces').onclick = () => frame.classList.toggle('show-faces');
  currentKeys = { left: () => leave('drop'), right: () => leave('keep'), zoom: toggleZoom, faces: () => frame.classList.toggle('show-faces') };
}

// 결정하면 그리스 펜슬 표시를 그린 뒤 사진을 옆으로 날려 보낸다. 뒤에 있던 다음 사진은 스프링으로 앞으로 나온다
// fromX: 손가락으로 끌던 위치 (버튼과 키보드는 0)
let promotedPhotoId = null; // 뒤에서 앞으로 나온 사진 (다음 그리기에서 등장 애니메이션을 건너뛴다)
const SWIPE_LEAVE_MS = 320; // 결정을 반영하기까지 (펜슬 표시를 그리고 사진이 나가는 시간). 하루 수백 번 하는 동작이라 짧게
function leaveFrame(frame, mark, status, onDone, { peek = null, nextId = null, fromX = 0 } = {}) {
  if (frame.dataset.leaving) return;
  frame.dataset.leaving = '1';
  mark.style.opacity = '';
  mark.classList.add('drawn');
  const direction = status === 'keep' ? 1 : -1;
  const startTransform = `translateX(${fromX}px) rotate(${fromX / 40}deg)`;
  // 끌던 중이면 바로, 버튼이면 표시가 그려지는 걸 잠깐 보여준 뒤 날린다
  setTimeout(() => {
    motion(frame, { transform: [startTransform, `translateX(${direction * Math.max(window.innerWidth, 600) * 0.9}px) rotate(${direction * 14}deg)`], opacity: [1, 0] },
      { duration: 0.22, ease: [0.4, 0, 1, 1] });
    if (peek) {
      promotedPhotoId = nextId;
      // transform과 opacity만 움직인다 (filter는 매 프레임 다시 그려서 휴대폰에서 버벅인다)
      motion(peek, { transform: [peek.style.transform || 'translateY(16px) scale(.93)', 'translateY(0px) scale(1)'], opacity: [0.55, 1] },
        { type: 'spring', bounce: 0.1, duration: 0.3 });
    }
  }, fromX ? 0 : 110);
  setTimeout(() => onDone(status), fromX ? SWIPE_LEAVE_MS - 120 : SWIPE_LEAVE_MS);
}

// 손가락으로 끌기: 놓을 때 충분히 멀리 갔거나 빠르게 튕겼으면 결정, 아니면 스프링으로 제자리
const SWIPE_DISTANCE = 120;      // 이만큼(px) 끌면 결정
const SWIPE_FLICK_SPEED = 0.6;   // 이보다 빠르게(px/ms) 튕기면 짧게 끌어도 결정
function attachSwipe(frame, keepMark, dropMark, onDecide, peek = null) {
  let startX = 0, deltaX = 0, dragging = false, lastX = 0, lastTime = 0, velocity = 0;
  frame.addEventListener('pointerdown', (event) => {
    if (frame.classList.contains('zoomed') || event.button !== 0 || frame.dataset.leaving) return;
    dragging = true; startX = lastX = event.clientX; deltaX = 0; velocity = 0; lastTime = performance.now();
    frame.setPointerCapture(event.pointerId);
  });
  frame.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const now = performance.now();
    velocity = (event.clientX - lastX) / Math.max(1, now - lastTime);
    lastX = event.clientX; lastTime = now;
    deltaX = event.clientX - startX;
    frame.style.transform = `translateX(${deltaX}px) rotate(${deltaX / 40}deg)`;
    keepMark.style.opacity = clamp01(deltaX / 140);
    dropMark.style.opacity = clamp01(-deltaX / 140);
    // 끄는 만큼 뒤 사진이 조금씩 앞으로
    if (peek) {
      const progress = clamp01(Math.abs(deltaX) / 260);
      peek.style.transform = `translateY(${16 - progress * 8}px) scale(${0.93 + progress * 0.03})`;
    }
  });
  const finish = () => {
    if (!dragging) return;
    dragging = false;
    const flicked = Math.abs(velocity) > SWIPE_FLICK_SPEED && Math.abs(deltaX) > 30 && Math.sign(velocity) === Math.sign(deltaX);
    if (Math.abs(deltaX) > SWIPE_DISTANCE || flicked) {
      const status = deltaX > 0 ? 'keep' : 'drop';
      leaveFrame(frame, status === 'keep' ? keepMark : dropMark, status, onDecide, { peek, nextId: peek ? frame.dataset.nextId : null, fromX: deltaX });
      return;
    }
    motion(frame, { transform: [`translateX(${deltaX}px) rotate(${deltaX / 40}deg)`, 'translateX(0px) rotate(0deg)'] }, { type: 'spring', bounce: 0.2, duration: 0.35 });
    if (peek) motion(peek, { transform: [peek.style.transform || 'translateY(16px) scale(.93)', 'translateY(16px) scale(.93)'] }, { duration: 0.2, ease: [0.16, 1, 0.3, 1] });
    keepMark.style.opacity = ''; dropMark.style.opacity = '';
  };
  frame.addEventListener('pointerup', finish);
  frame.addEventListener('pointercancel', finish);
}

// ---------- 비슷한 컷 묶음 ----------
let burstSelection = { key: null, selected: new Set() };

function renderBurst(stage, unit) {
  keepOnlyUrls([]);
  const groupMembers = unit.ids.map((id) => state.photoById.get(id)).sort((left, right) => right.score - left.score);
  setAmbient(groupMembers[0]?.thumbUrl);
  // 묶음 밖에서 이미 남긴 비슷한 사진도 같이 보여준다 (따로 남긴 컷과 나란히 비교할 수 있게)
  const groupIdSet = new Set(unit.ids);
  const outsideKept = [];
  for (const member of groupMembers) {
    for (const lookalike of similarKeptPhotos(member)) {
      if (!groupIdSet.has(lookalike.id) && !outsideKept.includes(lookalike)) outsideKept.push(lookalike);
    }
  }
  const members = [...groupMembers, ...outsideKept.slice(0, 4)]; // 화면에 보여줄 사진 (묶음 사진, 묶음 밖 남긴 사진 순)
  if (burstSelection.key !== unit.key) {
    // 묶음 안에 이미 결정한 사진이 있으면 그 결정을 그대로 보여주고, 없으면 최고점 한 장을 미리 고른다
    const alreadyDecided = members.some((photo) => statusOf(photo.id));
    const initialIds = alreadyDecided
      ? members.filter((photo) => statusOf(photo.id) === 'keep').map((photo) => photo.id)
      : state.settings.useScore ? members.filter((photo) => photo.groupBest).map((photo) => photo.id) : [];
    burstSelection = { key: unit.key, selected: new Set(initialIds) };
  }
  // 좁은 화면(휴대폰 세로)에서는 사진을 크게 보이게 열 수를 줄인다
  const isNarrow = matchMedia('(max-width: 760px)').matches;
  const columns = isNarrow ? (members.length <= 3 ? 1 : 2) : members.length <= 4 ? 2 : members.length <= 9 ? 3 : 4;
  stage.innerHTML = `
    <div class="burst">
      <div class="burst-head">
        <h2>${t('burstTitle', groupMembers.length)}${outsideKept.length ? t('burstWithKept', Math.min(4, outsideKept.length)) : ''}</h2>
        ${probeUnitKey === unit.key ? `<span class="tag">${t('tagProbe')}</span>` : ''}
        <span class="hint">${t('burstHint')}${state.settings.useScore ? t('burstHintScored') : ''}</span>
        <span class="spacer"></span>
        <button class="button" id="burst-expand">${t('burstExpand')}</button>
      </div>
      <div class="burst-grid" style="--cols:${columns};--rows:${Math.min(3, Math.ceil(members.length / columns))}">
        ${members.map((photo, index) => `
          <button class="shot" data-id="${escapeHtml(photo.id)}" aria-pressed="${burstSelection.selected.has(photo.id)}" title="${t('burstShotTitle', escapeHtml(photo.relPath))}">
            <img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}" loading="lazy">
            ${index < 9 ? `<span class="index num">${index + 1}</span>` : ''}
            <span class="tags">${state.settings.useScore ? `<span class="tag num">${t('tagScore', photo.score)}</span>` : ''}${hasClosedEyes(photo) ? `<span class="tag flag"><i class="ph ph-eye-closed"></i> ${t('tagEyesShort')}</span>` : ''}${statusOf(photo.id) ? `<span class="tag">${t(statusOf(photo.id) === 'keep' ? 'tagAlreadyKept' : 'tagAlreadyDropped')}</span>` : ''}</span>
          </button>`).join('')}
      </div>
    </div>`;
  for (const shot of stage.querySelectorAll('.shot')) shot.append(cloneMark('keep'));
  const updateConfirm = () => { $('#act-confirm').innerHTML = `<i class="ph ph-circle"></i>${t('keepCount', burstSelection.selected.size)}<kbd>→</kbd>`; };
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
    // 묶음 사진은 고른 대로, 묶음 밖 남긴 사진은 선택을 풀었을 때만 뺀다
    const changes = members
      .filter((photo) => groupIdSet.has(photo.id) || !burstSelection.selected.has(photo.id))
      .map((photo) => ({ id: photo.id, status: burstSelection.selected.has(photo.id) ? 'keep' : 'drop' }));
    // 고른 사진은 안 고른 사진마다 한 번씩 이긴 것으로 기록 (다 뺐으면 비교 정보가 없다)
    const winnerIds = members.filter((photo) => burstSelection.selected.has(photo.id)).map((photo) => photo.id);
    const loserIds = changes.filter((change) => change.status === 'drop').map((change) => change.id);
    recordDecisions(changes, 'first-burst', winnerIds.flatMap((winnerId) => loserIds.map((loserId) => ({ winnerId, loserId }))));
    burstSelection = { key: null, selected: new Set() };
    render();
  };
  const dropAll = () => { burstSelection.selected.clear(); confirm(); };
  $('#controls').innerHTML = `
    <button class="button decide" id="act-dropall"><i class="ph ph-x"></i>${t('dropAll')}<kbd>←</kbd></button>
    ${undoButtonHtml()}
    <button class="button decide keep" id="act-confirm"></button>`;
  updateConfirm();
  $('#act-dropall').onclick = dropAll;
  $('#act-confirm').onclick = confirm;
  $('#act-undo').onclick = undoLast;
  currentKeys = { left: dropAll, right: confirm, number: (digit) => { const photo = members[digit - 1]; if (photo) toggleMember(photo.id); } };
}

// ---------- 이미 남긴 비슷한 사진과 비교 ----------
const SIMILAR_KEPT_HASH_LOOSE = 26;            // 비교를 권할 때는 묶기보다 조금 너그럽게 (결정은 사람이 하므로)
const SIMILAR_KEPT_TIME_GAP_MS = 10 * 60_000;  // 촬영 시각이 10분 넘게 떨어지면 비교하지 않는다
const singleViewIds = new Set();               // 사용자가 '한 장만 보기'를 고른 사진 (이번 방문 동안)
let keptCompareCursor = { photoId: null, index: 0 }; // 비슷한 남긴 사진이 여러 장일 때 지금 보여주는 순번

function similarKeptPhotos(photo) {
  if (!photo?.analysis || photo.analysis.failed) return [];
  return keepIds()
    .filter((id) => id !== photo.id)
    .map((id) => state.photoById.get(id))
    .filter((other) => {
      if (photo.takenAt && other.takenAt && Math.abs(photo.takenAt - other.takenAt) > SIMILAR_KEPT_TIME_GAP_MS) return false;
      if (other.analysis.failed) return false;
      const distance = hashDistance(photo.analysis.hash, other.analysis.hash);
      if (distance <= SIMILAR_HASH_DISTANCE) return true;
      return distance <= SIMILAR_KEPT_HASH_LOOSE && photo.analysis.tiny && other.analysis.tiny && correlation(photo.analysis.tiny, other.analysis.tiny) >= SIMILAR_CORRELATION;
    })
    .sort((left, right) => hashDistance(photo.analysis.hash, left.analysis.hash) - hashDistance(photo.analysis.hash, right.analysis.hash));
}

function renderKeptCompare(stage, currentPhoto, keptLookalikes, stageName) {
  stagePhotoId = currentPhoto.id;
  if (keptCompareCursor.photoId !== currentPhoto.id) keptCompareCursor = { photoId: currentPhoto.id, index: 0 };
  const keptPhoto = keptLookalikes[keptCompareCursor.index % keptLookalikes.length];
  keepOnlyUrls([currentPhoto.id, keptPhoto.id]);
  setAmbient(currentPhoto.thumbUrl);
  stage.innerHTML = `
    <div class="kept-compare">
      <div class="burst-head">
        <h2>${t('keptCompareTitle')}</h2>
        <span class="hint">${t('keptCompareHint')}</span>
        <span class="spacer"></span>
        ${keptLookalikes.length > 1 ? `<button class="button" id="kept-next">${t('keptCompareNext', { index: (keptCompareCursor.index % keptLookalikes.length) + 1, total: keptLookalikes.length })}</button>` : ''}
        <button class="button" id="kept-single">${t('keptCompareSingle')}</button>
      </div>
      <div class="compare">
        ${[[currentPhoto, 'labelCurrent'], [keptPhoto, 'labelKept']].map(([photo, labelKey]) => `
          <button class="shot" data-id="${escapeHtml(photo.id)}" title="${t('burstShotTitle', escapeHtml(photo.relPath))}">
            <img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}">
            <span class="index">${t(labelKey)}</span>
            <span class="tags">${photoTagsHtml(photo, { showAdded: false })}</span>
          </button>`).join('')}
      </div>
    </div>`;
  const shots = stage.querySelectorAll('.shot');
  shots.forEach((shot) => {
    const photo = state.photoById.get(shot.dataset.id);
    fullImageUrl(photo).then((url) => { const image = $('img', shot); if (image.isConnected) image.src = url; });
    shot.onclick = () => openViewer(photo); // 누르면 크게 보기 (결정은 아래 버튼으로)
  });
  let deciding = false; // 표시를 그리는 동안 들어온 두 번째 입력은 무시한다 (같은 결정이 두 번 기록되지 않게)
  const finish = (changes, keepMarkOn, preferences = []) => {
    if (deciding) return;
    deciding = true;
    if (keepMarkOn !== null) { const mark = cloneMark('keep'); shots[keepMarkOn].append(mark); mark.classList.add('drawn'); }
    setTimeout(() => {
      const undoEntry = changes.length ? recordDecisions(changes, `${stageName}-kept-compare`, preferences) : emptyUndo();
      if (!changes.length) state.undoStack.push(undoEntry);
      if (stageName === 'rescue') markRoundSeen('rescue', [currentPhoto.id], undoEntry);
      render();
    }, keepMarkOn !== null ? 260 : 0);
  };
  // 왼쪽: 지금 사진 빼기. 다시 보기에서는 이미 빠진 사진이라 결정은 그대로 두고 넘어간다
  const dropCurrent = () => finish(stageName === 'rescue' ? [] : [{ id: currentPhoto.id, status: 'drop' }], null, [{ winnerId: keptPhoto.id, loserId: currentPhoto.id }]);
  const keepBoth = () => finish([{ id: currentPhoto.id, status: 'keep' }], 0);
  const swapToCurrent = () => finish([{ id: currentPhoto.id, status: 'keep' }, { id: keptPhoto.id, status: 'drop' }], 0, [{ winnerId: currentPhoto.id, loserId: keptPhoto.id }]);
  $('#kept-single').onclick = () => { singleViewIds.add(currentPhoto.id); render(); };
  if ($('#kept-next')) $('#kept-next').onclick = () => { keptCompareCursor.index++; render(); };
  $('#controls').innerHTML = `
    <button class="button decide" id="act-drop-current" aria-label="${t('dropCurrent')}"><i class="ph ph-x"></i><span class="label-long">${t('dropCurrent')}</span><span class="label-short">${t('dropCurrentShort')}</span><kbd>←</kbd></button>
    <button class="button decide" id="act-keep-both" aria-label="${t('keepBoth')}"><span class="label-long">${t('keepBoth')}</span><span class="label-short">${t('keepBothShort')}</span><kbd>↓</kbd></button>
    ${undoButtonHtml()}
    <button class="button decide keep" id="act-swap" aria-label="${t('swapToCurrent')}"><i class="ph ph-circle"></i><span class="label-long">${t('swapToCurrent')}</span><span class="label-short">${t('swapShort')}</span><kbd>→</kbd></button>`;
  $('#act-drop-current').onclick = dropCurrent;
  $('#act-keep-both').onclick = keepBoth;
  $('#act-swap').onclick = swapToCurrent;
  $('#act-undo').onclick = undoLast;
  currentKeys = { left: dropCurrent, right: swapToCurrent, down: keepBoth };
}

// ---------- 두 장 비교 ----------
function renderCompare(stage, [leftPhoto, rightPhoto]) {
  keepOnlyUrls([leftPhoto.id, rightPhoto.id]);
  setAmbient(leftPhoto.thumbUrl);
  stage.innerHTML = `<div class="compare">${[leftPhoto, rightPhoto].map((photo, index) => `
    <button class="shot" data-side="${index}" aria-label="${index === 0 ? t('compareKeepLeft') : t('compareKeepRight')}">
      <img src="${photo.thumbUrl}" alt="${escapeHtml(photo.name)}">
      <span class="choose-key"><kbd>${index === 0 ? '←' : '→'}</kbd></span>
      <span class="tags">${photoTagsHtml(photo, { showAdded: true })}</span>
    </button>`).join('')}</div>`;
  const shots = stage.querySelectorAll('.shot');
  shots.forEach((shot, index) => {
    shot.append(cloneMark('keep'));
    fullImageUrl(index === 0 ? leftPhoto : rightPhoto).then((url) => { const image = $('img', shot); if (image.isConnected) image.src = url; });
  });
  let deciding = false; // 표시를 그리는 동안 들어온 두 번째 입력은 무시한다
  const keepOne = (winnerIndex) => {
    if (deciding) return;
    deciding = true;
    const winner = winnerIndex === 0 ? leftPhoto : rightPhoto;
    const loser = winnerIndex === 0 ? rightPhoto : leftPhoto;
    $('.mark', shots[winnerIndex]).classList.add('drawn');
    setTimeout(() => {
      const undoEntry = recordDecisions([{ id: loser.id, status: 'drop' }], 'distill-compare', [{ winnerId: winner.id, loserId: loser.id }]);
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
    <button class="button decide" id="act-left">${t('keepLeft')}<kbd>←</kbd></button>
    <button class="button decide" id="act-both">${t('keepBoth')}<kbd>↓</kbd></button>
    ${undoButtonHtml()}
    <button class="button decide" id="act-right">${t('keepRight')}<kbd>→</kbd></button>`;
  $('#act-left').onclick = () => keepOne(0);
  $('#act-right').onclick = () => keepOne(1);
  $('#act-both').onclick = keepBoth;
  $('#act-undo').onclick = undoLast;
  currentKeys = { left: () => keepOne(0), right: () => keepOne(1), down: keepBoth };
}

// ---------- 팝오버를 지원하지 않는 브라우저 (iOS 16 이하 Safari, Firefox 124 이하) ----------
// 점수 기준과 주인공 패널에 쓰는 만큼만 흉내 낸다: 버튼으로 열고 닫기, 바깥을 누르거나 Esc로 닫기, toggle 이벤트.
// 열린 상태는 popover-open 클래스로 나타내고, 숨기기와 위치는 styles.css의 @supports 규칙이 맡는다
if (!('popover' in HTMLElement.prototype)) {
  const setPanelOpen = (panel, open) => {
    if (panel.classList.contains('popover-open') === open) return;
    panel.classList.toggle('popover-open', open);
    panel.dispatchEvent(Object.assign(new Event('toggle'), { newState: open ? 'open' : 'closed', oldState: open ? 'closed' : 'open' }));
  };
  for (const panel of document.querySelectorAll('[popover]')) {
    panel.showPopover = () => setPanelOpen(panel, true);
    panel.hidePopover = () => setPanelOpen(panel, false);
    panel.togglePopover = () => setPanelOpen(panel, !panel.classList.contains('popover-open'));
  }
  document.addEventListener('click', (event) => {
    const opener = event.target.closest('[popovertarget]');
    const target = opener ? document.getElementById(opener.getAttribute('popovertarget')) : null;
    for (const panel of document.querySelectorAll('[popover].popover-open')) {
      if (panel !== target && !panel.contains(event.target)) panel.hidePopover();
    }
    target?.togglePopover();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') for (const panel of document.querySelectorAll('[popover].popover-open')) panel.hidePopover();
  });
}

// ---------- 점수 기준 패널 ----------
$('#criteria').addEventListener('toggle', (event) => {
  if (event.newState !== 'open') return;
  const model = activeModel();
  const totalPositive = model.theta.reduce((sum, weight) => sum + Math.max(0, weight), 0) || 1;
  const currentPhoto = currentPhotoOnStage();
  const sourceText = model.source === 'learned'
    ? t('criteriaLearned', { name: escapeHtml(reviewer().name), decisions: model.decisions ?? model.trainedOn, pairs: model.pairs ?? 0, every: LEARN_EVERY })
    : model.source === 'server'
      ? t('criteriaServer', { preset: presetText(state.settings.preset).label, min: LEARN_MIN_DECISIONS })
      : t('criteriaPreset', { preset: presetText(state.settings.preset).label, perClass: LEARN_MIN_PER_CLASS, min: LEARN_MIN_DECISIONS });
  $('#criteria').innerHTML = `
    <h2>${t('criteriaTitle')}</h2>
    <p class="hint source">${sourceText}</p>
    <label class="check criteria-eyes"><input type="checkbox" id="criteria-eyes" ${state.settings.penalizeClosedEyes ? 'checked' : ''}><span>${t('eyesCheck')}<br><span class="hint">${t('eyesHint')}</span></span></label>
    <table class="num">
      <tr><td class="hint">${t('colItem')}</td><td class="hint">${t('colWeight')}</td>${currentPhoto ? `<td class="hint">${t('colThis')}</td>` : ''}</tr>
      ${FEATURE_KEYS.map((key, index) => `<tr><td>${featureLabel(key)}</td><td>${Math.round((Math.max(0, model.theta[index]) / totalPositive) * 100)}%</td>${currentPhoto?.features ? `<td>${Math.round(currentPhoto.features[key] * 100)}</td>` : ''}</tr>`).join('')}
    </table>
    <p class="hint source">${t(state.settings.penalizeClosedEyes ? 'criteriaNoteEyesOn' : 'criteriaNote')}</p>
    <h2 class="diag-title">${t('diagTitle')}</h2>
    <p class="hint source">${diagnosticsText()}</p>`;
  $('#criteria-eyes').onchange = (changeEvent) => {
    applyClosedEyesSetting(changeEvent.target.checked);
    $('#criteria').hidePopover();
    $('#criteria').showPopover(); // 바뀐 비중으로 패널을 다시 그린다
  };
});

// ---------- 주인공 고르기 ----------
// 자주 나온 얼굴을 보여주고 누르면 주인공으로 넣고 뺀다. 얼굴 사진은 썸네일에서 잘라 이 기기 안에서만 만든다
const faceCropUrls = new Map(); // 사람 key에서 얼굴 사진 data URL로
async function faceCropUrl(person) {
  if (faceCropUrls.has(person.key)) return faceCropUrls.get(person.key);
  // 가장 크게 나온 얼굴을 대표로
  let best = null;
  for (const { photoId, faceIndex } of person.faces) {
    const photo = state.photoById.get(photoId);
    const face = photo?.analysis?.faces?.[faceIndex];
    if (face && photo.thumbUrl && (!best || face.box.h > best.face.box.h)) best = { photo, face };
  }
  if (!best) return '';
  const image = new Image();
  image.src = best.photo.thumbUrl;
  await image.decode();
  const { box } = best.face;
  const side = Math.max(box.w * image.naturalWidth, box.h * image.naturalHeight) * 1.5; // 얼굴보다 조금 넓게
  const centerX = (box.x + box.w / 2) * image.naturalWidth, centerY = (box.y + box.h / 2) * image.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 112;
  canvas.getContext('2d').drawImage(image, centerX - side / 2, centerY - side / 2, side, side, 0, 0, 112, 112);
  const url = canvas.toDataURL('image/jpeg', 0.85);
  faceCropUrls.set(person.key, url);
  return url;
}

async function renderCastPanel() {
  const panel = $('#cast');
  const castKeys = new Set(effectiveCastKeys());
  const candidates = state.people.slice(0, CAST_CANDIDATES);
  const crops = await Promise.all(candidates.map((person) => faceCropUrl(person).catch(() => '')));
  panel.innerHTML = `
    <h2>${t('castTitle')}</h2>
    <p class="hint source">${t(state.castKeys ? 'castHint' : 'castHintAuto')}</p>
    <div class="cast-grid">
      ${candidates.map((person, index) => `
        <button class="cast-face" data-key="${escapeHtml(person.key)}" aria-pressed="${castKeys.has(person.key)}" aria-label="${t('castPersonAria', { order: index + 1, count: person.photoIds.size })}">
          ${crops[index] ? `<img src="${crops[index]}" alt="">` : '<span class="cast-blank"></span>'}
          <span class="num">${t('castPhotos', person.photoIds.size)}</span>
        </button>`).join('')}
    </div>
    <p class="hint source">${t('castNote')}</p>`;
  for (const button of panel.querySelectorAll('.cast-face')) {
    button.onclick = () => {
      const next = new Set(effectiveCastKeys());
      if (next.has(button.dataset.key)) next.delete(button.dataset.key); else next.add(button.dataset.key);
      applyCast([...next]);
      button.setAttribute('aria-pressed', next.has(button.dataset.key));
      $('.hint.source', panel).textContent = t('castHint');
    };
  }
}
$('#cast').addEventListener('toggle', (event) => { if (event.newState === 'open') renderCastPanel(); });

// 주인공이 바뀌면 점수 항목과 학습을 다시 계산한다
function applyCast(castKeys) {
  state.castKeys = castKeys;
  computeFeatures();
  for (const reviewerKey of Object.keys(state.models)) {
    if (state.models[reviewerKey] && !trainModel(reviewerKey)) state.models[reviewerKey] = null;
  }
  rescoreAll();
  buildUnits();
  saveProjectSoon();
  render();
}

// 고르는 도중에 눈 감음 감점을 켜거나 끄면, 항목과 기본 모델을 다시 만들고 학습도 다시 한다
function applyClosedEyesSetting(penalize) {
  state.settings.penalizeClosedEyes = penalize;
  $('#eyes-input').checked = penalize;
  computeFeatures();
  state.priorModel = modelFromWeights(preset().weights, 'preset', penalize);
  for (const reviewerKey of Object.keys(state.models)) {
    if (state.models[reviewerKey] && !trainModel(reviewerKey)) state.models[reviewerKey] = null;
  }
  rescoreAll();
  buildUnits();
  saveProjectSoon();
  render();
  showToast(t(penalize ? 'eyesOnToast' : 'eyesOffToast'), [], 3000);
}
// 분석 정보: 얼굴 검출이 어떤 방식으로 돌았고 얼마나 찾았는지 (문제 제보를 받을 때 확인용)
function diagnosticsText() {
  const analyzedWithFaces = state.photos.filter((photo) => photo.analysis?.faces);
  const photosWithFaces = analyzedWithFaces.filter((photo) => mainFacesOf(photo).length > 0).length;
  const lines = [];
  lines.push(faceEngine.backend === 'none' || !analyzedWithFaces.length ? t('diagNoFace') : t('diagBackend', { backend: faceEngine.backend === 'webgl' ? 'WebGL' : 'CPU', float32: faceEngine.float32 }));
  if (faceEngine.crossCheck) lines.push(t('diagCrossCheck', faceEngine.crossCheck));
  if (analyzedWithFaces.length) lines.push(t('diagFaces', { found: photosWithFaces, total: analyzedWithFaces.length }));
  if (faceEngine.darkRetries) lines.push(t('diagDark', { hits: faceEngine.darkRetryHits, tries: faceEngine.darkRetries }));
  // 같은 사람 알아보기와 미적 점수
  if (analyzedWithFaces.length) {
    const judgedFaces = analyzedWithFaces.flatMap((photo) => photo.analysis.faces).filter((face) => face.eyesJudged);
    lines.push(t('diagEyes', {
      mesh: judgedFaces.filter((face) => face.eyeMethod === 'mesh').length,
      ear: judgedFaces.filter((face) => face.eyeMethod !== 'mesh').length,
      status: eyeEngine.status,
      msPerFace: eyeEngine.faces ? Math.round(eyeEngine.ms / eyeEngine.faces) : null,
    }));
  }
  if (analyzedWithFaces.length) lines.push(t('diagPeople', { people: state.people.filter((person) => person.photoIds.size >= 2).length, cast: effectiveCastKeys().length }));
  const aestheticCount = state.photos.filter((photo) => photo.analysis?.aesthetic).length;
  lines.push(aestheticCount ? t('diagAesthetic', { scored: aestheticCount, total: state.photos.length })
    : aestheticModel && faceEngine.backend !== 'webgl' ? t('diagAestheticCpu') : t('diagAestheticOff'));
  if (sceneCount() >= 2) lines.push(t('diagScenes', sceneCount()));
  // 점수 적중률: 결정할 때 보이던 점수로 계산 (50%면 무작위 수준)
  if (state.settings.useScore) {
    const accuracy = scoreAccuracy(state.currentReviewer);
    lines.push(accuracy.auc === null ? t('diagAccuracyWait') : t('diagAccuracy', { percent: Math.round(accuracy.auc * 100), count: accuracy.decisions }));
    if (accuracy.pairTotal) lines.push(t('diagPairAccuracy', { hits: Math.round(accuracy.pairHits), total: accuracy.pairTotal }));
  }
  lines.push(t('diagHint'));
  return lines.map(escapeHtml).join('<br>');
}

function currentPhotoOnStage() {
  return ($('#stage .frame') || $('#stage .kept-compare')) && stagePhotoId ? state.photoById.get(stagePhotoId) : null;
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
  const lines = [t('listHeader', photos.length), t('listFolder', state.folderName), t('listCreated', formatTime(Date.now())), '', ...photos.map((photo) => photo.relPath)];
  if (closedEyePhotos.length) lines.push('', t('listEyes', { count: closedEyePhotos.length, wedding: state.settings.preset === 'wedding' && !state.settings.penalizeClosedEyes }), ...closedEyePhotos.map((photo) => photo.relPath));
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
    <h1>${t('exportTitle')}</h1>
    <p class="summary num">${bothDecided
      ? t('summaryBoth', { nameA, nameB, keepA: keepA.length, keepB: keepB.length, both: bothCount, onlyA, onlyB })
      : t('summaryOne', { name: escapeHtml(reviewer().name), count: keepIds().length })}</p>

    <section>
      <h2>${t('saveTitle')}</h2>
      <p class="hint">${canPickFolder
        ? t('saveHintFolder')
        : canShareFiles
          ? t('saveHintShare', SHARE_BATCH_SIZE)
          : t('saveHintDownload')}</p>
      <div class="row">
        ${bothDecided ? `<div class="segmented" id="basis-options" role="radiogroup" aria-label="${t('basisAria')}">
          <label><input type="radio" name="basis" value="current" ${exportBasis === 'current' ? 'checked' : ''}>${t('basisCurrent', escapeHtml(reviewer().name))}</label>
          <label><input type="radio" name="basis" value="both" ${exportBasis === 'both' ? 'checked' : ''}>${t('basisBoth')}</label>
          <label><input type="radio" name="basis" value="either" ${exportBasis === 'either' ? 'checked' : ''}>${t('basisEither')}</label>
        </div>` : ''}
      </div>
      <div class="row">
        ${canPickFolder ? `<button class="button primary" id="export-copy" ${ids.length ? '' : 'disabled'}><i class="ph ph-folder-simple-plus"></i>${t('copyToFolder', ids.length)}</button>` : ''}
        ${!canPickFolder && canShareFiles ? `<button class="button primary" id="export-share" ${ids.length ? '' : 'disabled'}><i class="ph ph-share-network"></i>${shareButtonLabel(ids)}</button>` : ''}
        ${!canPickFolder ? `<button class="button${canShareFiles ? '' : ' primary'}" id="export-download" ${ids.length ? '' : 'disabled'}><i class="ph ph-download-simple"></i>${t('downloadEach')}</button>` : ''}
        <button class="button" id="export-list" ${ids.length ? '' : 'disabled'}><i class="ph ph-copy"></i>${t('copyList')}</button>
        <span class="hint num" id="export-progress"></span>
      </div>
      <div class="row">
        <label for="export-target" class="hint">${t('targetLabel')}</label><input type="number" id="export-target" min="1" value="${target}">
        <span class="hint">${ids.length > target ? t('targetOver', ids.length - target) : ids.length < target ? t('targetUnder', target - ids.length) : t('targetExact')}${closedEyeCount ? t('eyesInList', closedEyeCount) : ''}</span>
      </div>
    </section>

    <section>
      <h2>${t('mergeTitle')}</h2>
      <p class="hint">${t('mergeHint')}</p>
      <div class="row">
        <input type="text" id="rename-a" value="${nameA}" style="width:120px" aria-label="${t('nameAAria')}">
        <input type="text" id="rename-b" value="${nameB}" style="width:120px" aria-label="${t('nameBAria')}">
        <button class="button" id="decisions-save"><i class="ph ph-download-simple"></i>${t('saveDecisions', escapeHtml(reviewer().name))}</button>
        <button class="button" id="decisions-load"><i class="ph ph-upload-simple"></i>${t('loadDecisions')}</button>
        <input type="file" id="decisions-file" accept="application/json" hidden>
      </div>
    </section>

    <section>
      <h2>${t('pickedTitle')} <span class="hint num">${t('pickedCount', ids.length)}</span></h2>
      ${ids.length ? '' : `<p class="hint">${t('noneKept')}</p>`}
      <div class="contact">
        ${ids.map((id) => { const photo = state.photoById.get(id); const decision = decisionOf(id); return `
          <figure data-id="${escapeHtml(id)}">
            <button class="thumb" aria-label="${t('viewLarge', escapeHtml(photo.name))}"><img src="${photo.thumbUrl}" alt="" loading="lazy"></button>
            <figcaption><span title="${escapeHtml(photo.relPath)}">${escapeHtml(photo.name)}</span>${hasClosedEyes(photo) ? `<i class="ph ph-eye-closed flagged" title="${t('tagEyes')}"></i>` : ''}${decision?.addedAt ? `<i class="ph ph-arrow-u-up-left" title="${t('reAddedTitle')}"></i>` : ''}${exportBasis === 'current' ? `<button class="remove" title="${t('removeTitle')}" aria-label="${t('removeTitle')}"><i class="ph ph-x"></i></button>` : ''}</figcaption>
          </figure>`; }).join('')}
      </div>
    </section>`;

  if ($('#basis-options')) $('#basis-options').onchange = (event) => { exportBasis = event.target.value; render(); };
  $('#export-target').onchange = (event) => { state.settings.target = Math.max(1, parseInt(event.target.value, 10) || 1); saveProjectSoon(); render(); };
  $('#export-list').onclick = async () => {
    // http로 연 주소나 권한이 막힌 브라우저는 클립보드를 쓸 수 없다. 그때는 목록 파일로 내려받는다
    try {
      await navigator.clipboard.writeText(selectionListText(ids));
      showToast(t('listCopied'));
    } catch {
      downloadBlob(new Blob([selectionListText(ids)], { type: 'text/plain' }), t('listFileName'));
      showToast(t('listCopyFailed'));
    }
  };
  if ($('#export-copy')) $('#export-copy').onclick = () => copySelection(ids);
  if ($('#export-download')) $('#export-download').onclick = () => downloadSelection(ids);
  if ($('#export-share')) $('#export-share').onclick = () => shareNextBatch(ids);
  for (const figure of wrap.querySelectorAll('.contact figure')) {
    $('.thumb', figure).onclick = () => openViewer(state.photoById.get(figure.dataset.id));
    const removeButton = $('.remove', figure);
    if (removeButton) removeButton.onclick = () => { recordDecisions([{ id: figure.dataset.id, status: 'drop' }], 'export-remove'); render(); };
  }
  $('#rename-a').onchange = (event) => { state.reviewers.a.name = event.target.value.trim() || t('defaultNameA'); saveProjectSoon(); render(); };
  $('#rename-b').onchange = (event) => { state.reviewers.b.name = event.target.value.trim() || t('defaultNameB'); saveProjectSoon(); render(); };
  $('#decisions-save').onclick = () => {
    const payload = { app: 'moment-select', version: 2, folder: state.folderName, reviewer: reviewer().name, exportedAt: Date.now(), decisions: reviewer().decisions };
    downloadBlob(new Blob([JSON.stringify(payload)], { type: 'application/json' }), t('decisionFileName', reviewer().name));
  };
  $('#decisions-load').onclick = () => $('#decisions-file').click();
  $('#decisions-file').onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (payload.app !== 'moment-select') throw new Error(t('notDecisionFile'));
      const otherKey = state.currentReviewer === 'a' ? 'b' : 'a';
      const matched = Object.fromEntries(Object.entries(payload.decisions).filter(([id]) => state.photoById.has(id)));
      state.reviewers[otherKey].decisions = matched;
      if (payload.reviewer) state.reviewers[otherKey].name = payload.reviewer;
      saveProjectSoon();
      showToast(t('decisionsLoaded', { name: escapeHtml(state.reviewers[otherKey].name), count: Object.keys(matched).length, skipped: Object.keys(matched).length < Object.keys(payload.decisions).length }));
      render();
    } catch (error) {
      showToast(t('decisionsLoadError', escapeHtml(error.message)));
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
    const folder = await destination.getDirectoryHandle(`${t('exportFolderPrefix')}${timestampForFolder()}`, { create: true });
    for (const [index, photo] of photos.entries()) {
      const source = await getPhotoFile(photo);
      const writable = await (await folder.getFileHandle(outputName(photo), { create: true })).createWritable();
      await source.stream().pipeTo(writable); // 큰 원본도 메모리에 다 올리지 않고 흘려서 복사
      progress.textContent = t('copyingProgress', { done: index + 1, total: photos.length });
    }
    const listWritable = await (await folder.getFileHandle(t('listFileName'), { create: true })).createWritable();
    await listWritable.write(selectionListText(ids));
    await listWritable.close();
    progress.textContent = '';
    showToast(t('copiedToast', { count: photos.length, path: `${escapeHtml(destination.name)}/${escapeHtml(folder.name)}` }), [], 6000);
  } catch (error) {
    progress.textContent = '';
    if (error.name !== 'AbortError') showToast(t('copyError', escapeHtml(error.message)));
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
    progress.textContent = t('downloadingProgress', { done: index + 1, total: photos.length });
    await new Promise((resolve) => setTimeout(resolve, 350)); // 브라우저가 연속 다운로드를 막지 않게 간격을 둔다
  }
  downloadBlob(new Blob([selectionListText(ids)], { type: 'text/plain' }), t('listFileName'));
  progress.textContent = '';
}

// 공유 시트 저장: 공유 창은 버튼을 누를 때마다 한 번만 열 수 있어서 묶음 단위로 이어서 연다
let shareProgress = { signature: '', nextIndex: 0 }; // 지금 저장 중인 목록과 다음에 저장할 위치
function shareButtonLabel(ids) {
  const signature = ids.join('|');
  if (shareProgress.signature !== signature) shareProgress = { signature, nextIndex: 0 };
  if (shareProgress.nextIndex >= ids.length) return t('shareAllDone');
  const end = Math.min(ids.length, shareProgress.nextIndex + SHARE_BATCH_SIZE);
  return t('shareRange', { start: shareProgress.nextIndex + 1, end, total: ids.length });
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
    if (error.name !== 'AbortError') showToast(t('shareError', escapeHtml(error.message)));
  }
  render();
}

/* =====================================================================
   언어 선택 (한국어, 영어)
   ===================================================================== */
function applyLanguage() {
  applyStaticText();
  $('#language-select').value = getLanguage();
  $('#pick-note').textContent = t(pickNoteKey);
  if (resumeText) $('#resume-name').textContent = resumeText();
  // 사람 이름이 아직 기본값이면 언어에 맞게 바꾼다 (직접 입력한 이름은 그대로)
  for (const [inputId, reviewerKey, defaultKey] of [['#name-a', 'a', 'defaultNameA'], ['#name-b', 'b', 'defaultNameB']]) {
    if (isDefaultName($(inputId).value.trim())) $(inputId).value = t(defaultKey);
    if (isDefaultName(state.reviewers[reviewerKey].name)) state.reviewers[reviewerKey].name = t(defaultKey);
  }
  if (!presetTouched && !state.photos.length) choosePreset(presetOrder()[0]);
  renderPresetOptions();
  render();
  renderUpdateBar();
}
$('#language-select').onchange = (event) => {
  setLanguage(event.target.value);
  applyLanguage();
  saveProjectSoon();
};

// 설치형 웹앱: 한 번 열고 나면 인터넷 없이도 동작하게 앱 파일을 기기에 보관한다
// 서비스 워커는 앱 파일을 서버에서 먼저 받으므로, 인터넷이 되면 열 때마다 최신 코드가 뜬다.
// 그래도 이미 열려 있던 화면은 예전 코드로 돈다. 새 서비스 워커가 자리를 잡으면(controllerchange)
// 그 버전과 이 화면의 버전을 비교해서, 다르면
//   사진을 아직 안 불러왔으면: 잃을 게 없으니 바로 새로고침
//   사진을 고르는 중이면: 알림 줄을 띄우고 사용자가 고르게 한다 (닫을 수 있다)
let updateBarShown = false;
function askWorkerVersion(worker) {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => resolve(null), 2000); // 답이 없는 예전 서비스 워커
    channel.port1.onmessage = (event) => { clearTimeout(timer); resolve(event.data?.version ?? null); };
    worker.postMessage({ type: 'get-version' }, [channel.port2]);
  });
}
function updateBarMessage() {
  if (!state.photos.length) return t('updateReady');
  // 폴더 핸들로 연 사진은 '이어서 고르기'로 바로 돌아오고, 파일 선택으로 연 사진은 다시 골라야 한다
  return state.photos[0].handle ? t('updateReadyResume') : t('updateReadyRepick');
}
function renderUpdateBar() {
  if (!updateBarShown) return;
  const bar = $('#update-bar');
  bar.innerHTML = `<span class="spacer">${updateBarMessage()}</span>`;
  const reloadButton = document.createElement('button');
  reloadButton.className = 'button primary';
  reloadButton.textContent = t('updateReload');
  reloadButton.onclick = async () => {
    reloadButton.disabled = true;
    await saveProjectNow(); // 고른 결과를 먼저 저장하고 새로고침
    location.reload();
  };
  // 닫기: 쓰는 중에 거슬리지 않게 바로 치운다. 다음에 열 때는 어차피 새 버전으로 뜬다
  const closeButton = document.createElement('button');
  closeButton.className = 'button quiet close';
  closeButton.setAttribute('aria-label', t('updateClose'));
  closeButton.title = t('updateClose');
  closeButton.innerHTML = '<i class="ph ph-x" aria-hidden="true"></i>';
  closeButton.onclick = () => { updateBarShown = false; bar.classList.add('hidden'); };
  bar.append(reloadButton, closeButton);
  bar.classList.remove('hidden');
}
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  // 처음 방문해서 서비스 워커가 막 생긴 경우는 새 버전이 아니니 알리지 않는다
  const hadController = Boolean(navigator.serviceWorker.controller);
  navigator.serviceWorker.addEventListener('controllerchange', async () => {
    const newWorker = navigator.serviceWorker.controller;
    if (!hadController || updateBarShown || !newWorker) return;
    // 이 화면이 이미 서버에서 받은 최신 코드면 새로고침할 필요가 없다
    if (await askWorkerVersion(newWorker) === APP_VERSION) return;
    if (!state.photos.length && state.view === 'setup') { location.reload(); return; }
    updateBarShown = true;
    renderUpdateBar();
  });
  navigator.serviceWorker.register('./sw.js').then((registration) => {
    // 휴대폰은 탭을 며칠씩 열어 두기도 하니, 화면으로 돌아올 때마다 새 버전을 확인한다
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') registration.update().catch(() => {});
    });
  }).catch((error) => console.warn('service worker', error));
}
// 진단용: ?debug=1 이면 콘솔에서 상태를 볼 수 있게 한다 (사진 데이터는 이 기기 안에만 있다)
if (new URLSearchParams(location.search).get('debug') === '1') {
  window.momentSelectDebug = { APP_VERSION, state, faceEngine, aestheticEngine, eyeEngine, hashDistance, correlation, looksSimilar, similarKeptPhotos, scoreAccuracy, trainingPairs, learningEvidence, effectiveCastKeys, firstPassQueue, distillQueue, isLastKeptOfScene, activeModel, diagnosticsText, get probeUnitKey() { return probeUnitKey; } };
}
applyLanguage();
