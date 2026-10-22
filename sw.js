// 모먼트 셀렉 서비스 워커
// 앱 파일만 기기에 보관해서 오프라인으로 열리게 한다. 사진은 다루지 않는다.
const CACHE_NAME = 'moment-select-v7';
// 앱을 이루는 파일 전체 (라이브러리, 모델, 글꼴 포함)
const APP_FILES = [
  './',
  './app.js',
  './i18n.js',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './index.html',
  './manifest.webmanifest',
  './styles.css',
  './vendor/exifr/lite.esm.mjs',
  './vendor/face-api/face-api.esm.js',
  './vendor/face-api/model/face_expression_model-weights_manifest.json',
  './vendor/face-api/model/face_expression_model.bin',
  './vendor/face-api/model/face_landmark_68_model-weights_manifest.json',
  './vendor/face-api/model/face_landmark_68_model.bin',
  './vendor/face-api/model/tiny_face_detector_model-weights_manifest.json',
  './vendor/face-api/model/tiny_face_detector_model.bin',
  './vendor/fonts/nanum-pen-marks-latin.woff2',
  './vendor/fonts/nanum-pen-marks.woff2',
  './vendor/fonts/pretendard-subset.woff2',
  './vendor/icons/phosphor-subset.woff2',
  './vendor/icons/phosphor.css',
];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  // 이전 버전 캐시 정리
  event.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;
  // 보관한 파일이 있으면 그것을, 없으면 네트워크에서 받는다
  event.respondWith(caches.match(request, { ignoreSearch: true }).then((cached) => cached ?? fetch(request)));
});
