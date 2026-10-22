// 모먼트 셀렉 서비스 워커
// 앱 파일을 기기에 보관해서 오프라인으로도 열리게 한다. 사진은 다루지 않는다.
//
// 받는 순서
//   앱 코드(HTML, app.js, i18n.js, styles.css, manifest): 서버에서 먼저 받고, 안 되면 보관본
//     인터넷이 되면 열 때마다 최신 버전이 뜬다. 신호가 약해 NETWORK_TIMEOUT_MS 안에 답이 없으면 보관본을 쓴다.
//   vendor/(얼굴 모델, 글꼴, 아이콘)와 앱 아이콘: 보관본 먼저. 크고 거의 바뀌지 않는다.

const APP_VERSION = 'v9'; // 배포할 때마다 app.js의 APP_VERSION과 같이 올린다
const CACHE_NAME = `moment-select-${APP_VERSION}`;
const NETWORK_TIMEOUT_MS = 4000;
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
  // cache: 'reload' 로 브라우저 HTTP 캐시를 건너뛴다. GitHub Pages는 파일을 10분 캐시해서,
  // 배포 직후 설치되면 새 서비스 워커가 예전 app.js를 담아 버릴 수 있다.
  const freshRequests = APP_FILES.map((path) => new Request(path, { cache: 'reload' }));
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(freshRequests)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  // 이전 버전 캐시 정리
  event.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
    .then(() => self.clients.claim()));
});

// 화면이 "지금 서비스 워커 버전이 뭐냐"고 물으면 답한다 (새 버전 알림 판단용)
self.addEventListener('message', (event) => {
  if (event.data?.type === 'get-version') event.ports[0]?.postMessage({ version: APP_VERSION });
});

const isAppCode = (url) => !url.pathname.includes('/vendor/') && !url.pathname.includes('/icons/')
  && (url.pathname.endsWith('/') || /\.(html|js|css|webmanifest)$/.test(url.pathname));

// 서버 먼저, 실패하거나 너무 느리면 보관본
async function networkFirst(event) {
  const cacheKey = new URL(event.request.url);
  cacheKey.search = ''; // ?debug=1 같은 꼬리는 떼고 보관
  const cache = await caches.open(CACHE_NAME);
  // no-cache: HTTP 캐시가 있어도 서버에 바뀌었는지 묻는다 (안 바뀌었으면 304라 가볍다)
  const fromNetwork = fetch(event.request, { cache: 'no-cache' }).then((response) => {
    if (response.ok) event.waitUntil(cache.put(cacheKey.href, response.clone()));
    return response;
  });
  const cached = await cache.match(cacheKey.href);
  if (!cached) return fromNetwork;
  const timeout = new Promise((resolve) => setTimeout(() => resolve(cached), NETWORK_TIMEOUT_MS));
  return Promise.race([fromNetwork.catch(() => cached), timeout]);
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (isAppCode(url)) { event.respondWith(networkFirst(event)); return; }
  // 보관한 파일이 있으면 그것을, 없으면 네트워크에서 받는다
  event.respondWith(caches.match(request, { ignoreSearch: true }).then((cached) => cached ?? fetch(request)));
});
