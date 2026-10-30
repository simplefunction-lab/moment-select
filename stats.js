// 익명 합계 보내기 (v47): "몇 명이 어느 단계까지 갔나"를 하루 합계로만 센다 (서버: simplefunction-stats, 저장소 dlehddnjs/simplefunction-stats)
// 보내는 것: 사건 이름과 정해 둔 꼬리표(화면 언어, 기기 종류, 앱으로 열었는지, 들어온 곳, 장수·시간 구간)뿐이다
// 보내지 않는 것: 사진, 파일 이름, 점수, 얼굴, 기기 번호, 쿠키. 이 기기에 아무것도 저장하지 않는다(탭이 닫히면 잊는다)
// 탭마다 사건 하나는 한 번만 보낸다. 퍼널 단계(STEPS)는 앞 단계를 보낸 탭만 보낸다(닫힌 퍼널)
// 브라우저의 추적 거부 신호(GPC, Do Not Track)가 켜져 있으면 아무것도 보내지 않는다
// 실제 사이트가 아니면(로컬 시험) 보내지 않고 window.momentSelectStatsLog에 쌓기만 한다 (출시 게이트가 이것을 검사한다)
export const STATS_ENDPOINT = 'https://simplefunction-stats.vercel.app/api/e';
const LIVE_HOST = 'simplefunction-lab.github.io';
export const STEPS = ['visit', 'pick_open', 'loaded', 'analyzed', 'decided10', 'first_done', 'export_view', 'export_done'];
export const EVENTS = [...STEPS, 'trial_start', 'trial_done', 'receive_fail', 'target_hit', 'together', 'resume', 'feedback'];

const optedOut = navigator.globalPrivacyControl === true || navigator.doNotTrack === '1' || window.doNotTrack === '1';
const live = location.hostname === LIVE_HOST;
const sent = new Set();
let getLanguage = () => null;
let tags = null;

// 들어온 곳: 우리가 올린 링크의 ?ref=·?c=, 없으면 앞 사이트 종류만 (주소·검색어는 보내지 않는다)
const REFERRERS = [
  [/(^|\.)instagram\.com$/, 'instagram'], [/(^|\.)threads\.(net|com)$/, 'threads'], [/(^|\.)reddit\.com$/, 'reddit'],
  [/(^|\.)disquiet\.io$/, 'disquiet'], [/(^|\.)producthunt\.com$/, 'producthunt'], [/(^|\.)(x\.com|twitter\.com|t\.co)$/, 'x'],
  [/(^|\.)kakao(talk)?\.(com|co\.kr)$/, 'kakao'], [/(^|\.)google\./, 'google'], [/(^|\.)bing\.com$/, 'bing'], [/(^|\.)naver\.com$/, 'naver'],
  [/(^|\.)duckduckgo\.com$/, 'duckduckgo'], [/(^|\.)(chatgpt\.com|openai\.com)$/, 'chatgpt'], [/(^|\.)perplexity\.ai$/, 'perplexity'],
];
function sourceOf() {
  const params = new URLSearchParams(location.search);
  const ref = (params.get('ref') || '').toLowerCase();
  const content = params.get('c') || null;
  if (navigator.webdriver) return { ref: 'bot', c: null };
  if (ref) return { ref, c: content };
  let host = '';
  try { host = document.referrer ? new URL(document.referrer).hostname : ''; } catch { host = ''; }
  if (!host) return { ref: 'none', c: content };
  if (host === LIVE_HOST) return { ref: 'root', c: content }; // 루트 사이트(프로필 링크)에서 들어옴
  return { ref: REFERRERS.find(([pattern]) => pattern.test(host))?.[1] ?? 'other', c: content };
}
function deviceOf() {
  const touch = matchMedia('(pointer: coarse)').matches || navigator.maxTouchPoints > 0;
  if (!touch) return 'pc';
  return Math.min(screen.width, screen.height) >= 600 ? 'tablet' : 'phone';
}
const modeOf = () => (matchMedia('(display-mode: standalone)').matches || navigator.standalone === true ? 'app' : 'browser');

// 정확한 숫자 대신 구간만 보낸다
export const photoBucket = (count) => (count < 50 ? '1-49' : count < 200 ? '50-199' : count < 500 ? '200-499' : count < 1000 ? '500-999' : '1000+');
export const keptBucket = (count) => (count <= 10 ? '1-10' : count <= 30 ? '11-30' : count <= 100 ? '31-100' : '101+');
export const timeBucket = (ms) => (ms < 10_000 ? '<10s' : ms < 30_000 ? '10-30s' : ms < 60_000 ? '30-60s' : ms < 180_000 ? '1-3m' : ms < 600_000 ? '3-10m' : '10m+');

export function initStats(languageGetter) {
  getLanguage = languageGetter;
  tags = { ...sourceOf(), dev: deviceOf(), mode: modeOf() };
}

// extra: { n, k, t, how } 중 필요한 것 (구간 값만)
export function track(event, extra = {}) {
  if (optedOut || !tags || !EVENTS.includes(event) || sent.has(event)) return;
  const step = STEPS.indexOf(event);
  if (step > 0 && !sent.has(STEPS[step - 1])) return;
  sent.add(event);
  const payload = { p: 'ms', e: event, lang: getLanguage(), ...tags, ...extra };
  if (!live) { (window.momentSelectStatsLog ??= []).push(payload); return; }
  const body = new Blob([JSON.stringify(payload)], { type: 'text/plain' });
  try { if (!navigator.sendBeacon?.(STATS_ENDPOINT, body)) fetch(STATS_ENDPOINT, { method: 'POST', body, keepalive: true, mode: 'cors' }).catch(() => {}); } catch { /* 보내기 실패는 앱에 영향을 주지 않는다 */ }
}
export const hasTracked = (event) => sent.has(event);
