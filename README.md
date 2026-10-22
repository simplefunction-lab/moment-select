# Moment Select

Swipe through hundreds of photos and keep the ones you want. Built for the day the studio sends you 1,500 wedding shots and asks you to pick 30.

**Use it:** https://simplefunction-lab.github.io/moment-select/

[한국어 안내는 아래에 있어요.](#한국어)

## Your photos stay on your device

- Photos are read and analyzed in the browser. Nothing is uploaded.
- Face detection, fonts and icons ship with the app. A Content Security Policy blocks every request to another origin, so the page cannot send anything out even by mistake.
- After the first visit the app works offline.

## How it works

1. **Choose photos.** On a computer, pick a folder (subfolders included). Google Drive works through Google Drive for desktop, which shows your Drive as a folder. On a phone, pick photos from your library.
2. **First pass.** Swipe right to keep, left to drop. The best-scoring shots come first. Near-identical burst shots are grouped so you can pick one from the set.
3. **Narrow down.** Set how many photos you want. Kept photos come back lowest score first until you reach that number. A side-by-side mode compares the weakest photo with its closest look-alike.
4. **Second look.** Dropped photos come back highest score first. Anything you rescue is stamped with the time it came back.
5. **Export.** Copy the originals into a new folder with a file list. On phones, save through the share sheet.

Two people can pick separately and merge: kept by both, or kept by either.

### Scoring

Each photo gets eight values between 0 and 1: subject sharpness, sharpness of the in-focus area, exposure, absence of frowns, smiles, faces not cut off at the edge, low noise, and open eyes. Sharpness and noise are ranked against the other photos in the same set.

Starting weights depend on the photo type (wedding, travel, events, family, other). After about 20 decisions, a logistic regression refits the weights to the person picking, and the remaining photos are reordered. Closed-eye shots score lower by default. Turn this off if your photographer will retouch the eyes, as wedding studios often do; the shots are then only flagged.

## Run locally

The app is plain HTML, CSS and JavaScript with no build step. It has to be served over HTTP, because module scripts and the service worker do not run from `file://`.

```sh
npx serve .
```

## Files

| Path | What it is |
|---|---|
| `index.html` | Page markup and Content Security Policy |
| `app.js` | Reading files, analysis, scoring, learning, swiping, export |
| `i18n.js` | Korean and English text |
| `styles.css` | Layout and theme |
| `sw.js` | Service worker that caches the app for offline use |
| `vendor/` | face-api models, exifr, subset fonts and icons ([licenses](vendor/LICENSES.txt)) |

---

## 한국어

수백 장의 사진을 넘기며 남길 사진을 고르는 웹앱이에요. 스튜디오에서 웨딩 원본 1,500장을 받고 30장을 골라야 할 때를 생각하고 만들었어요.

**바로 쓰기:** https://simplefunction-lab.github.io/moment-select/

### 사진은 기기 밖으로 나가지 않아요

- 사진은 브라우저 안에서 읽고 분석해요. 어디에도 올리지 않아요.
- 얼굴 분석 모델, 글꼴, 아이콘을 앱과 함께 배포해요. 콘텐츠 보안 정책(CSP)이 다른 출처로 가는 요청을 모두 막아서, 실수로라도 밖으로 보낼 수 없어요.
- 한 번 열고 나면 인터넷 없이도 동작해요.

### 사용 흐름

1. **사진 고르기:** 컴퓨터에서는 폴더를 골라요(하위 폴더 포함). 구글 드라이브는 Google Drive 데스크톱을 설치하면 폴더로 보여요. 휴대폰에서는 사진 보관함에서 골라요.
2. **1차 셀렉:** 오른쪽으로 넘기면 남기고, 왼쪽이면 빼요. 점수 높은 사진부터 나와요. 거의 같은 연사 컷은 한 화면에 모아서 그중 남길 것만 골라요.
3. **줄이기:** 남길 장수를 정하면, 남긴 사진을 점수 낮은 순으로 다시 보여줘요. 가장 약한 사진과 가장 닮은 사진을 나란히 비교할 수도 있어요.
4. **다시 보기:** 뺀 사진을 점수 높은 순으로 다시 봐요. 다시 남긴 사진에는 그 시각이 기록돼요.
5. **내보내기:** 원본을 새 폴더로 복사하고 파일 목록을 남겨요. 휴대폰에서는 공유 창으로 저장해요.

두 사람이 따로 고른 뒤 '둘 다 남긴 사진'이나 '한 명이라도 남긴 사진'으로 합칠 수 있어요.

### 점수

사진마다 0에서 1 사이 값 8개를 매겨요. 인물 선명도, 초점 부분 선명도, 노출, 찡그림 없음, 웃음, 얼굴 안 잘림, 노이즈 적음, 눈 뜸이에요. 선명도와 노이즈는 같은 폴더 사진들 사이의 순위로 계산해요.

처음 가중치는 사진 종류(웨딩과 스냅, 여행, 행사와 모임, 아이와 가족, 기타)마다 달라요. 결정이 20건쯤 쌓이면 고르는 사람의 결정으로 로지스틱 회귀를 다시 맞추고 남은 사진 순서를 바꿔요. 눈 감은 사진은 기본으로 점수를 깎아요. 웨딩 촬영처럼 작가가 눈을 합성해 준다면 설정에서 끄면 되고, 그때는 표시만 해요.

### 내 컴퓨터에서 실행

빌드 단계 없는 HTML, CSS, JavaScript예요. 모듈 스크립트와 서비스 워커가 `file://`에서는 동작하지 않아서 HTTP로 열어야 해요.

```sh
npx serve .
```
