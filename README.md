# Moment Select

Swipe through hundreds of photos and keep the ones you want. Built for the day the studio sends you 1,500 wedding shots and asks you to pick 30.

**Use it:** https://simplefunction-lab.github.io/moment-select/

[한국어 안내는 아래에 있어요.](#한국어)

## Your photos stay on your device

- Photos are read and analyzed in the browser. Nothing is uploaded.
- Face detection, face recognition, the aesthetic model, fonts and icons ship with the app. A Content Security Policy blocks every request to another origin, so the page cannot send anything out even by mistake.
- The models (about 15 MB) download once, the first time you analyze photos, with a note saying what each one is for. After that the app works offline.

## How it works

1. **Choose photos.** On a computer, pick a folder (subfolders included). Google Drive works through Google Drive for desktop, which shows your Drive as a folder. On a phone, pick photos from your library.
2. **First pass.** Swipe right to keep, left to drop. The best-scoring shots come first. Near-identical burst shots are grouped so you can pick one from the set.
3. **Narrow down.** Set how many photos you want. Kept photos come back lowest score first until you reach that number. A side-by-side mode compares the weakest photo with its closest look-alike.
4. **Second look.** Dropped photos come back highest score first. Anything you rescue is stamped with the time it came back.
5. **Export.** Copy the originals into a new folder with a file list. On phones, save through the share sheet.

Two people can pick separately and merge: kept by both, or kept by either.

### Scoring

Each photo gets ten values between 0 and 1: subject sharpness, sharpness of the in-focus area, exposure, absence of frowns, smiles, faces not cut off at the edge, low noise, open eyes, an aesthetic score, and whether the main people are in it. Sharpness, noise and the aesthetic score are ranked against the other photos in the same set.

- **Aesthetic score:** NIMA (MobileNet), trained on about 250,000 rated photos from the AVA dataset ([idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment), Apache 2.0). It runs with TensorFlow.js and is skipped on devices without graphics acceleration, where it would take one to two seconds per photo.
- **Main people:** faces are grouped by person with face-api's face recognition model. The people who appear most often are preselected (two for weddings), and you can change them under Main people.

Starting weights depend on the photo type (wedding, travel, events, family, other). The model then learns from the person picking:

- Keep and drop decisions train a logistic regression.
- Choosing one shot out of a similar set adds "this one beats that one" pairs (Bradley-Terry). Scene and people are the same in such pairs, so they isolate quality and taste.
- Early in the first pass, every fifth photo is one the model is unsure about, marked as a taste check.

Closed-eye shots score lower by default. Turn this off if your photographer will retouch the eyes, as wedding studios often do; the shots are then only flagged.

When narrowing down, photos are split into scenes by capture time. The last kept photo of a scene comes last, so a whole scene does not disappear by accident.

The Scoring panel shows how well the score matches your decisions, measured with the score each photo had when you decided. 50% is chance.

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
| `aesthetic.js` | Aesthetic score model (NIMA) run with TensorFlow.js |
| `i18n.js` | Korean and English text |
| `styles.css` | Layout and theme |
| `sw.js` | Service worker that caches the app for offline use |
| `vendor/` | face-api and its models, the NIMA model, exifr, subset fonts and icons ([licenses](vendor/LICENSES.txt)) |

---

## 한국어

수백 장의 사진을 넘기며 남길 사진을 고르는 웹앱이에요. 스튜디오에서 웨딩 원본 1,500장을 받고 30장을 골라야 할 때를 생각하고 만들었어요.

**바로 쓰기:** https://simplefunction-lab.github.io/moment-select/

### 사진은 기기 밖으로 나가지 않아요

- 사진은 브라우저 안에서 읽고 분석해요. 어디에도 올리지 않아요.
- 얼굴 분석, 같은 사람 알아보기, 미적 점수 모델과 글꼴, 아이콘을 앱과 함께 배포해요. 콘텐츠 보안 정책(CSP)이 다른 출처로 가는 요청을 모두 막아서, 실수로라도 밖으로 보낼 수 없어요.
- 모델(약 15MB)은 처음 분석할 때 한 번만 받아요. 받는 동안 어떤 용도의 모델인지 보여줘요. 그다음부터는 인터넷 없이도 동작해요.

### 사용 흐름

1. **사진 고르기:** 컴퓨터에서는 폴더를 골라요(하위 폴더 포함). 구글 드라이브는 Google Drive 데스크톱을 설치하면 폴더로 보여요. 휴대폰에서는 사진 보관함에서 골라요.
2. **1차 셀렉:** 오른쪽으로 넘기면 남기고, 왼쪽이면 빼요. 점수 높은 사진부터 나와요. 거의 같은 연사 컷은 한 화면에 모아서 그중 남길 것만 골라요.
3. **줄이기:** 남길 장수를 정하면, 남긴 사진을 점수 낮은 순으로 다시 보여줘요. 가장 약한 사진과 가장 닮은 사진을 나란히 비교할 수도 있어요.
4. **다시 보기:** 뺀 사진을 점수 높은 순으로 다시 봐요. 다시 남긴 사진에는 그 시각이 기록돼요.
5. **내보내기:** 원본을 새 폴더로 복사하고 파일 목록을 남겨요. 휴대폰에서는 공유 창으로 저장해요.

두 사람이 따로 고른 뒤 '둘 다 남긴 사진'이나 '한 명이라도 남긴 사진'으로 합칠 수 있어요.

### 점수

사진마다 0에서 1 사이 값 10개를 매겨요. 인물 선명도, 초점 부분 선명도, 노출, 찡그림 없음, 웃음, 얼굴 안 잘림, 노이즈 적음, 눈 뜸, 보기 좋음, 주인공이에요. 선명도, 노이즈, 보기 좋음은 같은 폴더 사진들 사이의 순위로 계산해요.

- **보기 좋음:** AVA 데이터셋의 사진 평점 약 25만 장으로 학습된 NIMA(MobileNet) 모델이에요([idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment), Apache 2.0). TensorFlow.js로 돌리고, 그래픽 가속이 없는 기기에서는 장당 1~2초가 걸려서 건너뛰어요.
- **주인공:** face-api의 얼굴 인식 모델로 같은 사람끼리 묶어요. 가장 자주 나온 사람을 미리 골라두고(웨딩은 2명), '주인공'에서 바꿀 수 있어요.

처음 가중치는 사진 종류(웨딩과 스냅, 여행, 행사와 모임, 아이와 가족, 기타)마다 달라요. 그다음부터는 고르는 사람에게서 배워요.

- 남김과 뺌 결정으로 로지스틱 회귀를 맞춰요.
- 비슷한 컷 중 하나를 고르면 "이 사진이 저 사진보다 낫다"는 쌍으로도 배워요(Bradley-Terry). 장면과 인물이 같아서 품질과 취향 차이만 남아요.
- 1차 셀렉 초반에는 다섯 번째마다 모델이 헷갈리는 사진을 '취향 확인용'으로 끼워 보여줘요.

눈 감은 사진은 기본으로 점수를 깎아요. 웨딩 촬영처럼 작가가 눈을 합성해 준다면 설정에서 끄면 되고, 그때는 표시만 해요.

줄이기 단계에서는 촬영 시각으로 장면을 나눠요. 어떤 장면의 마지막 남은 사진은 맨 뒤로 미뤄서, 장면 하나가 통째로 빠지지 않게 해요.

점수 기준 패널에서 점수가 내 결정을 얼마나 맞혔는지 볼 수 있어요. 결정할 때 보이던 점수로 계산하고, 50%면 무작위 수준이에요.

### 내 컴퓨터에서 실행

빌드 단계 없는 HTML, CSS, JavaScript예요. 모듈 스크립트와 서비스 워커가 `file://`에서는 동작하지 않아서 HTTP로 열어야 해요.

```sh
npx serve .
```
