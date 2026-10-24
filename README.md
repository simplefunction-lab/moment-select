# Moment Select

Swipe through hundreds of photos and keep the ones you want. Built for the day the studio sends you 1,500 wedding shots and asks you to pick 30.

**Use it:** https://simplefunction-lab.github.io/moment-select/

[한국어 안내는 아래에 있어요.](#한국어)

## Your photos stay on your device

- Photos are read and analyzed in the browser. Nothing is uploaded.
- Face detection, face recognition, the aesthetic model, fonts and icons ship with the app. A Content Security Policy blocks every request to another origin, so the page cannot send anything out even by mistake.
- The MediaPipe library normally sends usage statistics (task type and timing, no photos) to Google. The app switches this off when it loads MediaPipe, and the same policy would block it anyway, so nothing leaves the device.
- The models (about 15 MB, plus about 18 MB for the closed-eye model when closed-eye scoring is on) download once, the first time you analyze photos, with a note saying what each one is for. After that the app works offline.

## How it works

1. **Choose photos.** On a computer, pick a folder (subfolders included). Google Drive works through Google Drive for desktop, which shows your Drive as a folder. On a phone, pick photos from your library.
2. **First pass.** Swipe right to keep, left to drop. The best-scoring shots come first. Near-identical burst shots are grouped so you can pick one from the set.
3. **Narrow down.** Set how many photos you want. Kept photos come back lowest score first until you reach that number. A side-by-side mode compares the weakest photo with its closest look-alike.
4. **Second look.** Dropped photos come back highest score first. Anything you rescue is stamped with the time it came back.
5. **Export.** On a computer (Chrome or Edge, folder opened), sort the folder in place: your picks move into a new folder you name (default `moment_YYMMDD`), and everything else moves into its `others` folder. Nothing is copied, RAW and sidecar files with the same name move along, subfolders left empty are removed, a `moment-select-moves.txt` record is written, and you can undo (removed subfolders come back). You can also just copy the picks into a new folder. Phones cannot move or delete library photos from a web page, so there you save the picks through the share sheet.

Two people can pick separately and merge: kept by both, or kept by either. Photos you disagree on can be decided together, one at a time.

### Scoring

Each photo gets ten values between 0 and 1: subject sharpness, sharpness of the in-focus area, exposure, absence of frowns, smiles, faces not cut off at the edge, low noise, open eyes, an aesthetic score, and whether the main people are in it. Sharpness, noise and the aesthetic score are ranked against the other photos in the same set.

- **Aesthetic score:** NIMA (MobileNet), trained on about 250,000 rated photos from the AVA dataset ([idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment), Apache 2.0). It runs with TensorFlow.js and is skipped on devices without graphics acceleration, where it would take one to two seconds per photo.
- **Main people:** faces are grouped by person with face-api's face recognition model. The people who appear most often are preselected (two for weddings), and you can change them under Main people.
- **Closed eyes and smiles:** when closed-eye scoring is on, each large face (up to six per photo) also goes through MediaPipe Face Landmarker (Apache 2.0). A face counts as eyes closed when both `eyeBlinkLeft` and `eyeBlinkRight` are 0.5 or higher, and the smile value averages face-api's happy score with MediaPipe's mouth-smile scores. If MediaPipe finds no face, or the setting is off, eyes are judged from the shape of face-api's 68 landmarks (eye aspect ratio). Analysis details under Scoring show which method judged how many faces.

Starting weights depend on the photo type (wedding, travel, events, family, other). The model then learns from the person picking:

- Keep and drop decisions train a logistic regression.
- Choosing one shot out of a similar set adds "this one beats that one" pairs (Bradley-Terry). Scene and people are the same in such pairs, so they isolate quality and taste.
- Early in the first pass, every fifth photo is one the model is unsure about, marked as a taste check.

Closed-eye shots score lower by default. Turn this off if your photographer will retouch the eyes, as wedding studios often do; the shots are then only flagged.

When narrowing down, photos are split into scenes by capture time. The last kept photo of a scene comes last, so a whole scene does not disappear by accident.

The Scoring panel shows how well the score matches your decisions, measured with the score each photo had when you decided. 50% is chance.

## Support and the iPhone app

Moment Select is free, with no ads and no tracking. If it saved you an evening, you can [buy me a coffee](https://buymeacoffee.com/simplefunction).

An iPhone app is next. It will sort your picks into an album in your photo library, with no duplicate copies. [Get notified](https://tally.so/r/vXG9N4?source=readme) when it's out.

The same two links sit at the bottom of the Export screen. They open in a new tab and send no referrer; the page itself still makes no requests to other sites.

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
| `vendor/` | face-api and its models, the NIMA model, MediaPipe Face Landmarker, exifr, subset fonts and icons ([licenses](vendor/LICENSES.txt)) |

---

## 한국어

수백 장의 사진을 넘기며 남길 사진을 고르는 웹앱이에요. 스튜디오에서 웨딩 원본 1,500장을 받고 30장을 골라야 할 때를 생각하고 만들었어요.

**바로 쓰기:** https://simplefunction-lab.github.io/moment-select/

### 사진은 기기 밖으로 나가지 않아요

- 사진은 브라우저 안에서 읽고 분석해요. 어디에도 올리지 않아요.
- 얼굴 분석, 같은 사람 알아보기, 미적 점수 모델과 글꼴, 아이콘을 앱과 함께 배포해요. 콘텐츠 보안 정책(CSP)이 다른 출처로 가는 요청을 모두 막아서, 실수로라도 밖으로 보낼 수 없어요.
- MediaPipe 라이브러리는 원래 사용 통계(작업 종류와 처리 시간, 사진은 아님)를 구글로 보내요. 앱이 MediaPipe를 불러올 때 이 기능을 끄고, 혹시 남아 있어도 같은 정책이 막아서 기기 밖으로 나가는 것은 없어요.
- 모델(약 15MB, 눈 감은 사진 감점을 켜면 눈 감음 모델 약 18MB 추가)은 처음 분석할 때 한 번만 받아요. 받는 동안 어떤 용도의 모델인지 보여줘요. 그다음부터는 인터넷 없이도 동작해요.

### 사용 흐름

1. **사진 고르기:** 컴퓨터에서는 폴더를 골라요(하위 폴더 포함). 구글 드라이브는 Google Drive 데스크톱을 설치하면 폴더로 보여요. 휴대폰에서는 사진 보관함에서 골라요.
2. **1차 셀렉:** 오른쪽으로 넘기면 남기고, 왼쪽이면 빼요. 점수 높은 사진부터 나와요. 거의 같은 연사 컷은 한 화면에 모아서 그중 남길 것만 골라요.
3. **줄이기:** 남길 장수를 정하면, 남긴 사진을 점수 낮은 순으로 다시 보여줘요. 가장 약한 사진과 가장 닮은 사진을 나란히 비교할 수도 있어요.
4. **다시 보기:** 뺀 사진을 점수 높은 순으로 다시 봐요. 다시 남긴 사진에는 그 시각이 기록돼요.
5. **내보내기:** 컴퓨터(크롬, 엣지에서 폴더로 열었을 때)에서는 폴더를 그 자리에서 정리해요. 고른 사진은 직접 이름 붙인 새 폴더(기본 `moment_YYMMDD`)로, 나머지는 그 안의 `others` 폴더로 옮겨요. 복사하지 않고, 이름이 같은 RAW나 보정 파일도 함께 옮기며, 옮기고 나서 빈 하위 폴더는 지우고, `moment-select-moves.txt`에 기록을 남겨요. 되돌리면 지운 하위 폴더도 다시 생겨요. 고른 사진만 새 폴더로 복사할 수도 있어요. 휴대폰은 웹페이지가 사진 보관함의 사진을 옮기거나 지울 수 없어서 공유 창으로 고른 사진을 저장해요.

두 사람이 따로 고른 뒤 '둘 다 남긴 사진'이나 '한 명이라도 남긴 사진'으로 합칠 수 있어요. 의견이 갈린 사진은 한 장씩 같이 정할 수 있어요.

### 점수

사진마다 0에서 1 사이 값 10개를 매겨요. 인물 선명도, 초점 부분 선명도, 노출, 찡그림 없음, 웃음, 얼굴 안 잘림, 노이즈 적음, 눈 뜸, 보기 좋음, 주인공이에요. 선명도, 노이즈, 보기 좋음은 같은 폴더 사진들 사이의 순위로 계산해요.

- **보기 좋음:** AVA 데이터셋의 사진 평점 약 25만 장으로 학습된 NIMA(MobileNet) 모델이에요([idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment), Apache 2.0). TensorFlow.js로 돌리고, 그래픽 가속이 없는 기기에서는 장당 1~2초가 걸려서 건너뛰어요.
- **주인공:** face-api의 얼굴 인식 모델로 같은 사람끼리 묶어요. 가장 자주 나온 사람을 미리 골라두고(웨딩은 2명), '주인공'에서 바꿀 수 있어요.
- **눈 감음과 웃음:** 눈 감은 사진 감점을 켜면 큰 얼굴(사진마다 최대 6개)을 MediaPipe Face Landmarker(Apache 2.0)로도 봐요. `eyeBlinkLeft`와 `eyeBlinkRight`가 둘 다 0.5 이상이면 눈 감음이고, 웃음은 face-api의 웃는 표정 값과 MediaPipe의 입꼬리 값을 평균해요. MediaPipe가 얼굴을 못 찾거나 설정이 꺼져 있으면 face-api 랜드마크 68개로 눈 모양(EAR)을 봐요. 점수 기준 아래 분석 정보에서 어느 방식으로 몇 개 얼굴을 판정했는지 볼 수 있어요.

처음 가중치는 사진 종류(웨딩과 스냅, 여행, 행사와 모임, 아이와 가족, 기타)마다 달라요. 그다음부터는 고르는 사람에게서 배워요.

- 남김과 뺌 결정으로 로지스틱 회귀를 맞춰요.
- 비슷한 컷 중 하나를 고르면 "이 사진이 저 사진보다 낫다"는 쌍으로도 배워요(Bradley-Terry). 장면과 인물이 같아서 품질과 취향 차이만 남아요.
- 1차 셀렉 초반에는 다섯 번째마다 모델이 헷갈리는 사진을 '취향 확인용'으로 끼워 보여줘요.

눈 감은 사진은 기본으로 점수를 깎아요. 웨딩 촬영처럼 작가가 눈을 합성해 준다면 설정에서 끄면 되고, 그때는 표시만 해요.

줄이기 단계에서는 촬영 시각으로 장면을 나눠요. 어떤 장면의 마지막 남은 사진은 맨 뒤로 미뤄서, 장면 하나가 통째로 빠지지 않게 해요.

점수 기준 패널에서 점수가 내 결정을 얼마나 맞혔는지 볼 수 있어요. 결정할 때 보이던 점수로 계산하고, 50%면 무작위 수준이에요.

### 응원과 아이폰 앱

모먼트 셀렉은 무료이고 광고도 추적도 없어요. 도움이 됐다면 [커피 한 잔으로 응원](https://buymeacoffee.com/simplefunction)해 주세요.

다음은 아이폰 앱이에요. 고른 사진을 복사본 없이 사진 보관함의 앨범으로 정리해요. 나오면 [알림 받기](https://tally.so/r/vXG9N4?source=readme).

내보내기 화면 맨 아래에도 같은 두 링크가 있어요. 새 탭으로 열리고 어디서 왔는지(referrer)도 보내지 않아요. 페이지 자체는 여전히 다른 사이트로 아무 요청도 보내지 않아요.

### 내 컴퓨터에서 실행

빌드 단계 없는 HTML, CSS, JavaScript예요. 모듈 스크립트와 서비스 워커가 `file://`에서는 동작하지 않아서 HTTP로 열어야 해요.

```sh
npx serve .
```
