/* =====================================================================
   화면 문구 사전 (한국어, 영어)
   - 문자열은 그대로, 값이 들어가는 문구는 함수로 둔다
   - 정적 문구는 index.html 요소의 data-i18n(글자), data-i18n-aria(aria-label), data-i18n-title(title)로 연결
   ===================================================================== */

const SUPPORTED_LANGUAGES = ['ko', 'en'];
const LANGUAGE_STORAGE_KEY = 'moment-select-language';

// 영어 단수와 복수: photo, photos
const photosEn = (count) => `${count} ${count === 1 ? 'photo' : 'photos'}`;

const MESSAGES = {
  ko: {
    // ---------- 공통, 상단 바 ----------
    brand: '모먼트 셀렉',
    tabSetup: '시작', tabFirst: '1차 셀렉', tabDistill: '줄이기', tabRescue: '다시 보기', tabExport: '내보내기',
    stepsAria: '단계',
    goalKept: '남긴 사진', goalUnit: '장',
    goalOver: (count) => `${count}장 많음`,
    reviewerAria: '고르는 사람', languageAria: '언어',
    defaultNameA: '나', defaultNameB: '파트너',

    // ---------- 시작 화면 ----------
    setupTitle: '남길 사진 고르기',
    setupIntro: '사진을 고르면 이 기기 안에서 분석하고, 한 장씩 넘기며 남길 사진을 고를 수 있어요. 사진은 어디에도 올라가지 않고, 이 앱은 외부 서버와 통신하지 않아요.',
    driveNote: '구글 드라이브 사진은 컴퓨터에 Google Drive 데스크톱을 설치하면 폴더로 보여요. Windows는 탐색기의 Google Drive 드라이브, Mac은 Finder 사이드바의 Google Drive 아래 사진 폴더를 고르세요. 처음 분석할 때 사진을 내려받느라 느릴 수 있어서, 사진 폴더를 미리 오프라인 사용으로 설정해 두면 빨라요.',
    resumeButton: '이어서 고르기',
    resumeName: (name) => `지난번 폴더 '${name}'`,
    fieldPreset: '사진 종류', presetAria: '사진 종류',
    fieldTarget: '남길 장수', targetHint: '장. 고르는 중에도 바꿀 수 있어요.',
    fieldGrouping: '나눠 보기', categoryCheck: '인원수로 나눠 보기',
    categoryHint: '혼자, 둘, 소그룹, 단체로 걸러 볼 수 있어요. 끄면 전체를 한 줄로 봅니다.',
    crowdLabel: '단체로 볼 인원', crowdUnit: '명 이상',
    fieldScore: '점수 순서', scoreCheck: '잘 나온 사진부터 보기',
    scoreHint: '얼굴 선명도, 노출, 표정을 보고 순서를 정해요. 고르는 동안 내 결정을 배워서 순서를 다시 맞춥니다.',
    fieldPeople: '고르는 사람', personAAria: '첫 번째 사람', personBAria: '두 번째 사람',
    peopleHint: '둘이 따로 고르고 나중에 합칠 수 있어요. 혼자 고르면 그대로 두면 됩니다.',
    fieldShare: '통계 보내기', shareCheck: '점수 기준 개선에 익명 숫자 보내기',
    shareHint: '사진, 파일 이름, 얼굴 정보는 보내지 않아요. 사진마다 점수 항목 7개의 값(0에서 1 사이 숫자)과 남김 여부, 사진 종류만 보냅니다.',
    pickFolder: '사진 폴더 선택', pickPhotos: '사진 여러 장 선택',
    pickNoteDefault: '폴더를 고르면 하위 폴더까지 JPG, PNG, WEBP를 읽어요. RAW는 건너뜁니다.',
    pickNoteTouch: '사진 보관함이나 갤러리에서 고를 사진을 모두 선택하세요. 사진은 이 기기 밖으로 나가지 않아요.',
    pickNoteNoFolder: '이 브라우저는 폴더에 바로 저장하지 못해서, 내보낼 때 한 장씩 내려받아요. 크롬이나 엣지에서 열면 새 폴더로 바로 복사돼요.',
    pickNoteReading: '사진 목록을 읽는 중이에요.',
    fallbackFolderName: '선택한 폴더',
    pickedSourceName: ({ count, first }) => `사진 ${count}장 (${first}부터)`,
    noPhotosFound: '읽을 수 있는 사진(JPG, PNG, WEBP)을 찾지 못했어요. 다른 폴더나 사진을 골라주세요.',
    folderOpenError: (message) => `폴더를 열지 못했어요. ${message}`,
    permissionNeeded: '폴더를 읽으려면 권한을 허용해야 해요.',

    // ---------- 사진 종류 ----------
    presets: {
      wedding: { label: '웨딩과 스냅', hint: '차분한 포즈가 많아서 웃음보다 선명도와 찡그림 없음을 크게 봐요.' },
      travel: { label: '여행', hint: '풍경 비중이 커서 사진 전체의 초점과 노출을 크게 봐요.' },
      event: { label: '행사와 모임', hint: '단체 사진이 많아서 얼굴이 다 보이는지와 웃음을 크게 봐요.' },
      family: { label: '아이와 가족', hint: '움직임이 많아서 인물 선명도를 가장 크게 보고 웃음도 봐요.' },
      other: { label: '기타', hint: '항목을 고르게 봐요.' },
    },

    // ---------- 분석 ----------
    analyzeTitle: '사진 분석 중',
    skipFaces: '얼굴 분석 끄고 빨리 진행',
    loadingModel: '얼굴 분석 모델을 불러오는 중이에요.',
    modelFailed: '얼굴 분석 모델을 불러오지 못해서 선명도와 노출만 보고 진행해요.',
    analyzeProgress: ({ done, total, seconds, faceOff }) => `${done} / ${total}장, 남은 시간 약 ${seconds >= 60 ? `${Math.round(seconds / 60)}분` : `${seconds}초`}${faceOff ? ', 얼굴 분석 꺼짐' : ''}`,
    noReadable: '이 브라우저가 연 사진이 없어요. JPG로 바꾼 사진을 골라주세요.',
    skipNote: (count) => ` 열 수 없는 파일 ${count}개(RAW, HEIC 등)는 건너뛰었어요.`,
    analyzedToast: ({ count, groups, skipNote }) => `${count}장을 분석했어요. 비슷한 컷 묶음 ${groups}개.${skipNote}`,
    skipFacesToast: '남은 사진은 선명도와 노출만 보고 빠르게 분석해요.',
    learnedToast: (count) => `결정 ${count}건으로 취향을 배웠어요. 남은 사진 순서를 다시 정했어요. 점수 기준에서 무엇을 중요하게 보는지 확인할 수 있어요.`,

    // ---------- 점수 항목, 인원 분류 ----------
    features: { subjectSharp: '인물 선명도', focusSharp: '초점 부분 선명도', exposure: '노출', expression: '찡그림 없음', smile: '웃음', framing: '얼굴 안 잘림', clean: '노이즈 적음', eyesOpen: '눈 뜸' },
    fieldEyes: '눈 감은 사진', eyesCheck: '눈 감은 사진은 점수 깎기',
    eyesHint: '눈을 감아도 괜찮거나, 웨딩 촬영처럼 작가가 눈을 합성해 준다면 체크를 해제하세요.',
    eyesOnToast: '눈 감은 사진의 점수를 깎도록 바꿨어요. 남은 사진 순서를 다시 정했어요.',
    eyesOffToast: '눈 감은 사진도 감점하지 않도록 바꿨어요. 남은 사진 순서를 다시 정했어요.',
    categories: { all: '전체', solo: '혼자', pair: '둘', small: '소그룹', crowd: '단체', none: '인물 없음', unknown: '분류 안 됨' },

    // ---------- 고르기 화면 ----------
    compareToggle: '두 장 나란히', criteriaButton: '점수 기준', filmstripAria: '다음에 볼 사진',
    nothingToUndo: '되돌릴 결정이 없어요.',
    switchedReviewer: (name) => `이제 ${name}의 결정으로 골라요. 결정과 학습은 사람마다 따로 저장돼요.`,
    filmCurrent: '지금 보는 사진', filmNth: (order) => `${order}번째로 볼 사진`, filmMore: (count) => `${count}장 더`,
    tagScore: (score) => `점수 ${score}`,
    tagSimilar: (count) => `비슷한 컷 ${count}장 중 하나`,
    tagEyes: '눈 감음 의심', tagEyesWedding: '눈 감음 의심, 합성 요청 후보', tagEyesShort: '눈 감음',
    tagReAdded: (time) => `${time}에 다시 남김`,
    noticeFirst: ({ remaining, scored }) => `남은 사진 <b class="num">${remaining}</b>장${scored ? '. 점수 높은 사진부터 보여드려요.' : ''}`,
    firstDoneTitle: '1차 셀렉을 마쳤어요',
    firstDoneBody: ({ keep, target }) => `남긴 사진 ${keep}장, 목표 ${target}장. ${keep > target ? '줄이기에서 목표 장수까지 줄여보세요.' : '뺀 사진 중 아까운 컷이 있는지 다시 볼 수 있어요.'}`,
    btnDistill: '줄이기', btnRescueDropped: '뺀 사진 다시 보기', btnExport: '내보내기',
    noticeDistillOver: (count) => `목표까지 <b class="num">${count}장</b> 더 빼야 해요. 점수 낮은 사진부터 보여드려요.`,
    noticeDistillOk: (target) => `목표 ${target}장을 맞췄어요. 더 줄이려면 계속 넘기세요.`,
    distillDoneTitle: '이번 줄이기를 다 봤어요',
    distillDoneBody: (keep) => `남긴 사진은 ${keep}장이에요. 한 번 더 돌면 남은 사진을 다시 점수 낮은 순으로 보여줘요.`,
    btnDistillAgain: '한 번 더 줄이기',
    noticeRescue: '뺀 사진을 점수 높은 순으로 다시 봐요. 남기면 다시 남긴 시각이 기록돼요.',
    rescueDoneTitle: '뺀 사진을 다 다시 봤어요',
    rescueDoneBody: ({ keep, target }) => `남긴 사진 ${keep}장, 목표 ${target}장.`,
    btnRescueAgain: '처음부터 다시 보기',
    dividerTitle: '점수 높은 사진을 다 봤어요',
    dividerBody: (count) => `점수가 낮은 ${count}개 묶음과 낱장이 남았어요. 점수가 낮을 뿐 볼만한 컷이 섞여 있을 수 있어요. 지금 모두 빼도 나중에 '다시 보기'에서 볼 수 있어요.`,
    dividerContinue: '남은 사진도 보기', dividerDrop: '남은 사진 모두 빼기',
    droppedRest: (count) => `남은 ${count}장을 뺐어요.`,
    undo: '되돌리기', undoTitle: '되돌리기 (Z)', zoomTitle: '얼굴 크게 보기 (Space)', facesTitle: '얼굴 표시 (F)',
    drop: '빼기', keep: '남기기', keepStill: '남김 유지',
    rescuedOver: (count) => `다시 남겼어요. 이제 목표보다 ${count}장 많아요. 줄이기에서 점수 낮은 사진부터 빼주세요.`,
    goDistill: '줄이기로 가기',
    burstTitle: (count) => `비슷한 컷 ${count}장`,
    burstHint: '남길 사진을 누르세요. 안 누른 사진은 빠져요.',
    burstHintScored: ' 점수가 가장 높은 한 장을 미리 골라뒀어요.',
    burstExpand: '한 장씩 보기',
    burstShotTitle: (path) => `${path}. 더블클릭하면 크게 봐요`,
    keepCount: (count) => `${count}장 남기기`,
    dropAll: '모두 빼기',
    compareKeepLeft: '왼쪽 사진 남기기', compareKeepRight: '오른쪽 사진 남기기',
    keepLeft: '왼쪽 남기기', keepBoth: '둘 다 남기기', keepRight: '오른쪽 남기기',
    tagAlreadyKept: '이미 남김', tagAlreadyDropped: '이미 뺌',
    burstWithKept: (count) => `, 이미 남긴 비슷한 사진 ${count}장`,
    keptCompareTitle: '이미 남긴 비슷한 사진이 있어요',
    keptCompareHint: '둘 중 무엇을 남길지 고르세요. 사진을 누르면 크게 볼 수 있어요.',
    keptCompareNext: ({ index, total }) => `다른 비슷한 사진 (${index}/${total})`,
    keptCompareSingle: '이 사진만 보기',
    labelCurrent: '지금 사진', labelKept: '이미 남긴 사진',
    dropCurrent: '지금 사진 빼기', swapToCurrent: '이 사진으로 바꾸기',
    dropCurrentShort: '빼기', keepBothShort: '둘 다', swapShort: '바꾸기',
    markKeep: '남김', markDrop: '뺌',

    checkingBackend: '이 기기에 맞는 얼굴 분석 방식을 확인하는 중이에요.',
    diagTitle: '분석 정보',
    diagNoFace: '얼굴 분석을 하지 않았어요.',
    diagBackend: ({ backend, float32 }) => `얼굴 분석 계산 방식: ${backend}${float32 === false ? ' (이 기기 GPU는 32비트 실수 계산을 지원하지 않음)' : ''}`,
    diagCrossCheck: ({ samples, webglFaces, cpuFaces, switched }) => `처음 ${samples}장 비교: WebGL로 얼굴 ${webglFaces}개, CPU로 ${cpuFaces}개 찾음.${switched ? ' WebGL 결과가 부족해서 CPU로 바꿔 분석했어요.' : ''}`,
    diagFaces: ({ found, total }) => `얼굴을 찾은 사진: ${found} / ${total}장`,
    diagDark: ({ hits, tries }) => `어두운 사진 다시 찾기: ${tries}장 중 ${hits}장에서 얼굴을 찾음`,
    diagHint: '얼굴이 있는데 못 찾는 것 같으면 주소 끝에 ?faceBackend=cpu 를 붙여 다시 열어보세요.',
    // ---------- 점수 기준 패널 ----------
    criteriaTitle: '점수 기준',
    criteriaLearned: ({ name, count, every }) => `${name}의 결정 ${count}건으로 배운 기준이에요. 결정이 ${every}건 쌓일 때마다 다시 배워요.`,
    criteriaServer: ({ preset, min }) => `다른 사용자들의 ${preset} 사진 결정으로 만든 기본 기준이에요. 결정이 ${min}건 쌓이면 내 취향으로 다시 맞춰요.`,
    criteriaPreset: ({ preset, perClass, min }) => `${preset} 기본 기준이에요. 남김과 뺌이 각각 ${perClass}건 이상, 합쳐서 ${min}건이 쌓이면 내 취향으로 다시 맞춰요.`,
    colItem: '항목', colWeight: '비중', colThis: '이 사진',
    criteriaNote: '눈 감은 사진은 점수에 넣지 않고 표시만 해요. 선명도와 노이즈는 이 폴더 사진들 사이의 순위로 계산해요.',
    criteriaNoteEyesOn: '눈 감은 사진은 점수를 깎아요. 선명도와 노이즈는 이 폴더 사진들 사이의 순위로 계산해요.',

    // ---------- 내보내기 ----------
    exportTitle: '내보내기',
    summaryBoth: ({ nameA, nameB, keepA, keepB, both, onlyA, onlyB }) => `${nameA} <b>${keepA}장</b>, ${nameB} <b>${keepB}장</b>을 남겼어요. 둘 다 남긴 사진은 <b>${both}장</b>, 한 명만 남긴 사진은 ${onlyA + onlyB}장(${nameA} ${onlyA}, ${nameB} ${onlyB})이에요.`,
    summaryOne: ({ name, count }) => `${name} 기준으로 <b>${count}장</b>을 남겼어요.`,
    saveTitle: '원본 저장',
    saveHintFolder: '고른 사진의 원본을 새 폴더로 복사하고, 같은 폴더에 파일 목록을 남겨요. 원래 폴더는 건드리지 않아요.',
    saveHintShare: (count) => `공유 창을 ${count}장씩 열어요. 공유 창에서 이미지 저장을 누르면 사진 보관함에 저장돼요.`,
    saveHintDownload: '고른 사진을 한 장씩 내려받아요.',
    basisAria: '내보낼 사진', basisCurrent: (name) => `${name} 기준`, basisBoth: '둘 다 남긴 사진', basisEither: '한 명이라도 남긴 사진',
    copyToFolder: (count) => `${count}장 복사할 폴더 선택`,
    downloadEach: '한 장씩 내려받기', copyList: '파일 목록 복사',
    targetLabel: '남길 장수',
    targetOver: (count) => `지금 ${count}장 많아요.`, targetUnder: (count) => `${count}장 더 남길 수 있어요.`, targetExact: '딱 맞아요.',
    eyesInList: (count) => ` 눈 감음 의심 ${count}장은 목록 파일에 따로 적어둬요.`,
    mergeTitle: '둘이 고른 결과 합치기',
    mergeHint: '같은 기기라면 위쪽에서 사람을 바꿔 고르면 돼요. 서로 다른 기기에서 골랐다면 한 사람이 결정 파일을 저장해 보내고, 다른 사람이 같은 사진을 연 상태에서 불러오면 됩니다. 결정은 파일 이름으로 맞추니 같은 원본을 써야 해요.',
    nameAAria: '첫 번째 사람 이름', nameBAria: '두 번째 사람 이름',
    saveDecisions: (name) => `${name} 결정 파일 저장`, loadDecisions: '다른 사람 결정 파일 불러오기',
    pickedTitle: '고른 사진', pickedCount: (count) => `${count}장`,
    noneKept: '아직 남긴 사진이 없어요. 1차 셀렉에서 오른쪽으로 넘기면 여기에 모여요.',
    viewLarge: (name) => `${name} 크게 보기`, reAddedTitle: '다시 남긴 사진', removeTitle: '빼기',
    listCopied: '파일 목록을 복사했어요.',
    listHeader: (count) => `모먼트 셀렉 목록, ${count}장`, listFolder: (name) => `폴더: ${name}`, listCreated: (time) => `만든 시각: ${time}`,
    listEyes: ({ count, wedding }) => `눈 감음 의심 ${count}장${wedding ? ' (합성 요청 후보)' : ''}`,
    listFileName: '셀렉목록.txt', exportFolderPrefix: '모먼트셀렉_',
    decisionFileName: (name) => `셀렉결정_${name}.json`,
    notDecisionFile: '모먼트 셀렉에서 저장한 결정 파일이 아니에요.',
    decisionsLoaded: ({ name, count, skipped }) => `${name}의 결정 ${count}건을 불러왔어요.${skipped ? ' 이 폴더에 없는 사진의 결정은 건너뛰었어요.' : ''}`,
    decisionsLoadError: (message) => `결정 파일을 불러오지 못했어요. ${message}`,
    copyingProgress: ({ done, total }) => `${done} / ${total}장 복사 중`,
    downloadingProgress: ({ done, total }) => `${done} / ${total}장 받는 중`,
    copiedToast: ({ count, path }) => `${count}장을 '${path}' 폴더에 복사했어요.`,
    copyError: (message) => `복사하지 못했어요. ${message} 저장 공간과 폴더 권한을 확인해주세요.`,
    shareAllDone: '모두 저장했어요. 처음부터 다시 저장',
    shareRange: ({ start, end, total }) => `${start}~${end}번째 사진 저장 (전체 ${total}장)`,
    shareError: (message) => `공유 창을 열지 못했어요. ${message} 한 장씩 내려받기를 써주세요.`,
    viewerClose: '닫기',
    formatTime: (ms) => { const date = new Date(ms); return `${date.getMonth() + 1}월 ${date.getDate()}일 ${date.getHours()}시 ${String(date.getMinutes()).padStart(2, '0')}분`; },
  },

  en: {
    brand: 'Moment Select',
    tabSetup: 'Start', tabFirst: 'First pass', tabDistill: 'Narrow down', tabRescue: 'Second look', tabExport: 'Export',
    stepsAria: 'Steps',
    goalKept: 'Kept', goalUnit: '',
    goalOver: (count) => `${count} over`,
    reviewerAria: 'Who is picking', languageAria: 'Language',
    defaultNameA: 'Me', defaultNameB: 'Partner',

    setupTitle: 'Pick the photos to keep',
    setupIntro: 'Choose your photos and they are analyzed right on this device, then you swipe through them one by one to decide what to keep. Nothing is uploaded, and the app never talks to an outside server.',
    driveNote: 'Photos in Google Drive show up as a folder once Google Drive for desktop is installed. On Windows, pick your photo folder under the Google Drive drive in File Explorer. On Mac, find it under Google Drive in the Finder sidebar. The first analysis can be slow while photos download, so make the folder available offline first.',
    resumeButton: 'Continue',
    resumeName: (name) => `Last folder: ${name}`,
    fieldPreset: 'Photo type', presetAria: 'Photo type',
    fieldTarget: 'Photos to keep', targetHint: 'You can change this at any time.',
    fieldGrouping: 'Group by', categoryCheck: 'Group by number of people',
    categoryHint: 'Filter by solo, two people, small group, and crowd. Turn this off to see everything in one line.',
    crowdLabel: 'Count as a crowd from', crowdUnit: 'people',
    fieldScore: 'Order', scoreCheck: 'Show the best shots first',
    scoreHint: 'Orders photos by face sharpness, exposure and expression, then learns from your choices and reorders as you go.',
    fieldPeople: 'Who is picking', personAAria: 'First person', personBAria: 'Second person',
    peopleHint: 'Two people can pick separately and merge later. If you are picking alone, leave this as is.',
    fieldShare: 'Share stats', shareCheck: 'Send anonymous numbers to improve scoring',
    shareHint: 'No photos, file names or face data are sent. Only the 7 scoring values (numbers between 0 and 1) per photo, whether you kept it, and the photo type.',
    pickFolder: 'Choose photo folder', pickPhotos: 'Choose photos',
    pickNoteDefault: 'Reads JPG, PNG and WEBP files, including subfolders. RAW files are skipped.',
    pickNoteTouch: 'Select all the photos you want to go through from your library or gallery. They never leave this device.',
    pickNoteNoFolder: 'This browser cannot save into a folder, so exports download one photo at a time. Open this page in Chrome or Edge to copy straight into a new folder.',
    pickNoteReading: 'Reading the photo list.',
    fallbackFolderName: 'Selected folder',
    pickedSourceName: ({ count, first }) => `${photosEn(count)} (from ${first})`,
    noPhotosFound: 'No readable photos (JPG, PNG, WEBP) found. Choose another folder or other photos.',
    folderOpenError: (message) => `Could not open the folder. ${message}`,
    permissionNeeded: 'Allow access to read the folder.',

    presets: {
      wedding: { label: 'Wedding and portraits', hint: 'Posed shots are often calm, so sharpness and the absence of frowns count more than smiles.' },
      travel: { label: 'Travel', hint: 'Lots of scenery, so overall focus and exposure count most.' },
      event: { label: 'Events and gatherings', hint: 'Lots of group shots, so visible faces and smiles count more.' },
      family: { label: 'Kids and family', hint: 'Lots of movement, so subject sharpness counts most, with smiles next.' },
      other: { label: 'Other', hint: 'Every factor counts about equally.' },
    },

    analyzeTitle: 'Analyzing photos',
    skipFaces: 'Skip face analysis to go faster',
    loadingModel: 'Loading the face analysis model.',
    modelFailed: 'Could not load the face analysis model, so only sharpness and exposure are used.',
    analyzeProgress: ({ done, total, seconds, faceOff }) => `${done} / ${total}, about ${seconds >= 60 ? `${Math.round(seconds / 60)} min` : `${seconds} s`} left${faceOff ? ', face analysis off' : ''}`,
    noReadable: 'This browser could not open any of these photos. Choose photos converted to JPG.',
    skipNote: (count) => ` Skipped ${count} unreadable ${count === 1 ? 'file' : 'files'} (RAW, HEIC and similar).`,
    analyzedToast: ({ count, groups, skipNote }) => `Analyzed ${photosEn(count)}. Found ${groups} ${groups === 1 ? 'set' : 'sets'} of similar shots.${skipNote}`,
    skipFacesToast: 'The remaining photos are analyzed for sharpness and exposure only.',
    learnedToast: (count) => `Learned your taste from ${count} decisions and reordered the remaining photos. Open Scoring to see what matters most now.`,

    features: { subjectSharp: 'Subject sharpness', focusSharp: 'In-focus area sharpness', exposure: 'Exposure', expression: 'No frowns', smile: 'Smiles', framing: 'Faces not cut off', clean: 'Low noise', eyesOpen: 'Eyes open' },
    fieldEyes: 'Closed eyes', eyesCheck: 'Lower the score of closed-eye shots',
    eyesHint: 'Uncheck this if closed eyes are fine, or if your photographer will retouch them, as wedding studios often do.',
    eyesOnToast: 'Closed-eye shots now score lower. The remaining photos were reordered.',
    eyesOffToast: 'Closed-eye shots are no longer penalized. The remaining photos were reordered.',
    categories: { all: 'All', solo: 'Solo', pair: 'Two', small: 'Small group', crowd: 'Crowd', none: 'No people', unknown: 'Unsorted' },

    compareToggle: 'Side by side', criteriaButton: 'Scoring', filmstripAria: 'Up next',
    nothingToUndo: 'Nothing to undo.',
    switchedReviewer: (name) => `Now picking as ${name}. Decisions and learning are saved separately for each person.`,
    filmCurrent: 'Current photo', filmNth: (order) => `Photo ${order} in line`, filmMore: (count) => `${count} more`,
    tagScore: (score) => `Score ${score}`,
    tagSimilar: (count) => `One of ${count} similar shots`,
    tagEyes: 'Eyes may be closed', tagEyesWedding: 'Eyes may be closed, ask for a retouch', tagEyesShort: 'Eyes closed',
    tagReAdded: (time) => `Kept again ${time}`,
    noticeFirst: ({ remaining, scored }) => `<b class="num">${remaining}</b> left${scored ? '. Best scores come first.' : ''}`,
    firstDoneTitle: 'First pass done',
    firstDoneBody: ({ keep, target }) => `Kept ${keep}, target ${target}. ${keep > target ? 'Narrow down to reach your target.' : 'You can take a second look at the ones you dropped.'}`,
    btnDistill: 'Narrow down', btnRescueDropped: 'Second look at dropped', btnExport: 'Export',
    noticeDistillOver: (count) => `Drop <b class="num">${count}</b> more to reach your target. Lowest scores come first.`,
    noticeDistillOk: (target) => `You are at your target of ${target}. Keep swiping to narrow further.`,
    distillDoneTitle: 'This round is done',
    distillDoneBody: (keep) => `${keep} kept. Another round shows the remaining photos again, lowest score first.`,
    btnDistillAgain: 'Another round',
    noticeRescue: 'Dropped photos, highest score first. Anything you keep again gets a timestamp.',
    rescueDoneTitle: 'You have seen every dropped photo',
    rescueDoneBody: ({ keep, target }) => `Kept ${keep}, target ${target}.`,
    btnRescueAgain: 'Start over',
    dividerTitle: 'You have seen all the top-scoring photos',
    dividerBody: (count) => `${count} lower-scoring shots and sets are left. A low score does not mean a bad photo, so some may be worth a look. If you drop them all now, you can still find them in Second look.`,
    dividerContinue: 'Keep going', dividerDrop: 'Drop the rest',
    droppedRest: (count) => `Dropped the remaining ${photosEn(count)}.`,
    undo: 'Undo', undoTitle: 'Undo (Z)', zoomTitle: 'Zoom to face (Space)', facesTitle: 'Show faces (F)',
    drop: 'Drop', keep: 'Keep', keepStill: 'Still keep',
    rescuedOver: (count) => `Kept again. You are now ${count} over your target. Narrow down, starting with the lowest scores.`,
    goDistill: 'Narrow down',
    burstTitle: (count) => `${count} similar shots`,
    burstHint: 'Tap the ones to keep. The rest are dropped.',
    burstHintScored: ' The top-scoring shot is already selected.',
    burstExpand: 'One at a time',
    burstShotTitle: (path) => `${path}. Double-click to enlarge`,
    keepCount: (count) => `Keep ${count}`,
    dropAll: 'Drop all',
    compareKeepLeft: 'Keep the left photo', compareKeepRight: 'Keep the right photo',
    keepLeft: 'Keep left', keepBoth: 'Keep both', keepRight: 'Keep right',
    tagAlreadyKept: 'Already kept', tagAlreadyDropped: 'Already dropped',
    burstWithKept: (count) => `, plus ${count} similar already kept`,
    keptCompareTitle: 'You already kept a similar photo',
    keptCompareHint: 'Choose which one to keep. Tap a photo to enlarge it.',
    keptCompareNext: ({ index, total }) => `Another similar photo (${index}/${total})`,
    keptCompareSingle: 'Show this photo alone',
    labelCurrent: 'This photo', labelKept: 'Already kept',
    dropCurrent: 'Drop this one', swapToCurrent: 'Keep this one instead',
    dropCurrentShort: 'Drop', keepBothShort: 'Both', swapShort: 'Swap',
    markKeep: 'keep', markDrop: 'drop',

    checkingBackend: 'Checking which face analysis mode works best on this device.',
    diagTitle: 'Analysis details',
    diagNoFace: 'Face analysis was not run.',
    diagBackend: ({ backend, float32 }) => `Face analysis runs on: ${backend}${float32 === false ? ' (this GPU cannot render 32-bit floats)' : ''}`,
    diagCrossCheck: ({ samples, webglFaces, cpuFaces, switched }) => `Compared the first ${samples} photos: WebGL found ${webglFaces} faces, CPU found ${cpuFaces}.${switched ? ' WebGL missed too many, so analysis switched to CPU.' : ''}`,
    diagFaces: ({ found, total }) => `Photos with faces: ${found} / ${total}`,
    diagDark: ({ hits, tries }) => `Dark photo retries: faces found in ${hits} of ${tries}`,
    diagHint: 'If faces seem to be missed, reopen the page with ?faceBackend=cpu at the end of the address.',
    criteriaTitle: 'Scoring',
    criteriaLearned: ({ name, count, every }) => `Learned from ${count} decisions by ${name}. It relearns every ${every} decisions.`,
    criteriaServer: ({ preset, min }) => `Default weights for ${preset}, built from other people's choices. After ${min} decisions it adapts to your taste.`,
    criteriaPreset: ({ preset, perClass, min }) => `Default weights for ${preset}. Once you have at least ${perClass} keeps and ${perClass} drops, ${min} in total, it adapts to your taste.`,
    colItem: 'Factor', colWeight: 'Weight', colThis: 'This photo',
    criteriaNote: 'Closed eyes are flagged but not scored. Sharpness and noise are ranked against the other photos in this set.',
    criteriaNoteEyesOn: 'Closed-eye shots score lower. Sharpness and noise are ranked against the other photos in this set.',

    exportTitle: 'Export',
    summaryBoth: ({ nameA, nameB, keepA, keepB, both, onlyA, onlyB }) => `${nameA} kept <b>${keepA}</b> and ${nameB} kept <b>${keepB}</b>. <b>${both}</b> were kept by both, ${onlyA + onlyB} by only one (${nameA} ${onlyA}, ${nameB} ${onlyB}).`,
    summaryOne: ({ name, count }) => `${name} kept <b>${photosEn(count)}</b>.`,
    saveTitle: 'Save originals',
    saveHintFolder: 'Copies the originals into a new folder along with a file list. Your original folder is left untouched.',
    saveHintShare: (count) => `Opens the share sheet ${count} photos at a time. Tap Save Image to put them in your photo library.`,
    saveHintDownload: 'Downloads the selected photos one by one.',
    basisAria: 'Photos to export', basisCurrent: (name) => `${name}'s picks`, basisBoth: 'Kept by both', basisEither: 'Kept by either',
    copyToFolder: (count) => `Copy ${photosEn(count)} to a folder`,
    downloadEach: 'Download one by one', copyList: 'Copy file list',
    targetLabel: 'Photos to keep',
    targetOver: (count) => `${count} over.`, targetUnder: (count) => `Room for ${count} more.`, targetExact: 'Right on target.',
    eyesInList: (count) => ` ${count} with possibly closed eyes are listed separately in the file list.`,
    mergeTitle: 'Merge two people\'s picks',
    mergeHint: 'On the same device, switch the person at the top and pick again. On different devices, one person saves a decision file and sends it, and the other loads it with the same photos open. Decisions are matched by file name, so use the same originals.',
    nameAAria: 'First person\'s name', nameBAria: 'Second person\'s name',
    saveDecisions: (name) => `Save ${name}'s decisions`, loadDecisions: 'Load the other person\'s decisions',
    pickedTitle: 'Selected photos', pickedCount: (count) => photosEn(count),
    noneKept: 'Nothing kept yet. Swipe right in the first pass and photos collect here.',
    viewLarge: (name) => `Enlarge ${name}`, reAddedTitle: 'Kept again', removeTitle: 'Drop',
    listCopied: 'File list copied.',
    listHeader: (count) => `Moment Select list, ${photosEn(count)}`, listFolder: (name) => `Folder: ${name}`, listCreated: (time) => `Created: ${time}`,
    listEyes: ({ count, wedding }) => `Eyes may be closed: ${count}${wedding ? ' (ask for a retouch)' : ''}`,
    listFileName: 'selection-list.txt', exportFolderPrefix: 'MomentSelect_',
    decisionFileName: (name) => `decisions_${name}.json`,
    notDecisionFile: 'This is not a decision file saved by Moment Select.',
    decisionsLoaded: ({ name, count, skipped }) => `Loaded ${count} decisions from ${name}.${skipped ? ' Decisions for photos not in this set were skipped.' : ''}`,
    decisionsLoadError: (message) => `Could not load the decision file. ${message}`,
    copyingProgress: ({ done, total }) => `Copying ${done} / ${total}`,
    downloadingProgress: ({ done, total }) => `Downloading ${done} / ${total}`,
    copiedToast: ({ count, path }) => `Copied ${photosEn(count)} to '${path}'.`,
    copyError: (message) => `Could not copy. ${message} Check free space and folder permissions.`,
    shareAllDone: 'All saved. Save again from the start',
    shareRange: ({ start, end, total }) => `Save photos ${start} to ${end} (of ${total})`,
    shareError: (message) => `Could not open the share sheet. ${message} Use Download one by one instead.`,
    viewerClose: 'Close',
    formatTime: (ms) => new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }).format(new Date(ms)),
  },
};

// 처음 언어: 저장해 둔 선택, 없으면 브라우저 언어 (한국어가 아니면 영어)
function detectLanguage() {
  let saved = null;
  try { saved = localStorage.getItem(LANGUAGE_STORAGE_KEY); } catch { saved = null; }
  if (SUPPORTED_LANGUAGES.includes(saved)) return saved;
  return (navigator.language || 'en').toLowerCase().startsWith('ko') ? 'ko' : 'en';
}

let currentLanguage = detectLanguage();

export const getLanguage = () => currentLanguage;

export function setLanguage(language) {
  if (!SUPPORTED_LANGUAGES.includes(language)) return;
  currentLanguage = language;
  try { localStorage.setItem(LANGUAGE_STORAGE_KEY, language); } catch { /* 저장이 막힌 환경이면 이번 방문에만 적용 */ }
  applyStaticText();
}

// t('key') 는 문구를, t('key', 값) 은 함수 문구에 값을 넣은 결과를 돌려준다
export function t(key, params) {
  const entry = MESSAGES[currentLanguage][key] ?? MESSAGES.ko[key];
  return typeof entry === 'function' ? entry(params) : entry;
}

// 언어마다 바뀌는 묶음 (사진 종류, 점수 항목, 인원 분류)
export const presetText = (presetKey) => MESSAGES[currentLanguage].presets[presetKey];
export const featureLabel = (featureKey) => MESSAGES[currentLanguage].features[featureKey];
export const categoryLabel = (categoryKey) => MESSAGES[currentLanguage].categories[categoryKey];

// 내보낸 폴더를 다시 읽지 않도록, 두 언어의 폴더 이름 앞부분을 모두 알려준다
export const allExportPrefixes = () => SUPPORTED_LANGUAGES.map((language) => MESSAGES[language].exportFolderPrefix);
// 사람 이름이 기본값이면 언어를 바꿀 때 같이 바꿔준다
export const isDefaultName = (name) => SUPPORTED_LANGUAGES.some((language) => [MESSAGES[language].defaultNameA, MESSAGES[language].defaultNameB].includes(name));

export function applyStaticText(root = document) {
  document.documentElement.lang = currentLanguage;
  document.title = t('brand');
  for (const element of root.querySelectorAll('[data-i18n]')) element.textContent = t(element.dataset.i18n);
  for (const element of root.querySelectorAll('[data-i18n-aria]')) element.setAttribute('aria-label', t(element.dataset.i18nAria));
  for (const element of root.querySelectorAll('[data-i18n-title]')) element.setAttribute('title', t(element.dataset.i18nTitle));
}
