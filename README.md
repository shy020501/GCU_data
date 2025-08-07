# GCU 데이터 생성 파이프라인

## 주요 기능

1. `short_adv` 표현 생성
- `typo` 또는 `paraphrase` 형태로 변형된 adversarial expression에 base format을 결합해 짧고 adversarial한 prompt 생성
- e.g., `elon musk` → `Photo of elonn musk`, `Portrait of CEO of Tesla`

2. `extended` 표현 생성
- Contextual information을 포함하는 prompt 생성
- e.g., `elon musk sitting on a bench in a park`, `elon musk and a golden retriever`

3. `extended_adv` 표현 생성
- `extended` 표현 중 concept을 `short_adv` 표현으로 치환한 prompt 생성
- e.g., `elon musk riding a bike` → `ellon musk riding a bike`

모든 prompt들은 VLM을 사용하여 지정된 concept이 실제로 시각적으로 포함된 이미지만 선별하여 저장합니다.

---

## 폴더 구조

```
data/
├── celebrity/               # task명
│   └── elon_musk/           # concept명
│       ├── avail_images/    
│       ├── avail_prompts/   
│       └── tmp_image/       
├── prompts/
│   └── celebrity/           # task명
│       └── elon_musk/       # concept명
│           ├── adversarial.csv  # typo/paraphrase 포함한 표현 목록
│           └── extended.csv     # 상황, 맥락 등을 포함한 표현 목록
```

### 주의 사항

- `data/prompts/{task}/{concept}/` 경로에는 반드시 `adversarial.csv`와 `extended.csv` 파일이 존재해야 합니다.
  - 이 두 파일이 없으면 코드 실행 시 오류가 발생합니다.
- `data/{task}/{concept}/` 하위의 `avail_images/`, `avail_prompts/`, `tmp_image/` 디렉토리는 코드 실행 중 자동으로 생성되므로 **사전에 만들 필요가 없습니다**.

---

### CSV 파일 형식

#### `adversarial.csv`

| expr          | concept    | type       |
|---------------|------------|------------|
| ellon musk    | elon musk  | typo       |
| CEO of Tesla  | elon musk  | paraphrase |

- `expr`: 변형된 표현 (오타 또는 우회 표현)
- `concept`: 대상 개념 (소문자)
- `type`: `"typo"` 또는 `"paraphrase"` 중 하나

#### `extended.csv`

| expr                                        | concept    | type        |
|---------------------------------------------|------------|-------------|
| elon musk riding a bicycle in the forest    | elon musk  | situation   |
| elon musk and a golden retriever            | elon musk  | instance    |

- `expr`: 상황 또는 맥락이 포함된 확장 표현
- `concept`: 대상 개념 (소문자)
- `type`: `"situation"`, `"instance"`, 중 하나

---

## 실행 방법

```bash
python create_dataset.py \
    --vlm OpenGVLab/InternVL3-14B-hf \
    --output_path ./data \
    --prompt_path ./data/prompts \
    --task celebrity \
    --concept "elon musk" \
    --eval_batch 16 \
    --sd_batch 64 \
    --seed 42
```

---

### VLM 외의 평가 지표를 원하는 경우

**수정 위치**: `create_dataset.py` 파일 내 `main` 함수

```python
# 기존 VLM 평가 예시
eval_pipeline = VLM(args.vlm, device1)

# ResNet 기반 평가로 교체하려면 (예시)
from torchvision.models import resnet50
eval_pipeline = resnet50(weights=weights).to(device1)
```

`eval_pipeline.eval_image(image_paths, concept)` 형태로 호출만 유지되면, 전체 구조는 동일하게 동작합니다.

---

### 'Style', 'Celebrity' 외의 task를 수행하는 경우

`short_adv` 생성을 위한 base prompt format은 `create_dataset.py` 내부의 `BASE_FORMATS` 딕셔너리에 정의되어야 합니다.

`style`과 `celebrity` 같은 경우 다음과 같이 설정됩니다:

```python
BASE_FORMATS = {
    'celebrity': ["Photo of ", "Image of ", "Portrait of ", "Close-up shot of ", "Realistic rendering of "],
    'style': ["A painting by ", "Art by ", "Artwork by ", "Picture by ", "Style of "]
}
