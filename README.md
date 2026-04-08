# Scalp Vision Agent

> 두피 이미지 기반 증상 등급 예측 및 상담 리포트 자동 생성 시스템

병원·클리닉 환경에서 두피 사진 한 장을 업로드하면, 6개 증상을 자동으로 등급화하고 상담용 자연어 리포트를 생성합니다. 방문 이력을 누적 저장해 이전 방문과의 변화를 비교할 수 있습니다.

---

## 기획 의도

두피 상태 평가는 검사자에 따라 기록 형식과 표현 수준이 달라질 수 있습니다.

- 같은 환자라도 검사자마다 기록 방식이 다름
- 이전 방문과 현재 상태를 일관되게 비교하기 어려움
- 상담 내용이 비정형적으로 남아 추후 추적이 어려움

이 프로젝트는 **검사 결과의 일관성, 비교 가능성, 기록 표준화**를 높이기 위한 B2B 의료 보조 서비스입니다.

---

## 서비스 흐름

```
두피 이미지 업로드
       ↓
EfficientNet-B0 추론 (6개 증상 등급 예측)
       ↓
규칙 기반 위험도 계산 (risk_score / risk_level)
       ↓
OpenAI API 자연어 리포트 생성
       ↓
DB 저장 + snapshot JSON 기록
       ↓
이전 방문 대비 변화 비교 제공
```
https://github.com/user-attachments/assets/1b4ca108-46e1-449c-9899-84fc22ea4387  


**예측 대상 증상**

| 항목 | 증상 | 등급 |
|------|------|------|
| value_1 | 미세각질 | 0~3 |
| value_2 | 피지과다 | 0~3 |
| value_3 | 모낭사이홍반 | 0~3 |
| value_4 | 모낭홍반/농포 | 0~3 |
| value_5 | 비듬 | 0~3 |
| value_6 | 탈모 | 0~3 |

---

## 모델링

### 구조

하나의 두피 이미지에서 6개 증상을 동시에 예측하는 **멀티헤드 CNN** 구조를 사용했습니다.  
각 head는 독립적으로 0~3등급을 예측합니다.

### 실험 비교

| 실험 | 백본 | 손실함수 | val_acc | val_value6_acc |
|------|------|----------|---------|----------------|
| E4_baseline | ResNet18 | CrossEntropy | 0.7601 | 0.7509 |
| E6_baseline | EfficientNet-B0 | CrossEntropy | 0.7593 | - |
| **E8_v6_focal** | **EfficientNet-B0** | **Focal Loss (value_6)** | **0.7630** | **0.7609** |

### 핵심 의사결정: Focal Loss 적용

탈모(value_6)는 데이터셋 특성상 **0등급(정상) 비율이 압도적으로 높아** 모델이 정상 편향 예측을 학습하는 문제가 있었습니다.

단순 CrossEntropy 대신 **Focal Loss(γ=2)**를 value_6 head에만 적용해, 모델이 소수 클래스(중·고등급 탈모)에 집중하도록 유도했습니다.

**value_6 클래스별 recall 비교 (E4 → E8)**

| 등급 | E4_baseline | E8_v6_focal | 변화 |
|------|-------------|-------------|------|
| class_0 (정상) | 0.944 | 0.944 | - |
| class_1 (경미) | 0.370 | 0.326 | -4.4%p |
| class_2 (중간) | **0.181** | **0.444** | **+26.3%p ↑** |
| class_3 (심함) | 0.481 | 0.365 | -11.6%p |

전체 accuracy 하락 없이 **중간 단계(class_2) 감지 능력을 두 배 이상** 향상시켰습니다.  
임상적으로 중증으로 진행되기 전 단계인 class_2를 놓치지 않는 것이 서비스 목적에 부합한다고 판단했습니다.

> class_3 recall이 소폭 하락했으나, 중간 단계 조기 감지 개선 효과가 더 크다고 보아 E8을 최종 모델로 채택했습니다.

### 최종 모델 성능 (E8_v6_focal, head별 accuracy)

| 증상 | accuracy |
|------|----------|
| value_1 (미세각질) | 0.844 |
| value_2 (피지과다) | 0.591 |
| value_3 (모낭사이홍반) | 0.696 |
| value_4 (모낭홍반/농포) | 0.952 |
| value_5 (비듬) | 0.733 |
| value_6 (탈모) | 0.761 |

> value_4는 class_0(정상) 비율이 매우 높아 accuracy가 높게 나타나지만, 소수 클래스 recall은 낮습니다. 클래스 불균형 문제가 남아 있으며 향후 개선이 필요합니다.

---

## 데이터

- **출처**: [AI Hub 유형별 두피 이미지 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EB%91%90%ED%94%BC&aihubDataSe=data&dataSetSn=216)
- **구조**: 고해상도 두피 이미지 + 6개 증상 라벨(0~3등급) + 메타데이터
- **특징**: 0등급(정상) 비율이 높아 클래스 불균형 대응 전략이 필요

**전처리 파이프라인**

원본(meta/training/validation) 기준으로 단일 `master_index.csv`를 생성하고,  
학습 시 아래 augmentation을 적용했습니다.

- `224×224` resize
- `RandomResizedCrop`
- `HorizontalFlip`
- `ColorJitter`
- `Normalize`

---

## 저장소 구조

```
scalp-vision-agent/
├── src/
│   ├── api/          # FastAPI 엔드포인트
│   ├── analysis/     # 위험도 계산, 리포트 생성, snapshot 저장
│   ├── cnn/          # 모델 구조 및 학습 코드
│   ├── db/           # SQLAlchemy DB 설정
│   ├── io/           # 데이터셋 매칭 및 인덱싱
│   ├── config.py
│   ├── inference.py  # 서비스 추론 로직
│   └── schemas.py
├── scalp-admin/      # React + Vite 관리자 화면
├── notebooks/        # EDA 및 모델링 실험 (런타임 미사용)
├── results/          # 실험 결과 및 최종 체크포인트
├── requirements.txt
└── README.md
```

> `notebooks/`는 서비스 런타임에서 사용하지 않습니다. EDA, 모델 비교 실험, 결과 분석을 위한 참고 자료입니다.

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| Backend | FastAPI, SQLAlchemy, SQLite, Pydantic |
| Frontend | React, Vite |
| AI/ML | PyTorch, EfficientNet-B0, OpenAI API |
| Data | Jupyter Notebook, AI Hub dataset |

---

## 실행 방법

### 사전 준비

- Python 환경
- Node.js / npm
- OpenAI API Key
- AI Hub 데이터셋 다운로드
- 모델 체크포인트 파일

### 데이터 준비

```bash
python -m src.io.match_dataset
```

### 백엔드 실행

```bash
pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload
# http://127.0.0.1:8000
```

### 프론트엔드 실행

```bash
cd scalp-admin
npm install
npm run dev
# http://localhost:5173
```

### 환경 변수 (.env)

```env
OPENAI_API_KEY=your_api_key
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0.5
```

---

## 주요 API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/users` | 사용자 목록 |
| POST | `/users` | 사용자 등록 |
| POST | `/visits` | 방문 생성 |
| GET | `/visits/{visit_id}/full` | 방문 전체 조회 |
| POST | `/visits/{visit_id}/analyze-image` | 이미지 분석 |
| GET | `/visits/{visit_id}/report` | 리포트 조회 |

---

## 한계 및 주의사항

- 본 서비스는 **의료진의 판단을 보조**하기 위한 시스템입니다.
- 분석 결과는 **의학적 진단 또는 처방을 대체하지 않습니다.**

---

