[README.md](https://github.com/user-attachments/files/27084118/README.md)
# 🎓 학생 데이터 기반 고등학교 성적 하락 위험군 조기 예측 모델

> 1학년 성적·출결·통학 권역 데이터만으로 **3학년 시점의 성적 하락 위험군**을 2년 앞서 식별하는 머신러닝 파이프라인

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-orange?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 프로젝트 개요

고등학교 현장에서 교사가 수많은 학생의 정보를 종합적으로 판단하여 위기 학생을 조기에 발견하기란 현실적으로 어렵습니다. 이 프로젝트는 **1학년 입학 직후 수집 가능한 최소 데이터**를 활용하여 2년 후 결과를 미리 탐지하는 예측 모델을 구축하고, 그 의사결정 근거를 SHAP으로 해석하여 **교사의 전문 판단을 보조하는 조기 경보 시스템**으로 활용하는 것을 목표로 합니다.

### 핵심 설계 원칙

- **재현율(Recall) 우선** : 실제 위험 학생을 놓치는 것이 오탐보다 교육적으로 더 큰 손실
- **해석 가능성** : SHAP 분석으로 개별 학생별 위험 원인을 상담에 활용
- **모델 보조 원칙** : 예측 결과는 교사의 전문 판단을 *대체*하지 않고 *보조*하는 수단

---

## 📁 파일 구성

```
Student_Risk_Prediction/
│
├── ML파이프라인_위험군예측.ipynb     # Google Colab 실행 노트북 (셀 단위 구성)
├── ML파이프라인_코드.py              # 전체 파이프라인 완성 코드 (주석 포함)
├── final_student_data_for_ml.csv    # 분석 데이터 (비식별화, 257명)
└── README.md
```

---

## 📊 데이터 개요

| 항목 | 내용 |
|------|------|
| 대상 | 경기도 용인 지역 인문계 고등학교 졸업생 257명 (비식별화) |
| 타겟 변수 | `(3학년_평균등급 − 1학년_평균등급) ≥ 0.5` → 위험군(1) / 안정군(0) |
| 위험군 비율 | 80명 (31.1%) |
| 독립변수 | 1학년 평균등급, G1 출결 8개 항목, 통학 권역(Zone A~D) |

> **등급 척도 주의** : 등급 수치가 클수록 성취가 낮은 역방향 척도입니다.  
> 따라서 `3학년 등급 > 1학년 등급`은 성취 수준의 하락을 의미합니다.

---

## ⚙️ 파이프라인 구성 (6단계)

```
셀 02  환경 설정      패키지 설치 · 한글 폰트 · 임포트
셀 03  문제 정형화    데이터 로딩 · 타겟 정의 · 피처 선정
셀 04  EDA            분포 · 출결 · 통학권역 시각화 (Fig 1–2)
셀 05  전처리         Train/Test 분할 · 파이프라인 구성
셀 06  모델링         Grid Search + 5-Fold CV 하이퍼파라미터 탐색
셀 07  혼동행렬       Fig 3
셀 08  ROC · PR · 성능 비교   Fig 4–5
셀 09  피처 중요도    Fig 6
셀 10  SHAP 분석      Fig 7
셀 11  Ablation 실험  Fig 8
셀 12  임계값 분석 + 결과 저장   Fig 9
```

---

## 🤖 사용 모델

| 모델 | 선택 이유 | 주요 설정 |
|------|-----------|-----------|
| **KNN** | 거리 기반 직관적 해석 ("유사한 학생은 유사한 결과") | `n_neighbors=7`, `weights=distance`, StandardScaler 적용 |
| **Random Forest** | 변수 간 상호작용 포착, SHAP 해석 용이 | `max_depth=5`, `class_weight='balanced'` (불균형 보정) |

---

## 📈 주요 결과

### 모델 성능 비교 (테스트셋 기준)

| 지표 | KNN | Random Forest | 우위 |
|------|-----|---------------|------|
| 정확도 | 0.635 | 0.558 | KNN |
| **위험군 재현율** | **0.250** | **0.438** | **★ RF** |
| 위험군 F1 | 0.296 | 0.378 | RF |
| ROC-AUC | 0.425 | 0.513 | RF |
| AP (PR곡선) | 0.318 | 0.467 | RF |

> 조기 개입 목적에서 핵심 지표인 **위험군 재현율**에서 RF가 KNN보다 유의미하게 우수합니다.

### Ablation 실험 (1학년 평균등급 제거)

| 모델 | 등급 포함 재현율 | 등급 제외 재현율 | 해석 |
|------|----------------|----------------|------|
| KNN | 0.250 | 0.063 ↓ | 성적 변수에 강하게 의존 |
| RF  | 0.438 | 0.500 ↑ | 출결·통학만으로도 패턴 포착 가능 |

출결과 통학 데이터만으로도 RF는 위험군의 약 **50%를 탐지**합니다. 학업 슬럼프가 단순히 과거 성취의 연장이 아니라, 생활 패턴과 환경적 요인에 의해 독립적으로 유발될 수 있음을 시사합니다.

### 주요 예측 변수 (RF 피처 중요도 기준)

1. `1학년_평균등급` (0.234)
2. `G1_질병결석` (0.229)
3. `G1_질병지각` (0.168)
4. `G1_질병조퇴` (0.152)
5. `G1_미인정지각` (0.086)

---

## 🏫 교육적 활용 방안

### 조기 개입 전략

- **임계값 0.4 기준** : `predict_proba()[:, 1] ≥ 0.4`인 학생을 상담 우선 대상으로 분류
- **SHAP Waterfall** : 개별 학생의 위험 원인을 변수별로 분해 → 맞춤형 생활지도 근거 제공
- **출결 임계점** : G1 질병조퇴 **1~2회 초과 시점**부터 담임 면담 권장 (SHAP Dependence 분석 결과)

### 다중 지원 체계

```
예측 모델 (RF, 임계값 0.4)
       ↓
담임교사 1차 면담 및 관찰
       ↓
보건교사 · Wee 클래스 연계 (필요 시)
```

> ⚠️ **주의** : 본 모델의 ROC-AUC가 0.51 수준으로 낮아, **절대적 판정 도구로 사용하기에는 한계**가 있습니다. 반드시 교사의 전문 판단과 병행하여 활용하십시오.

---

## 🚀 실행 방법

### Google Colab (권장)

1. [Google Colab](https://colab.research.google.com) 접속
2. `파일 > 노트북 업로드` → `ML파이프라인_위험군예측.ipynb` 선택
3. 셀 순서대로 실행 (셀 2에서 패키지 자동 설치)
4. 데이터는 셀 3에서 GitHub URL로 자동 로딩

```python
# 셀 3 - 데이터 자동 로딩 (업로드 불필요)
url = "https://raw.githubusercontent.com/Jong-ho-chang/Student_Risk_Prediction/main/final_student_data_for_ml.csv"
df = pd.read_csv(url, encoding="utf-8-sig")
```

### 로컬 환경

```bash
git clone https://github.com/Jong-ho-chang/Student_Risk_Prediction.git
cd Student_Risk_Prediction
pip install numpy pandas matplotlib scikit-learn shap
python ML파이프라인_코드.py
```

---

## 📦 의존 패키지

```
numpy
pandas
matplotlib
scikit-learn
shap
```

> Colab 환경에서는 셀 2 실행 시 자동 설치됩니다.

---

## 🔬 향후 과제

- [ ] 2학년 중간 데이터 추가 시 성적 하락 발생 시점 정밀 특정 가능
- [ ] 상담 기록·행동 특성 등 정서적 행동 데이터 통합
- [ ] XGBoost, LightGBM, 로지스틱 회귀 등 추가 알고리즘 비교
- [ ] 학교 정보시스템과 연동한 실시간 조기 경보 대시보드 구현

---

## 📝 과제 정보

| 항목 | 내용 |
|------|------|
| 과목 | 교육문제해결 머신러닝 중간과제 |
| 전공 | 인공지능융합교육 |
| 데이터 | 경기도 용인 지역 인문계 고등학교 졸업생 257명 (비식별화) |

---

## 📄 라이선스

This project is licensed under the MIT License.  
데이터는 비식별화 처리되었으며 교육·연구 목적으로만 사용됩니다.
