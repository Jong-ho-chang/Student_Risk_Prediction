"""
================================================================================
학생 데이터 기반 고등학교 성적 하락 위험군 조기 예측 모델
--------------------------------------------------------------------------------
과제명  : 교육문제해결 머신러닝 중간과제 (8주차)
주  제  : 1학년 성적·출결·통학 권역 데이터를 활용한 3학년 성적 하락 위험군 분류
데이터  : 경기도 용인 지역 인문계 고등학교 졸업생 257명 비식별화 데이터
알고리즘: KNN (K-최근접 이웃), Random Forest (랜덤 포레스트)
핵심지표: 위험군 재현율(Recall) — 실제 위험 학생을 놓치지 않는 것을 최우선으로 설정
================================================================================

[파이프라인 6단계 구성]
  1단계 - 문제 정형화   : 타겟 변수 정의, 독립변수 선정, 성공 기준 설정
  2단계 - EDA           : 기술통계, 분포 시각화, 결측·이상치 진단
  3단계 - 전처리        : 인코딩, 스케일링, 불균형 처리, train/test 분할
  4단계 - 모델링        : Grid Search 기반 하이퍼파라미터 탐색
  5단계 - 평가          : 혼동행렬, ROC-AUC, PR곡선, 교차검증 결과
  6단계 - 해석·교육적 적용 : SHAP, 피처 중요도, 임계값 분석, Ablation 실험

[파일 구성]
  - 입력 : final_student_data_for_ml.csv   (전처리 완료 데이터)
  - 출력 : model_results/ 폴더 (시각화 이미지 + 결과 리포트 텍스트)
================================================================================
"""

# ── 표준 라이브러리 ──────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path

# ── 수치 연산 ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── 시각화 ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── sklearn: 전처리 ──────────────────────────────────────────────────────────
from sklearn.compose import ColumnTransformer          # 수치형·범주형 변수를 분리 처리하는 파이프라인 구성 요소
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # 범주형 인코딩 / 표준화

# ── sklearn: 모델 ────────────────────────────────────────────────────────────
from sklearn.neighbors import KNeighborsClassifier     # KNN 분류기
from sklearn.ensemble import RandomForestClassifier    # 랜덤 포레스트 분류기

# ── sklearn: 파이프라인 & 탐색 ───────────────────────────────────────────────
from sklearn.pipeline import Pipeline                  # 전처리 → 모델을 하나의 객체로 연결
from sklearn.model_selection import (
    GridSearchCV,          # 하이퍼파라미터 그리드 전탐색
    train_test_split,      # 훈련/테스트 분할
)

# ── sklearn: 평가 지표 ───────────────────────────────────────────────────────
from sklearn.metrics import (
    ConfusionMatrixDisplay,    # 혼동행렬 시각화
    accuracy_score,            # 정확도
    auc,                       # ROC-AUC 계산
    average_precision_score,   # Average Precision (PR 곡선 아래 넓이)
    classification_report,     # 정밀도·재현율·F1 종합 리포트
    confusion_matrix,          # 혼동행렬 원시 배열
    precision_recall_curve,    # PR 곡선 좌표
    recall_score,              # 재현율(위험군 탐지율)
    roc_curve,                 # ROC 곡선 좌표
    f1_score,                  # F1 점수
    precision_score,           # 정밀도
)

# ── SHAP: 모델 해석 ──────────────────────────────────────────────────────────
import shap  # SHapley Additive exPlanations — 개별 변수의 예측 기여도를 분해


# ============================================================
# [경로 설정]
# - ROOT   : 현재 스크립트 위치 (데이터 파일도 같은 폴더에 위치)
# - RESULT_DIR : 시각화 이미지·리포트 저장 폴더 (없으면 자동 생성)
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "final_student_data_for_ml.csv"
RESULT_DIR = ROOT / "model_results"
RESULT_DIR.mkdir(exist_ok=True)  # 폴더가 없으면 생성, 있으면 그대로 유지


# ============================================================
# [0단계] 환경 설정 함수들
# ============================================================

def set_korean_font() -> str:
    """
    [한글 폰트 자동 설정]
    matplotlib 기본 폰트는 한글을 지원하지 않아 그래프에 한글이 깨진다.
    시스템에 설치된 한글 폰트를 우선순위에 따라 탐색하여 자동 지정한다.

    우선순위:
      1. AppleGothic   (macOS 기본)
      2. Malgun Gothic  (Windows 기본)
      3. NanumGothic   (Linux 설치 필요: apt install fonts-nanum)
      4. 위 모두 없으면 'DejaVu Sans' (한글 깨짐 감수)

    axes.unicode_minus=False: matplotlib의 마이너스 기호('-')가 □로 깨지는
    버그를 방지한다. 이 옵션은 폰트와 무관하게 항상 설정해야 한다.
    """
    preferred = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Arial Unicode MS"]
    installed = {f.name for f in fm.fontManager.ttflist}
    selected = next((n for n in preferred if n in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def load_data(path: Path) -> pd.DataFrame:
    """
    [데이터 로딩]
    CSV 파일을 읽어 DataFrame으로 반환한다.

    encoding="utf-8-sig":
      한글 윈도우에서 Excel로 저장한 CSV는 UTF-8에 BOM(Byte Order Mark, \xef\xbb\xbf)이
      붙어 있어 일반 utf-8로 읽으면 첫 컬럼명이 깨진다.
      utf-8-sig는 BOM을 자동으로 제거하므로 윈도우·맥 모두 안전하다.
    """
    return pd.read_csv(path, encoding="utf-8-sig")


# ============================================================
# [1단계] 문제 정형화: 타겟 변수 정의 & 피처 선택
# ============================================================

def build_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    [타겟 변수 정의]
      is_risk = (3학년_평균등급 - 1학년_평균등급) >= 0.5

      - 성적 등급은 수치가 클수록 성취가 낮은 역방향 척도이다.
        따라서 등급 수치의 증가(+)는 성취 수준의 하락을 의미한다.
      - 0.5 이상 차이를 '의미 있는 하락'으로 정의한 것은, 등급 체계의
        자연 변동(측정 오차)을 넘어서는 실질적 슬럼프를 포착하기 위함이다.
      - 결과: 위험군(1) = 80명(31.1%), 안정군(0) = 177명(68.9%)

    [피처 선택 근거]
      - 통학_권역: 통학 거리·환경이 학업 지속성에 영향을 줄 수 있다는 교육학적 가설
      - 1학년_평균등급: 초기 성취 수준이 이후 하락 가능성의 기준점이 됨
        ※ 이 변수는 타겟 산출식에도 등장(구조적 중복). 6.2절 Ablation으로 별도 검증.
      - G1_* 출결 변수: 결석·지각·조퇴의 누적이 학업 이탈의 선행 신호라는 연구 근거

    [제외한 변수]
      - G2_, G3_ 출결: 2·3학년 데이터는 1학년 말 시점에 아직 존재하지 않으므로
        '조기 예측' 목적에 부합하지 않아 의도적으로 제외
      - 반, 성명: 개인 식별 정보로 예측 변수로 부적절
    """
    df = df.copy()

    # 타겟 생성: 등급 차이 ≥ 0.5이면 위험군(1), 아니면 안정군(0)
    df["is_risk"] = ((df["3학년_평균등급"] - df["1학년_평균등급"]) >= 0.5).astype(int)

    # G1_으로 시작하는 모든 출결 컬럼 자동 선택 (추후 변수 추가 시 유지보수 용이)
    g1_cols = [c for c in df.columns if c.startswith("G1_")]
    feature_cols = ["통학_권역", "1학년_평균등급"] + g1_cols

    X = df[feature_cols].copy()
    y = df["is_risk"].copy()
    return X, y


# ============================================================
# [2단계] EDA: 탐색적 데이터 분석 시각화
# ============================================================

def save_eda_plots(df: pd.DataFrame, save_dir: Path) -> None:
    """
    [EDA 시각화]
    데이터의 구조와 위험군 분포를 시각적으로 탐색한다.

    생성 파일:
      eda_target_distribution.png  : 타겟 클래스 분포 + 성적 분포
      eda_attendance_zone.png      : 출결 항목별 평균 + 통학 권역별 위험군 비율
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # [그림 1-左] 타겟 클래스 분포 막대그래프
    risk_counts = df["is_risk"].value_counts().sort_index()
    axes[0].bar(["안정군(0)", "위험군(1)"], risk_counts.values,
                color=["#1D3557", "#E76F51"], width=0.5, edgecolor="white")
    for i, val in enumerate(risk_counts.values):
        axes[0].text(i, val + 2, f"{val}명 ({val/len(df)*100:.1f}%)",
                     ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("타겟 클래스 분포", fontweight="bold")
    axes[0].set_ylabel("학생 수")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # [그림 1-右] 위험군·안정군 별 1학년 평균등급 히스토그램
    for label, color, name in [(0, "#1D3557", "안정군"), (1, "#E76F51", "위험군")]:
        subset = df[df["is_risk"] == label]["1학년_평균등급"]
        axes[1].hist(subset, bins=14, alpha=0.65, color=color,
                     label=f"{name} (n={len(subset)}, μ={subset.mean():.2f})",
                     edgecolor="white")
    axes[1].set_title("1학년 평균등급 분포 (위험군 vs 안정군)", fontweight="bold")
    axes[1].set_xlabel("1학년 평균등급 (낮을수록 우수)")
    axes[1].set_ylabel("학생 수")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "eda_target_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # [그림 2] 출결 항목별 평균 비교 + 통학 권역별 위험군 비율
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    att_cols = ["G1_미인정결석", "G1_미인정지각", "G1_미인정조퇴",
                "G1_질병결석", "G1_질병지각", "G1_질병조퇴"]
    att_labels = ["미인정결석", "미인정지각", "미인정조퇴", "질병결석", "질병지각", "질병조퇴"]

    # 위험군·안정군 별 출결 항목 평균 막대 비교
    means0 = df[df["is_risk"] == 0][att_cols].mean().values
    means1 = df[df["is_risk"] == 1][att_cols].mean().values
    x = np.arange(len(att_cols))
    axes[0].bar(x - 0.18, means0, 0.36, label="안정군", color="#1D3557", alpha=0.85)
    axes[0].bar(x + 0.18, means1, 0.36, label="위험군", color="#E76F51", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(att_labels, rotation=30, ha="right")
    axes[0].set_title("출결 항목별 평균 비교", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # 통학 권역별 위험군 비율 (소표본 권역은 시각화 후 해석 주의)
    zone_risk = df.groupby("통학_권역")["is_risk"].agg(["sum", "count"])
    zone_risk["ratio"] = zone_risk["sum"] / zone_risk["count"]
    bars = axes[1].bar(range(len(zone_risk)), zone_risk["ratio"].values,
                       color="#F4A261", edgecolor="white")
    for bar, (_, row) in zip(bars, zone_risk.iterrows()):
        # 소표본(n<10) 권역은 '*' 표시로 해석 주의를 환기
        note = f"{row['ratio']:.1%}" + ("*" if row["count"] < 10 else "")
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, note, ha="center", fontweight="bold")
    axes[1].set_xticks(range(len(zone_risk)))
    axes[1].set_xticklabels(
        [f"{z}\n(n={int(r['count'])})" for z, r in zone_risk.iterrows()],
        fontsize=8)
    axes[1].axhline(df["is_risk"].mean(), color="gray", linestyle="--",
                    label=f"전체 평균 {df['is_risk'].mean():.1%}")
    axes[1].set_title("통학 권역별 위험군 비율\n(*: 소표본, 해석 주의)", fontweight="bold")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "eda_attendance_zone.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[EDA] 시각화 저장 완료")


# ============================================================
# [3단계] 전처리: 파이프라인 구성 함수 (KNN / RF 각각 별도 구성)
# ============================================================

def build_pipeline(
    model_type: str,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> Pipeline:
    """
    [파이프라인 구성 원칙]
    전처리와 모델을 하나의 Pipeline 객체로 연결하면:
      1) train/test leakage 방지: fit은 train에서만, transform은 test에도 동일 적용
      2) Grid Search 시 전처리까지 포함하여 자동으로 교차검증 적용
      3) 코드 재사용성 및 유지보수성 향상

    [KNN 파이프라인]
      - 범주형: OneHotEncoder (통학_권역 → 더미 변수)
      - 수치형: StandardScaler (평균 0, 표준편차 1로 표준화)
        → KNN은 거리(유클리드/맨해튼) 기반이므로 스케일이 다른 변수가
          혼재하면 큰 스케일 변수가 거리를 지배하는 문제 발생.
          표준화로 모든 변수가 동등한 영향을 갖도록 조정.

    [RF 파이프라인]
      - 범주형: OneHotEncoder (동일)
      - 수치형: passthrough (변환 없이 그대로 사용)
        → 결정트리는 분할점(threshold)을 학습하므로 스케일에 무관.
          단, 파이프라인 구조 일관성을 위해 ColumnTransformer에 포함.
      - class_weight='balanced': 소수 클래스(위험군, 31%)에
          177/80 ≈ 2.2배의 가중치를 자동 부여.
          불균형 데이터에서 다수 클래스(안정군)에 치우친 학습을 보정.

    handle_unknown='ignore': 테스트 데이터에 훈련 시 본 적 없는
      범주가 등장할 경우 오류 대신 0 벡터로 처리. 소규모 데이터에서
      train/test 분할 후 특정 범주가 test에만 나타날 수 있는 상황 대비.
    """
    # 범주형 전처리기
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # 수치형 전처리기: KNN은 표준화, RF는 그대로
    num_transformer = StandardScaler() if model_type == "knn" else "passthrough"

    # ColumnTransformer: 범주형/수치형 컬럼에 다른 변환기를 동시에 적용
    preprocessor = ColumnTransformer(transformers=[
        ("cat", cat_transformer, categorical_cols),
        ("num", num_transformer, numeric_cols),
    ])

    # 모델 선택
    if model_type == "knn":
        model = KNeighborsClassifier()
    else:  # "rf"
        model = RandomForestClassifier(
            random_state=42,         # 결과 재현성 보장 (같은 시드 → 같은 트리 생성)
            class_weight="balanced", # 소수 클래스 가중치 자동 조정
        )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ============================================================
# [4단계] 모델링: Grid Search 기반 하이퍼파라미터 탐색
# ============================================================

def fit_with_gridsearch(
    pipeline: Pipeline,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "recall",
    cv: int = 5,
) -> GridSearchCV:
    """
    [Grid Search + Cross Validation 학습]

    [scoring='recall' 선택 이유]
      조기 개입 목적에서 실제 위험 학생을 놓치는 것(FN: False Negative)이
      위험하지 않은 학생을 위험으로 잘못 분류하는 것(FP)보다 교육적으로
      더 큰 손실이다. 따라서 재현율(Recall = TP / (TP + FN))을 기준으로
      가장 좋은 파라미터 조합을 선택한다.

    [5-Fold Cross Validation]
      훈련 데이터(205명)를 5개 폴드로 나눠 4개로 학습·1개로 검증을 반복.
      소규모 데이터에서 단일 validation split보다 안정적인 성능 추정 제공.
      각 파라미터 조합의 5개 재현율을 평균 내어 최적 파라미터를 선택한다.

    n_jobs=-1: 가용한 CPU 코어를 모두 사용하여 병렬 탐색 수행.
    """
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,      # 최적화 기준: 위험군 재현율
        cv=cv,                # 5-Fold 교차검증
        n_jobs=-1,            # 병렬 처리 (CPU 코어 전체 활용)
        verbose=0,            # 탐색 과정 출력 생략 (1 이상이면 출력)
        return_train_score=False,  # 훈련 점수 불필요 (과적합 진단 시 True로 변경)
    )
    gs.fit(X_train, y_train)
    return gs


def get_param_grids() -> tuple[dict, dict]:
    """
    [하이퍼파라미터 탐색 공간 정의]

    KNN 파라미터:
      n_neighbors: 참조할 이웃 수. 너무 작으면 과적합, 너무 크면 과소적합.
                   홀수를 포함해 2개 클래스 동점을 줄이는 것이 관행.
      weights: 'uniform'=거리 무관 동등 투표, 'distance'=가까울수록 높은 가중치.
      p: 거리 계산 방식. p=1(맨해튼), p=2(유클리드).

    RF 파라미터:
      n_estimators: 트리 수. 많을수록 안정적이나 학습 시간 증가. 100이면 충분한 경우가 많음.
      max_depth: 트리 최대 깊이. None이면 완전 성장(과적합 위험). 5~10이 소규모 데이터에 적합.
      min_samples_split: 노드를 분할하는 최소 샘플 수. 클수록 트리가 단순해짐.
      min_samples_leaf: 말단 노드의 최소 샘플 수. 클수록 일반화 성능 향상.
    """
    knn_params = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }
    rf_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    return knn_params, rf_params


# ============================================================
# [5단계] 평가: 시각화 함수들
# ============================================================

def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """
    [혼동행렬 시각화]
    혼동행렬의 4개 셀 의미:
      TN (좌상): 안정군을 안정군으로 정확히 예측 → 좋음
      FP (우상): 안정군을 위험군으로 잘못 예측 → 과잉 탐지 (상담 자원 낭비 가능)
      FN (좌하): 위험군을 안정군으로 잘못 예측 → 위험! 실제 위험 학생 누락
      TP (우하): 위험군을 위험군으로 정확히 예측 → 최우선 최대화 목표

    재현율(Recall) = TP / (TP + FN): FN을 줄이는 것이 핵심
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["안정군(0)", "위험군(1)"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_roc_and_pr_curves(
    y_true: pd.Series,
    knn_prob: np.ndarray,
    rf_prob: np.ndarray,
    save_path: Path,
) -> tuple[float, float, float, float]:
    """
    [ROC 곡선 & PR 곡선 비교 저장]

    ROC 곡선:
      - x축(FPR)과 y축(TPR)의 트레이드오프를 임계값 변화에 따라 그린 곡선.
      - AUC(곡선 아래 넓이)가 1에 가까울수록 우수. 0.5는 무작위 기준.
      - 단점: 클래스 불균형 시 낙관적 수치를 보일 수 있음.

    PR(정밀도-재현율) 곡선:
      - 불균형 클래스에서 ROC보다 실제 경보 품질을 잘 반영.
      - Average Precision(AP) = PR 곡선 아래 넓이.
      - 기준선(baseline): 무작위 분류 시 precision ≈ 위험군 비율(~0.31).
    """
    # ROC
    fpr_knn, tpr_knn, _ = roc_curve(y_true, knn_prob)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_prob)
    auc_knn = auc(fpr_knn, tpr_knn)
    auc_rf = auc(fpr_rf, tpr_rf)

    # PR
    p_knn, r_knn, _ = precision_recall_curve(y_true, knn_prob)
    p_rf, r_rf, _ = precision_recall_curve(y_true, rf_prob)
    ap_knn = average_precision_score(y_true, knn_prob)
    ap_rf = average_precision_score(y_true, rf_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC 곡선
    axes[0].plot(fpr_knn, tpr_knn, color="#457B9D", lw=2.5,
                 label=f"KNN (AUC={auc_knn:.3f})")
    axes[0].plot(fpr_rf, tpr_rf, color="#2A9D8F", lw=2.5,
                 label=f"Random Forest (AUC={auc_rf:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="무작위 기준선")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (재현율)")
    axes[0].set_title("ROC 곡선 비교")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # PR 곡선
    baseline = y_true.mean()  # 무작위 분류 시 precision의 기대값
    axes[1].plot(r_knn, p_knn, color="#457B9D", lw=2.5,
                 label=f"KNN (AP={ap_knn:.3f})")
    axes[1].plot(r_rf, p_rf, color="#2A9D8F", lw=2.5,
                 label=f"Random Forest (AP={ap_rf:.3f})")
    axes[1].axhline(baseline, color="gray", linestyle="--",
                    label=f"기준선 ({baseline:.2f})")
    axes[1].set_xlabel("Recall (재현율)")
    axes[1].set_ylabel("Precision (정밀도)")
    axes[1].set_title("정밀도-재현율 곡선 비교")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)

    return auc_knn, auc_rf, ap_knn, ap_rf


def save_performance_bar(
    knn_acc: float, knn_recall: float, knn_prec: float, knn_f1: float,
    rf_acc: float,  rf_recall: float,  rf_prec: float,  rf_f1: float,
    auc_knn: float, auc_rf: float,
    ap_knn: float,  ap_rf: float,
    save_path: Path,
) -> None:
    """
    [성능 지표 종합 비교 막대그래프]
    6개 지표를 한 화면에 비교하여 모델 선택의 근거를 시각적으로 제시.
    조기 개입 목적에서 재현율(Recall) 열을 핵심 지표로 강조 표시.
    """
    metrics = ["정확도\n(Accuracy)", "위험군\n재현율", "위험군\n정밀도",
               "위험군\nF1", "ROC-AUC", "AP\n(PR곡선)"]
    knn_vals = [knn_acc, knn_recall, knn_prec, knn_f1, auc_knn, ap_knn]
    rf_vals  = [rf_acc,  rf_recall,  rf_prec,  rf_f1,  auc_rf,  ap_rf]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - 0.18, knn_vals, 0.36, label="KNN",
                color="#457B9D", alpha=0.88)
    b2 = ax.bar(x + 0.18, rf_vals, 0.36, label="Random Forest",
                color="#2A9D8F", alpha=0.88)
    for bars, vals in [(b1, knn_vals), (b2, rf_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 재현율 열 강조 (조기 개입 핵심 지표)
    ax.axvspan(0.5, 1.5, alpha=0.06, color="orange")
    ax.text(1, 1.07, "★ 핵심 지표 (조기개입 관점)",
            ha="center", fontsize=9, color="darkorange", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("모델 성능 지표 종합 비교 — KNN vs Random Forest", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# ============================================================
# [6단계-A] 해석: 피처 중요도 (Random Forest)
# ============================================================

def save_feature_importance(
    rf_pipeline: Pipeline,
    top_n: int,
    save_path: Path,
) -> pd.DataFrame:
    """
    [RF 피처 중요도]
    Random Forest는 각 트리의 분할에서 불순도(Gini) 감소량을 변수별로 합산하여
    중요도를 계산한다. 값이 클수록 예측에 더 많이 기여한 변수이다.

    [주의]
      - 피처 중요도는 '방향'이 없다(양·음 구분 없음). SHAP 분석과 병행해야 함.
      - 높은 상관관계가 있는 변수들은 중요도가 분산되어 과소평가될 수 있음.
      - 해석: "이 변수가 없으면 예측이 얼마나 불안정해지는가"의 간접 측정.

    preprocessor.get_feature_names_out():
      ColumnTransformer 내부에서 변환 후 생성된 실제 컬럼명을 반환.
      OneHotEncoding으로 분해된 더미 변수명(예: cat__통학_권역_Zone A)도 포함.
    """
    preprocessor: ColumnTransformer = rf_pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = rf_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    # 중요도 DataFrame 생성 및 내림차순 정렬
    imp_df = (pd.DataFrame({"feature": feature_names, "importance": importances})
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))

    # 상위 N개만 시각화
    top_df = imp_df.head(top_n)
    colors = (["#E76F51"] * 3  # 1~3위: 강조
              + ["#2A9D8F"] * 4  # 4~7위: 중간
              + ["#B0C4DE"] * (top_n - 7))  # 나머지: 연한 색

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_df["feature"][::-1],
                   top_df["importance"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    for bar, val in zip(bars, top_df["importance"][::-1]):
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlabel("중요도 (Feature Importance)")
    ax.set_title(f"Random Forest 피처 중요도 (상위 {top_n}개)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)

    return imp_df


# ============================================================
# [6단계-B] 해석: SHAP 분석
# ============================================================

def prepare_shap(
    rf_pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> tuple[shap.TreeExplainer, np.ndarray, pd.DataFrame, float]:
    """
    [SHAP 준비: 전처리 후 데이터로 TreeExplainer 적용]

    SHAP(SHapley Additive exPlanations) 원리:
      게임 이론의 Shapley 값에서 착안. 각 변수가 예측에 기여한 양을
      "이 변수가 없었다면 예측이 얼마나 달라졌는가"로 측정.
      모든 변수의 SHAP 값 합 + base_value = 최종 예측 확률.

    [위험군 클래스(1) 기준 SHAP 추출]
      sklearn RF 이진분류에서 shap_values는 (n_samples, n_features, 2) 텐서로
      반환되는 경우가 많다(클래스 0, 1에 대한 값 각각). 위험군(클래스 1) 기준
      해석을 위해 마지막 차원의 인덱스 1을 슬라이싱한다.
    """
    preprocessor: ColumnTransformer = rf_pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = rf_pipeline.named_steps["model"]

    # 테스트 데이터 전처리 변환 (fit은 train에서 이미 완료)
    X_t = preprocessor.transform(X_test)
    if hasattr(X_t, "toarray"):  # 희소행렬(sparse matrix)이면 밀집 배열로 변환
        X_t = X_t.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_t, columns=feature_names, index=X_test.index)

    # TreeExplainer: tree 기반 모델에 최적화된 정확·빠른 SHAP 계산
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    sv = np.asarray(shap_values)

    # 클래스 1(위험군) SHAP 값 추출
    if sv.ndim == 3 and sv.shape[-1] == 2:
        class1_vals = sv[:, :, 1]
    elif isinstance(shap_values, list):
        class1_vals = np.asarray(shap_values[1])
    else:
        class1_vals = sv

    # base_value: 아무 변수 정보 없을 때의 예측 기댓값 (위험군 사전 확률에 해당)
    ev = explainer.expected_value
    base_value = float(ev[1]) if (hasattr(ev, "__len__") and len(ev) > 1) else float(ev)

    return explainer, class1_vals, X_df, base_value


def save_shap_summary(
    class1_vals: np.ndarray,
    X_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    [SHAP Summary Plot]
    - 각 변수의 SHAP 값 분포(점 하나 = 학생 한 명)를 한 화면에 표시.
    - 세로축: 변수(평균 |SHAP|로 내림차순 정렬)
    - 가로축: SHAP 값 (+면 위험 기여, -면 안전 기여)
    - 색상: 해당 변수 값의 크기(빨강=높은 값, 파랑=낮은 값)
    → "어떤 변수의 어떤 값이 위험 예측에 기여하는가"를 한눈에 파악.
    """
    plt.figure(figsize=(10, 6))
    shap.summary_plot(class1_vals, X_df, show=False)
    plt.title("SHAP Summary Plot (Random Forest, 위험군=1 기준)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_shap_dependence(
    class1_vals: np.ndarray,
    X_df: pd.DataFrame,
    feature_substring: str,
    title: str,
    save_path: Path,
) -> str | None:
    """
    [SHAP Dependence Plot]
    단일 변수의 값이 변함에 따라 해당 변수의 SHAP 기여도가 어떻게 변하는지를
    산점도로 보여준다. 비선형 관계나 임계점(threshold)을 발견하는 데 유용.

    - 가로축: 해당 변수의 실제 값
    - 세로축: 해당 변수의 SHAP 값 (위험 기여도)
    - 빨간 선: 각 값에 대한 평균 SHAP → 전반적인 추세 파악
    - 0선(회색 점선): 이 위는 위험 기여, 아래는 안전 기여

    임계점 해석:
      가로축의 특정 구간에서 세로축이 급격히 변하는 지점 →
      그 구간을 교육 현장의 '주의 모니터링 구간'으로 설정할 수 있음.
      단, 소표본 데이터에서는 과잉 해석을 피하고 추세 위주로 읽어야 함.
    """
    # feature_substring을 포함하는 컬럼명 탐색
    cols = [c for c in X_df.columns if feature_substring in c]
    if not cols:
        return None  # 해당 변수가 없으면 건너뜀
    col = cols[0]
    col_idx = list(X_df.columns).index(col)

    x_vals = X_df[col].to_numpy(dtype=float)
    y_shap = class1_vals[:, col_idx].astype(float)

    fig, ax = plt.subplots(figsize=(7, 5))
    # 점 색상: 실제 위험군(빨강) vs 안정군(파랑)
    ax.scatter(x_vals, y_shap, alpha=0.55, s=40, edgecolors="none",
               label="학생별 SHAP")

    # 이산 값이 20개 이하이면 각 값에 대한 평균 SHAP을 꺾은선으로 표시
    uniq = np.unique(x_vals[~np.isnan(x_vals)])
    if uniq.size <= 20:
        means = [float(np.mean(y_shap[x_vals == u])) for u in uniq]
        ax.plot(uniq, means, "o-", color="crimson", lw=2, ms=8,
                label="값별 평균 SHAP")

    ax.axhline(0, color="gray", linestyle="--", lw=1, alpha=0.7)
    ax.set_xlabel(col.replace("num__", "").replace("cat__", ""))
    ax.set_ylabel("SHAP 값 (위험 기여, +: 위험 ↑, -: 위험 ↓)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return col


def save_shap_waterfall(
    X_df: pd.DataFrame,
    y_test: pd.Series,
    rf_pred: np.ndarray,
    rf_prob: np.ndarray,
    class1_vals: np.ndarray,
    base_value: float,
    save_path: Path,
    rng: np.random.Generator,
) -> str:
    """
    [SHAP Waterfall Plot: 개별 학생 해석]
    모델이 '위험군'으로 예측한 학생 1명을 선택하여,
    각 변수가 위험 예측에 얼마나 기여했는지를 폭포수 그래프로 시각화.

    Waterfall 구조:
      - base_value(시작점): 변수 정보가 없을 때의 기댓값
      - 각 막대: 해당 변수가 base에서 예측을 얼마나 올리거나(+) 내렸는지(-)
      - 최종값 = base_value + 모든 SHAP 값의 합

    교육적 활용:
      상담 시 "이 학생에게는 어떤 요인이 위험 신호였는가"를 근거로
      맞춤형 생활지도 전략을 수립하는 데 활용 가능.

    후보 선정 로직:
      1) RF가 위험군(1)으로 예측한 학생 중 무작위 1명 선택
      2) 없으면 위험 확률이 가장 높은 학생으로 대체
    """
    pos_idx = np.flatnonzero(rf_pred == 1)
    if len(pos_idx) == 0:
        pos_idx = np.array([int(np.argmax(rf_prob))])
    pick = int(rng.choice(pos_idx))

    exp = shap.Explanation(
        values=class1_vals[pick],
        base_values=base_value,
        data=X_df.iloc[pick].values,
        feature_names=list(X_df.columns),
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(exp, max_display=20, show=False)
    plt.title(
        f"SHAP Waterfall — RF 위험군 예측 학생 (테스트 인덱스={pick}, "
        f"실제={int(y_test.iloc[pick])}, 위험확률={rf_prob[pick]:.3f})"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

    return (f"선택 인덱스(iloc)={pick}, RF예측={int(rf_pred[pick])}, "
            f"실제라벨={int(y_test.iloc[pick])}, 위험확률={rf_prob[pick]:.4f}")


# ============================================================
# [6단계-C] Ablation 실험: 1학년 성적 변수 제거 효과
# ============================================================

def run_ablation_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    knn_acc_full: float,
    knn_recall_full: float,
    rf_acc_full: float,
    rf_recall_full: float,
    save_path: Path,
) -> tuple[float, float, float, float]:
    """
    [Ablation Study: 1학년 평균등급 제거 실험]

    목적:
      모델이 '1학년 성적이 낮은 학생이 3학년에도 낮다'는 단순 편향에
      의존하는 것인지, 아니면 출결·통학 패턴에서 독립적인 위험 신호를
      실제로 학습하는지를 검증한다.

    방법:
      - 타겟(is_risk) 정의는 동일 유지 (3학년-1학년 차이 기반)
      - 예측 입력(X)에서만 '1학년_평균등급' 컬럼을 제거
      - 동일 train/test 분할, 동일 하이퍼파라미터 그리드로 재학습
      → 성능 변화를 통해 해당 변수의 독립적 기여를 추정

    해석 포인트:
      - KNN: 성적 제거 시 재현율이 크게 하락 → 거리 계산에 성적 의존도 높음
      - RF: 성적 제거 시 재현율이 유지 또는 상승 → 출결·통학만으로도 패턴 포착 가능
    """
    cat_cols = ["통학_권역"]
    # '1학년_평균등급'만 제거한 수치형 변수 목록
    num_ab = [c for c in X_train.columns
              if c not in cat_cols and c != "1학년_평균등급"]

    X_train_ab = X_train.drop(columns=["1학년_평균등급"])
    X_test_ab  = X_test.drop(columns=["1학년_평균등급"])

    knn_params, rf_params = get_param_grids()

    # KNN Ablation
    knn_ab = fit_with_gridsearch(
        build_pipeline("knn", cat_cols, num_ab), knn_params, X_train_ab, y_train
    )
    knn_pred_ab  = knn_ab.best_estimator_.predict(X_test_ab)
    knn_acc_ab   = accuracy_score(y_test, knn_pred_ab)
    knn_rec_ab   = recall_score(y_test, knn_pred_ab, pos_label=1)

    # RF Ablation
    rf_ab = fit_with_gridsearch(
        build_pipeline("rf", cat_cols, num_ab), rf_params, X_train_ab, y_train
    )
    rf_pred_ab   = rf_ab.best_estimator_.predict(X_test_ab)
    rf_acc_ab    = accuracy_score(y_test, rf_pred_ab)
    rf_rec_ab    = recall_score(y_test, rf_pred_ab, pos_label=1)

    # 비교 시각화
    labels = ["KNN\n정확도", "KNN\n재현율(위험군)", "RF\n정확도", "RF\n재현율(위험군)"]
    full_v = [knn_acc_full, knn_recall_full, rf_acc_full, rf_recall_full]
    ab_v   = [knn_acc_ab,   knn_rec_ab,      rf_acc_ab,   rf_rec_ab]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, full_v, 0.36, label="1학년 평균등급 포함",
           color="#457B9D", alpha=0.88)
    ax.bar(x + 0.18, ab_v, 0.36, label="1학년 평균등급 제외 (출결·통학만)",
           color="#F4A261", alpha=0.88)

    # 변화량(Δ) 표기: 상승은 초록, 하락은 빨강
    for i, (fv, av) in enumerate(zip(full_v, ab_v)):
        delta = av - fv
        color = "#2A9D8F" if delta > 0 else "#E76F51"
        ax.annotate(f"Δ{delta:+.3f}",
                    xy=(i + 0.18 + 0.05, max(fv, av) + 0.04),
                    ha="left", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Ablation Study: 1학년 평균등급 제거 전후 성능 변화", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return knn_acc_ab, knn_rec_ab, rf_acc_ab, rf_rec_ab


# ============================================================
# [6단계-D] 임계값(Threshold) 분석
# ============================================================

def save_threshold_analysis(
    y_true: pd.Series,
    rf_prob: np.ndarray,
    save_path: Path,
) -> None:
    """
    [분류 임계값에 따른 성능 변화 분석]

    배경:
      sklearn의 predict()는 기본적으로 임계값 0.5를 사용한다.
      즉 predict_proba()[:, 1] >= 0.5이면 위험군(1)으로 분류.

      그러나 조기 개입 목적에서 재현율을 높이려면 임계값을 낮춰야 한다.
      → 임계값 ↓: 재현율 ↑, 정밀도 ↓ (더 많이 탐지하되, 오탐도 증가)
      → 임계값 ↑: 재현율 ↓, 정밀도 ↑ (탐지 수 줄지만, 탐지 정확도 증가)

    교육적 해석:
      - 상담 자원이 충분하다면 낮은 임계값(0.3~0.4) → 더 많은 학생을 면담
      - 자원이 제한적이라면 높은 임계값(0.5~0.6) → 확실한 위험군에 집중
      - F1이 최대가 되는 임계값을 기본 기준점으로 제시하되,
        최종 결정은 학교 상황과 교사 판단에 맡기는 것이 타당
    """
    thresholds = np.arange(0.20, 0.86, 0.05)
    recalls, precisions, f1s = [], [], []

    for t in thresholds:
        pred_t = (rf_prob >= t).astype(int)
        recalls.append(recall_score(y_true, pred_t, pos_label=1, zero_division=0))
        precisions.append(precision_score(y_true, pred_t, pos_label=1, zero_division=0))
        f1s.append(f1_score(y_true, pred_t, pos_label=1, zero_division=0))

    best_f1_idx = int(np.argmax(f1s))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, recalls, "o-", color="#E76F51", lw=2.5, ms=7,
            label="재현율 (위험군 탐지율)")
    ax.plot(thresholds, precisions, "s-", color="#457B9D", lw=2.5, ms=7,
            label="정밀도 (탐지 정확도)")
    ax.plot(thresholds, f1s, "^-", color="#2A9D8F", lw=2.5, ms=7,
            label="F1-Score")

    # F1 최적 임계값 표시
    ax.axvline(thresholds[best_f1_idx], color="gray", linestyle="--", lw=1.5)
    ax.text(thresholds[best_f1_idx] + 0.01, 0.75,
            f"F1 최대 임계값\n= {thresholds[best_f1_idx]:.2f}",
            fontsize=10, color="gray")

    # 보고서 제안 임계값(0.4) 표시
    ax.axvline(0.4, color="#F4A261", linestyle=":", lw=2)
    ax.text(0.41, 0.42, "제안 임계값 0.4",
            fontsize=9.5, color="#F4A261", fontweight="bold")

    ax.set_xlabel("분류 임계값 (Threshold)", fontsize=11)
    ax.set_ylabel("점수", fontsize=11)
    ax.set_title("RF 분류 임계값에 따른 재현율·정밀도·F1 변화\n"
                 "(조기 개입 목적: 재현율 우선 → 낮은 임계값 설정)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# [메인 함수]: 전체 파이프라인 실행
# ============================================================

def main() -> None:
    # ── 환경 설정 ──────────────────────────────────────────────────────────
    selected_font = set_korean_font()
    print(f"[설정] 한글 폰트: {selected_font}")
    print(f"[설정] 결과 저장 폴더: {RESULT_DIR}")

    # ── 1단계: 데이터 로딩 & 문제 정형화 ──────────────────────────────────
    print("\n[1단계] 데이터 로딩 및 문제 정형화")
    df = load_data(DATA_PATH)
    X, y = build_features_and_target(df)

    print(f"  - 전체 샘플: {len(df)}명")
    print(f"  - 위험군(1): {y.sum()}명 ({y.mean()*100:.1f}%)")
    print(f"  - 안정군(0): {(1-y).sum()}명 ({(1-y).mean()*100:.1f}%)")
    print(f"  - 독립변수 수: {X.shape[1]}개")

    categorical_cols = ["통학_권역"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # ── 2단계: EDA ─────────────────────────────────────────────────────────
    print("\n[2단계] EDA 시각화")
    save_eda_plots(df, RESULT_DIR)

    # ── 3단계: 전처리 (train/test 분할) ────────────────────────────────────
    print("\n[3단계] 전처리 — train/test 분할 (stratify 적용)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 테스트 20%(약 52명), 훈련 80%(약 205명)
        random_state=42,    # 재현성 고정
        stratify=y,         # 분할 후에도 위험군 비율 유지 (~31%)
    )
    print(f"  - 훈련: {len(X_train)}명 | 테스트: {len(X_test)}명")
    print(f"  - 훈련 위험군 비율: {y_train.mean():.3f}")
    print(f"  - 테스트 위험군 비율: {y_test.mean():.3f}")

    # ── 4단계: 모델링 (Grid Search) ────────────────────────────────────────
    print("\n[4단계] Grid Search 기반 하이퍼파라미터 탐색 (5-Fold CV)")
    knn_params, rf_params = get_param_grids()

    print("  - KNN Grid Search 실행 중...")
    knn_gs = fit_with_gridsearch(
        build_pipeline("knn", categorical_cols, numeric_cols),
        knn_params, X_train, y_train
    )
    print(f"  - KNN 최적 파라미터: {knn_gs.best_params_}")
    print(f"  - KNN CV 최적 재현율: {knn_gs.best_score_:.4f}")

    print("  - RF Grid Search 실행 중...")
    rf_gs = fit_with_gridsearch(
        build_pipeline("rf", categorical_cols, numeric_cols),
        rf_params, X_train, y_train
    )
    print(f"  - RF 최적 파라미터: {rf_gs.best_params_}")
    print(f"  - RF CV 최적 재현율: {rf_gs.best_score_:.4f}")

    best_knn = knn_gs.best_estimator_
    best_rf  = rf_gs.best_estimator_

    # ── 5단계: 평가 ────────────────────────────────────────────────────────
    print("\n[5단계] 테스트셋 평가")
    knn_pred  = best_knn.predict(X_test)
    rf_pred   = best_rf.predict(X_test)
    knn_prob  = best_knn.predict_proba(X_test)[:, 1]
    rf_prob   = best_rf.predict_proba(X_test)[:, 1]

    knn_acc    = accuracy_score(y_test, knn_pred)
    rf_acc     = accuracy_score(y_test, rf_pred)
    knn_recall = recall_score(y_test, knn_pred, pos_label=1)
    rf_recall  = recall_score(y_test, rf_pred, pos_label=1)
    knn_prec   = precision_score(y_test, knn_pred, pos_label=1, zero_division=0)
    rf_prec    = precision_score(y_test, rf_pred, pos_label=1, zero_division=0)
    knn_f1     = f1_score(y_test, knn_pred, pos_label=1)
    rf_f1      = f1_score(y_test, rf_pred, pos_label=1)

    print(f"\n  [KNN 분류 리포트]")
    print(classification_report(y_test, knn_pred, digits=4))
    print(f"  [RF 분류 리포트]")
    print(classification_report(y_test, rf_pred, digits=4))

    # 시각화 저장
    print("[5단계] 시각화 저장 중...")
    save_confusion_matrix(y_test, knn_pred, "KNN 혼동행렬",
                          RESULT_DIR / "confusion_matrix_knn.png")
    save_confusion_matrix(y_test, rf_pred, "Random Forest 혼동행렬",
                          RESULT_DIR / "confusion_matrix_rf.png")

    auc_knn, auc_rf, ap_knn, ap_rf = save_roc_and_pr_curves(
        y_test, knn_prob, rf_prob,
        RESULT_DIR / "roc_pr_curves.png"
    )
    print(f"  - ROC-AUC: KNN={auc_knn:.4f}, RF={auc_rf:.4f}")
    print(f"  - AP:      KNN={ap_knn:.4f}, RF={ap_rf:.4f}")

    save_performance_bar(
        knn_acc, knn_recall, knn_prec, knn_f1,
        rf_acc,  rf_recall,  rf_prec,  rf_f1,
        auc_knn, auc_rf, ap_knn, ap_rf,
        RESULT_DIR / "performance_summary.png"
    )

    # ── 6단계-A: 피처 중요도 ───────────────────────────────────────────────
    print("\n[6단계-A] RF 피처 중요도 분석")
    imp_df = save_feature_importance(
        best_rf, top_n=15,
        save_path=RESULT_DIR / "feature_importance_rf.png"
    )
    print(imp_df.head(8).to_string(index=False))

    # ── 6단계-B: SHAP ──────────────────────────────────────────────────────
    print("\n[6단계-B] SHAP 분석")
    _, class1_vals, X_shap_df, base_val = prepare_shap(best_rf, X_test)

    save_shap_summary(class1_vals, X_shap_df,
                      RESULT_DIR / "shap_summary.png")

    save_shap_dependence(class1_vals, X_shap_df,
                         feature_substring="G1_질병조퇴",
                         title="SHAP Dependence: G1 질병조퇴 횟수",
                         save_path=RESULT_DIR / "shap_dependence_질병조퇴.png")

    save_shap_dependence(class1_vals, X_shap_df,
                         feature_substring="1학년_평균등급",
                         title="SHAP Dependence: 1학년 평균등급",
                         save_path=RESULT_DIR / "shap_dependence_평균등급.png")

    rng = np.random.default_rng(42)
    waterfall_note = save_shap_waterfall(
        X_shap_df, y_test, rf_pred, rf_prob,
        class1_vals, base_val,
        RESULT_DIR / "shap_waterfall.png", rng
    )
    print(f"  - Waterfall: {waterfall_note}")

    # ── 6단계-C: Ablation ──────────────────────────────────────────────────
    print("\n[6단계-C] Ablation 실험 (1학년 평균등급 제거)")
    knn_acc_ab, knn_rec_ab, rf_acc_ab, rf_rec_ab = run_ablation_experiment(
        X_train, X_test, y_train, y_test,
        knn_acc, knn_recall, rf_acc, rf_recall,
        RESULT_DIR / "ablation_study.png"
    )
    print(f"  - KNN: 정확도 {knn_acc:.4f}→{knn_acc_ab:.4f} | 재현율 {knn_recall:.4f}→{knn_rec_ab:.4f}")
    print(f"  - RF:  정확도 {rf_acc:.4f}→{rf_acc_ab:.4f}  | 재현율 {rf_recall:.4f}→{rf_rec_ab:.4f}")

    # ── 6단계-D: 임계값 분석 ───────────────────────────────────────────────
    print("\n[6단계-D] RF 임계값 분석")
    save_threshold_analysis(y_test, rf_prob,
                            RESULT_DIR / "threshold_analysis.png")

    # ── 결과 리포트 저장 ───────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "학생 데이터 기반 고등학교 성적 하락 위험군 조기 예측 모델",
        "머신러닝 파이프라인 최종 결과 리포트",
        "=" * 60,
        "",
        f"[데이터 정보]",
        f"  전체 샘플: {len(df)}명 | 위험군: {y.sum()}명({y.mean()*100:.1f}%) | 안정군: {(1-y).sum()}명",
        "",
        f"[최적 하이퍼파라미터]",
        f"  KNN: {knn_gs.best_params_} (CV 재현율: {knn_gs.best_score_:.4f})",
        f"  RF:  {rf_gs.best_params_}  (CV 재현율: {rf_gs.best_score_:.4f})",
        "",
        f"[테스트셋 성능 비교]",
        f"  {'지표':<20} {'KNN':>8} {'RF':>8}",
        f"  {'정확도(Accuracy)':<20} {knn_acc:>8.4f} {rf_acc:>8.4f}",
        f"  {'위험군 재현율':<20} {knn_recall:>8.4f} {rf_recall:>8.4f}",
        f"  {'위험군 정밀도':<20} {knn_prec:>8.4f} {rf_prec:>8.4f}",
        f"  {'위험군 F1':<20} {knn_f1:>8.4f} {rf_f1:>8.4f}",
        f"  {'ROC-AUC':<20} {auc_knn:>8.4f} {auc_rf:>8.4f}",
        f"  {'Average Precision':<20} {ap_knn:>8.4f} {ap_rf:>8.4f}",
        "",
        f"[Ablation 실험: 1학년 평균등급 제거]",
        f"  {'':22} {'정확도':>8} {'재현율':>8}",
        f"  {'KNN (등급 포함)':22} {knn_acc:>8.4f} {knn_recall:>8.4f}",
        f"  {'KNN (등급 제외)':22} {knn_acc_ab:>8.4f} {knn_rec_ab:>8.4f}",
        f"  {'RF  (등급 포함)':22} {rf_acc:>8.4f} {rf_recall:>8.4f}",
        f"  {'RF  (등급 제외)':22} {rf_acc_ab:>8.4f} {rf_rec_ab:>8.4f}",
        "",
        f"[RF 상위 중요 변수]",
    ] + [f"  {row['feature']:<45} {row['importance']:.4f}"
         for _, row in imp_df.head(8).iterrows()] + [
        "",
        f"[교육 현장 적용 의견]",
        "  RF가 위험군 재현율·AP 모두 우수 → 조기 개입 대상 발굴에 더 유리",
        "  제안 임계값 0.4: 재현율 우선 탐지, 이후 교사 전문 판단으로 최종 확인",
        "  출결 조기 신호: G1 질병조퇴 1~2회 초과 시점부터 담임 면담 권장",
        "=" * 60,
    ]
    report_path = RESULT_DIR / "model_comparison_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n[완료] 모든 결과가 '{RESULT_DIR}' 폴더에 저장되었습니다.")
    print(f"  보고서: {report_path.name}")
    print(f"  이미지: {len(list(RESULT_DIR.glob('*.png')))}개")


# ── 스크립트 직접 실행 시 main() 호출 ──────────────────────────────────────────
if __name__ == "__main__":
    main()
