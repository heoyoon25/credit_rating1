import io
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE


# =========================================================
# 0. 기본 설정
# =========================================================
st.set_page_config(
    page_title="개인신용평가 모델",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42

STATE_DEFAULTS = {
    "raw_df": None,
    "working_df": None,
    "upload_signature": None,
    "target_col": None,
    "splits": None,
    "resampled": None,
    "model_results": {},
    "encoding_maps": {},
    "preprocessing_log": [],
    "feature_selection_info": None,
}

for key, value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 1. 공통 함수
# =========================================================
def reset_downstream():
    """전처리 데이터가 바뀌면 분할/오버샘플링/모델 결과를 초기화."""
    st.session_state.splits = None
    st.session_state.resampled = None
    st.session_state.model_results = {}


def update_working_df(new_df: pd.DataFrame, action: str):
    st.session_state.working_df = new_df.copy()
    st.session_state.preprocessing_log.append(action)
    reset_downstream()


def get_current_df():
    return st.session_state.working_df


def class_count_table(y: pd.Series, label="class"):
    vc = y.value_counts().sort_index()
    return pd.DataFrame({label: vc.index, "count": vc.values})


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return np.nan


def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": safe_auc(y_true, y_prob),
    }


def snap_synthetic_to_training_domain(synthetic: pd.DataFrame, real: pd.DataFrame):
    """
    CTGAN이 생성한 값이 원 데이터의 범위를 크게 벗어나지 않도록 보정.
    특히 0/1 더미변수나 소수 개 정수 범주형 변수는 가장 가까운 실제 값으로 스냅.
    """
    result = synthetic.copy()

    for col in real.columns:
        if col not in result.columns:
            continue

        real_col = pd.to_numeric(real[col], errors="coerce")
        syn_col = pd.to_numeric(result[col], errors="coerce")

        if real_col.notna().sum() == 0:
            continue

        min_v = real_col.min()
        max_v = real_col.max()
        syn_col = syn_col.clip(min_v, max_v)

        unique_vals = np.sort(real_col.dropna().unique())

        # 0/1 dummy 또는 정수형의 소수 범주 변수
        is_discrete = (
            len(unique_vals) <= 30
            and np.all(np.isclose(unique_vals, np.round(unique_vals)))
        )

        if is_discrete and len(unique_vals) > 0:
            arr = syn_col.to_numpy(dtype=float)
            distances = np.abs(arr[:, None] - unique_vals[None, :])
            nearest_idx = distances.argmin(axis=1)
            syn_col = pd.Series(unique_vals[nearest_idx], index=result.index)

        result[col] = syn_col

    return result


def build_minority_profile(df: pd.DataFrame, included_cols):
    """API에 원본 행 대신 전달할 Minority class 통계 프로파일."""
    profile = {}
    for col in included_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        info = {
            "dtype": str(df[col].dtype),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "q1": float(s.quantile(0.25)),
            "median": float(s.median()),
            "q3": float(s.quantile(0.75)),
        }
        uniques = np.sort(s.unique())
        if len(uniques) <= 20:
            info["allowed_values"] = [float(v) for v in uniques]
        profile[col] = info
    return profile


def generate_openai_synthetic_rows(
    api_key,
    model_name,
    real_minority,
    n_rows,
    batch_size=20,
    excluded_cols=None,
    include_samples=False,
    sample_rows=5,
):
    """
    OpenAI Structured Outputs를 이용해 Minority class의 합성 feature 행을 생성.
    기본값은 실제 고객 행을 보내지 않고 통계 프로파일만 전송한다.
    excluded_cols는 API에 전송하지 않고 Minority 실제값에서 로컬 bootstrap한다.
    """
    from openai import OpenAI

    excluded_cols = excluded_cols or []
    all_cols = real_minority.columns.tolist()
    ai_cols = [c for c in all_cols if c not in excluded_cols]

    if not ai_cols:
        raise ValueError("API에 전달할 변수가 하나 이상 필요합니다.")

    profile = build_minority_profile(real_minority, ai_cols)

    row_properties = {c: {"type": "number"} for c in ai_cols}
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": row_properties,
                    "required": ai_cols,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    }

    prompt_payload = {
        "task": "Generate realistic synthetic minority-class tabular credit-risk feature rows.",
        "rules": [
            "Generate new synthetic observations, not explanations.",
            "Respect each feature's observed range and allowed_values when provided.",
            "Preserve plausible dependencies and correlations among variables.",
            "Do not copy an input row verbatim.",
            "Return exactly the requested number of rows.",
        ],
        "minority_feature_profile": profile,
    }

    if include_samples:
        safe_sample = real_minority[ai_cols].sample(
            n=min(sample_rows, len(real_minority)),
            random_state=RANDOM_STATE,
        )
        prompt_payload["example_minority_rows"] = safe_sample.to_dict(orient="records")

    client = OpenAI(api_key=api_key)
    generated_batches = []
    remaining = int(n_rows)
    batch_no = 0

    while remaining > 0:
        batch_no += 1
        current_n = min(int(batch_size), remaining)
        request_payload = dict(prompt_payload)
        request_payload["number_of_rows"] = current_n

        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You generate synthetic tabular data for an academic credit-risk "
                        "classification experiment. Follow the supplied statistical constraints "
                        "and return only data conforming to the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "synthetic_credit_rows",
                    "strict": True,
                    "schema": schema,
                }
            },
        )

        parsed = json.loads(response.output_text)
        rows = parsed.get("rows", [])
        if not rows:
            raise ValueError(f"API batch {batch_no}에서 합성 행을 받지 못했습니다.")

        batch_df = pd.DataFrame(rows)
        for c in ai_cols:
            if c not in batch_df.columns:
                raise ValueError(f"API 응답에 변수 '{c}'가 없습니다.")
            batch_df[c] = pd.to_numeric(batch_df[c], errors="coerce")

        batch_df = batch_df[ai_cols].dropna().copy()
        if batch_df.empty:
            raise ValueError(f"API batch {batch_no}의 데이터가 수치형 검증을 통과하지 못했습니다.")

        # API에 보내지 않은 변수는 외부 전송 없이 Minority 실제 분포에서 bootstrap
        for c in excluded_cols:
            rng = np.random.default_rng(RANDOM_STATE + batch_no)
            source = real_minority[c].dropna().to_numpy()
            if len(source) == 0:
                batch_df[c] = 0.0
            else:
                batch_df[c] = rng.choice(source, size=len(batch_df), replace=True)

        batch_df = batch_df.reindex(columns=all_cols)
        batch_df = snap_synthetic_to_training_domain(batch_df, real_minority)
        generated_batches.append(batch_df)
        remaining -= min(len(batch_df), current_n)

    synthetic = pd.concat(generated_batches, ignore_index=True).head(int(n_rows))
    return synthetic


def parse_hidden_layers(text):
    try:
        layers = tuple(int(x.strip()) for x in text.split(",") if x.strip())
        if not layers or any(x <= 0 for x in layers):
            raise ValueError
        return layers
    except Exception:
        return (128, 64)


def train_and_evaluate_selected_models(selected_models, params):
    splits = st.session_state.splits
    resampled = st.session_state.resampled

    if splits is None:
        st.error("먼저 데이터 전처리 페이지에서 Train / Validation / Test 분할을 수행하세요.")
        return

    if resampled is not None:
        X_train = resampled["X_train"].copy()
        y_train = resampled["y_train"].copy()
        sampling_name = resampled["method"]
    else:
        X_train = splits["X_train"].copy()
        y_train = splits["y_train"].copy()
        sampling_name = "No oversampling"

    X_val = splits["X_val"].copy()
    y_val = splits["y_val"].copy()
    X_test = splits["X_test"].copy()
    y_test = splits["y_test"].copy()

    results = {}

    # 선형/신경망 모델용 스케일러는 오직 Train에만 fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    progress = st.progress(0)
    total = max(len(selected_models), 1)

    for idx, model_name in enumerate(selected_models, start=1):
        with st.spinner(f"{model_name} 학습 중..."):
            if model_name == "Logistic Regression":
                model = LogisticRegression(
                    C=params["logistic_c"],
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train_scaled, y_train)

                val_prob = model.predict_proba(X_val_scaled)[:, 1]
                test_prob = model.predict_proba(X_test_scaled)[:, 1]
                val_pred = (val_prob >= 0.5).astype(int)
                test_pred = (test_prob >= 0.5).astype(int)

                model_obj = model
                model_scaler = scaler

            elif model_name == "Random Forest":
                max_depth = None if params["rf_max_depth"] == 0 else params["rf_max_depth"]
                model = RandomForestClassifier(
                    n_estimators=params["rf_n_estimators"],
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None,
                )
                model.fit(X_train, y_train)

                val_prob = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
                val_pred = (val_prob >= 0.5).astype(int)
                test_pred = (test_prob >= 0.5).astype(int)

                model_obj = model
                model_scaler = None

            elif model_name == "Decision Tree":
                max_depth = None if params["dt_max_depth"] == 0 else params["dt_max_depth"]
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=params["dt_min_samples_split"],
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train, y_train)

                val_prob = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
                val_pred = (val_prob >= 0.5).astype(int)
                test_pred = (test_prob >= 0.5).astype(int)

                model_obj = model
                model_scaler = None

            elif model_name == "Multilayer Perceptron":
                model = MLPClassifier(
                    hidden_layer_sizes=params["mlp_layers"],
                    activation="relu",
                    solver="adam",
                    learning_rate_init=params["mlp_lr"],
                    max_iter=params["mlp_max_iter"],
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_train_scaled, y_train)

                val_prob = model.predict_proba(X_val_scaled)[:, 1]
                test_prob = model.predict_proba(X_test_scaled)[:, 1]
                val_pred = (val_prob >= 0.5).astype(int)
                test_pred = (test_prob >= 0.5).astype(int)

                model_obj = model
                model_scaler = scaler

            elif model_name == "DNN":
                try:
                    import tensorflow as tf
                except ImportError:
                    st.error(
                        "TensorFlow가 설치되어 있지 않습니다. "
                        "`pip install tensorflow` 후 다시 실행하세요."
                    )
                    progress.progress(idx / total)
                    continue

                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(RANDOM_STATE)

                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                        tf.keras.layers.Dense(128, activation="relu"),
                        tf.keras.layers.Dropout(params["dnn_dropout"]),
                        tf.keras.layers.Dense(64, activation="relu"),
                        tf.keras.layers.Dropout(params["dnn_dropout"]),
                        tf.keras.layers.Dense(32, activation="relu"),
                        tf.keras.layers.Dense(1, activation="sigmoid"),
                    ]
                )

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=params["dnn_lr"]
                    ),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )

                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=12,
                        restore_best_weights=True,
                    )
                ]

                history = model.fit(
                    X_train_scaled,
                    y_train.to_numpy(),
                    validation_data=(X_val_scaled, y_val.to_numpy()),
                    epochs=params["dnn_epochs"],
                    batch_size=params["dnn_batch_size"],
                    verbose=0,
                    callbacks=callbacks,
                )

                val_prob = model.predict(X_val_scaled, verbose=0).ravel()
                test_prob = model.predict(X_test_scaled, verbose=0).ravel()
                val_pred = (val_prob >= 0.5).astype(int)
                test_pred = (test_prob >= 0.5).astype(int)

                model_obj = model
                model_scaler = scaler

            else:
                continue

            test_metrics = calculate_metrics(y_test, test_pred, test_prob)
            val_metrics = calculate_metrics(y_val, val_pred, val_prob)

            fpr, tpr, thresholds = roc_curve(y_test, test_prob)

            results[model_name] = {
                "model": model_obj,
                "scaler": model_scaler,
                "sampling": sampling_name,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
                "test_y_true": y_test.to_numpy(),
                "test_y_pred": np.asarray(test_pred),
                "test_y_prob": np.asarray(test_prob),
                "roc_fpr": fpr,
                "roc_tpr": tpr,
                "roc_thresholds": thresholds,
            }

            if model_name == "DNN":
                results[model_name]["history"] = history.history

        progress.progress(idx / total)

    st.session_state.model_results = results
    st.success("선택한 모델의 학습과 평가가 완료되었습니다.")


# =========================================================
# 2. 사이드바
# =========================================================
st.sidebar.title("💳 개인신용평가 모델")
st.sidebar.caption("Credit Scoring Modeling System")

page = st.sidebar.radio(
    "메뉴",
    [
        "1. 데이터 업로드",
        "2. 데이터 탐색",
        "3. 데이터 전처리",
        "4. 모델 학습",
        "5. 결과 분석",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("진행 상태")

st.sidebar.write(
    "✅ 데이터 업로드" if st.session_state.working_df is not None else "⬜ 데이터 업로드"
)
st.sidebar.write(
    "✅ 데이터 분할" if st.session_state.splits is not None else "⬜ 데이터 분할"
)
st.sidebar.write(
    f"✅ 오버샘플링: {st.session_state.resampled['method']}"
    if st.session_state.resampled is not None
    else "⬜ 오버샘플링"
)
st.sidebar.write(
    "✅ 모델 학습" if st.session_state.model_results else "⬜ 모델 학습"
)


# =========================================================
# 3. 데이터 업로드
# =========================================================
if page == "1. 데이터 업로드":
    st.title("1. 데이터 업로드")
    st.write("CSV 또는 Excel 형식의 개인신용평가 데이터를 업로드하세요.")

    uploaded_file = st.file_uploader(
        "데이터 파일 선택",
        type=["csv", "xlsx", "xls"],
        help="CSV, XLSX, XLS 파일을 지원합니다.",
    )

    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        signature = hashlib.md5(raw_bytes).hexdigest()

        if signature != st.session_state.upload_signature:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(raw_bytes))
                else:
                    df = pd.read_excel(io.BytesIO(raw_bytes))

                st.session_state.raw_df = df.copy()
                st.session_state.working_df = df.copy()
                st.session_state.upload_signature = signature
                st.session_state.target_col = None
                st.session_state.encoding_maps = {}
                st.session_state.preprocessing_log = []
                st.session_state.feature_selection_info = None
                reset_downstream()

                st.success("새 데이터가 정상적으로 업로드되었습니다.")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    df = get_current_df()

    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("행 수", f"{len(df):,}")
        c2.metric("열 수", f"{df.shape[1]:,}")
        c3.metric("결측치 수", f"{int(df.isna().sum().sum()):,}")

        st.subheader("데이터 미리보기")
        st.dataframe(df.head(20), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "현재 데이터 CSV로 다운로드",
            data=csv_bytes,
            file_name="credit_scoring_current_data.csv",
            mime="text/csv",
        )


# =========================================================
# 4. 데이터 탐색
# =========================================================
elif page == "2. 데이터 탐색":
    st.title("2. 데이터 탐색")

    df = get_current_df()

    if df is None:
        st.warning("먼저 데이터 업로드 메뉴에서 데이터를 업로드하세요.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["데이터 요약", "변수/타입", "시각화", "상관관계"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("행", f"{df.shape[0]:,}")
        c2.metric("열", f"{df.shape[1]:,}")
        c3.metric("전체 결측치", f"{int(df.isna().sum().sum()):,}")
        c4.metric("중복 행", f"{int(df.duplicated().sum()):,}")

        st.subheader("기술통계")
        summary = df.describe(include="all").T
        st.dataframe(summary, use_container_width=True)

        st.subheader("원자료")
        max_preview = max(1, min(100, len(df)))
        n_preview = st.slider("표시할 행 수", 1, max_preview, min(20, max_preview))
        st.dataframe(df.head(n_preview), use_container_width=True)

    with tab2:
        info_df = pd.DataFrame(
            {
                "변수명": df.columns,
                "데이터 타입": [str(x) for x in df.dtypes],
                "결측치 수": df.isna().sum().values,
                "결측치 비율(%)": (df.isna().mean().values * 100).round(2),
                "고유값 수": df.nunique(dropna=True).values,
            }
        )
        st.dataframe(info_df, use_container_width=True)

    with tab3:
        chart_type = st.selectbox(
            "차트 종류",
            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", "Line Chart"],
        )

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if chart_type == "Histogram":
            if not numeric_cols:
                st.warning("수치형 변수가 없습니다.")
            else:
                x_col = st.selectbox("X축 변수", numeric_cols)
                bins = st.slider("Bin 수", 5, 100, 30)
                fig = px.histogram(df, x=x_col, nbins=bins, title=f"Histogram: {x_col}")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box Plot":
            if not numeric_cols:
                st.warning("수치형 변수가 없습니다.")
            else:
                y_col = st.selectbox("Y축 수치 변수", numeric_cols)
                x_option = st.selectbox("X축 그룹 변수", ["사용 안 함"] + all_cols)
                x_col = None if x_option == "사용 안 함" else x_option
                fig = px.box(df, x=x_col, y=y_col, points="outliers")
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter Plot":
            if len(numeric_cols) < 2:
                st.warning("Scatter Plot에는 수치형 변수가 2개 이상 필요합니다.")
            else:
                x_col = st.selectbox("X축 변수", numeric_cols, key="scatter_x")
                y_candidates = [c for c in numeric_cols if c != x_col]
                y_col = st.selectbox("Y축 변수", y_candidates, key="scatter_y")
                color_option = st.selectbox("색상 구분", ["사용 안 함"] + all_cols)
                color_col = None if color_option == "사용 안 함" else color_option
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar Chart":
            x_col = st.selectbox("X축 변수", all_cols, key="bar_x")
            y_option = st.selectbox(
                "Y축 변수",
                ["빈도(Count)"] + numeric_cols,
                key="bar_y",
            )

            if y_option == "빈도(Count)":
                plot_df = (
                    df[x_col]
                    .astype(str)
                    .value_counts(dropna=False)
                    .rename_axis(x_col)
                    .reset_index(name="Count")
                )
                fig = px.bar(plot_df, x=x_col, y="Count")
            else:
                agg_method = st.selectbox("집계 방식", ["mean", "sum", "median"])
                plot_df = (
                    df.groupby(x_col, dropna=False)[y_option]
                    .agg(agg_method)
                    .reset_index()
                )
                fig = px.bar(plot_df, x=x_col, y=y_option)

            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line Chart":
            if not numeric_cols:
                st.warning("Y축으로 사용할 수치형 변수가 없습니다.")
            else:
                x_col = st.selectbox("X축 변수", all_cols, key="line_x")
                y_col = st.selectbox("Y축 변수", numeric_cols, key="line_y")
                sort_x = st.checkbox("X축 기준 정렬", value=True)
                plot_df = df[[x_col, y_col]].dropna().copy()

                if sort_x:
                    try:
                        plot_df = plot_df.sort_values(x_col)
                    except Exception:
                        pass

                fig = px.line(plot_df, x=x_col, y=y_col)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            st.warning("상관관계 분석에는 수치형 변수가 2개 이상 필요합니다.")
        else:
            method = st.selectbox(
                "상관계수",
                ["pearson", "spearman", "kendall"],
            )
            corr = numeric_df.corr(method=method)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title=f"Correlation Heatmap ({method})",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(corr.round(3), use_container_width=True)


# =========================================================
# 5. 데이터 전처리
# =========================================================
elif page == "3. 데이터 전처리":
    st.title("3. 데이터 전처리")

    df = get_current_df()

    if df is None:
        st.warning("먼저 데이터를 업로드하세요.")
        st.stop()

    st.subheader("Target 변수 설정")
    target_default_idx = 0
    if st.session_state.target_col in df.columns:
        target_default_idx = df.columns.tolist().index(st.session_state.target_col)

    target_col = st.selectbox(
        "종속변수(Target)를 선택하세요.",
        df.columns.tolist(),
        index=target_default_idx,
    )
    st.session_state.target_col = target_col

    if st.button("원본 업로드 데이터로 전처리 초기화"):
        st.session_state.working_df = st.session_state.raw_df.copy()
        st.session_state.encoding_maps = {}
        st.session_state.preprocessing_log = []
        st.session_state.feature_selection_info = None
        reset_downstream()
        st.success("전처리 내용을 초기화했습니다.")
        st.rerun()

    tab_missing, tab_outlier, tab_encoding, tab_fs, tab_partition = st.tabs(
        [
            "결측치 처리",
            "이상치 처리",
            "인코딩",
            "Feature Selection",
            "Data Partitioning / Oversampling",
        ]
    )

    # -----------------------------
    # 결측치
    # -----------------------------
    with tab_missing:
        df = get_current_df()

        missing_df = pd.DataFrame(
            {
                "변수": df.columns,
                "결측치 수": df.isna().sum().values,
                "결측치 비율(%)": (df.isna().mean().values * 100).round(2),
            }
        )
        missing_df = missing_df[missing_df["결측치 수"] > 0]

        if missing_df.empty:
            st.success("현재 결측치가 없습니다.")
        else:
            st.dataframe(missing_df, use_container_width=True)

            missing_cols = missing_df["변수"].tolist()
            selected_cols = st.multiselect(
                "처리할 변수",
                missing_cols,
                default=missing_cols,
            )

            method = st.selectbox(
                "결측치 처리 방법",
                [
                    "행 삭제",
                    "열 삭제",
                    "자동 대체 (수치형=Median / 범주형=Mode)",
                    "Mean 대체",
                    "Median 대체",
                    "Mode 대체",
                    "사용자 지정 값",
                ],
            )

            fill_value = None
            if method == "사용자 지정 값":
                fill_value = st.text_input("대체 값", value="0")

            if st.button("결측치 처리 적용", key="apply_missing"):
                if not selected_cols:
                    st.warning("처리할 변수를 선택하세요.")
                else:
                    new_df = df.copy()

                    try:
                        if method == "행 삭제":
                            new_df = new_df.dropna(subset=selected_cols)

                        elif method == "열 삭제":
                            if target_col in selected_cols:
                                st.error("Target 변수는 삭제할 수 없습니다.")
                                st.stop()
                            new_df = new_df.drop(columns=selected_cols)

                        elif method == "자동 대체 (수치형=Median / 범주형=Mode)":
                            for col in selected_cols:
                                if pd.api.types.is_numeric_dtype(new_df[col]):
                                    new_df[col] = new_df[col].fillna(new_df[col].median())
                                else:
                                    mode = new_df[col].mode(dropna=True)
                                    if not mode.empty:
                                        new_df[col] = new_df[col].fillna(mode.iloc[0])

                        elif method == "Mean 대체":
                            for col in selected_cols:
                                if pd.api.types.is_numeric_dtype(new_df[col]):
                                    new_df[col] = new_df[col].fillna(new_df[col].mean())
                                else:
                                    st.warning(f"{col}: 수치형이 아니므로 건너뜁니다.")

                        elif method == "Median 대체":
                            for col in selected_cols:
                                if pd.api.types.is_numeric_dtype(new_df[col]):
                                    new_df[col] = new_df[col].fillna(new_df[col].median())
                                else:
                                    st.warning(f"{col}: 수치형이 아니므로 건너뜁니다.")

                        elif method == "Mode 대체":
                            for col in selected_cols:
                                mode = new_df[col].mode(dropna=True)
                                if not mode.empty:
                                    new_df[col] = new_df[col].fillna(mode.iloc[0])

                        elif method == "사용자 지정 값":
                            for col in selected_cols:
                                value = fill_value
                                if pd.api.types.is_numeric_dtype(new_df[col]):
                                    try:
                                        value = float(fill_value)
                                    except Exception:
                                        pass
                                new_df[col] = new_df[col].fillna(value)

                        update_working_df(
                            new_df,
                            f"결측치 처리: {method} / {selected_cols}",
                        )
                        st.success("결측치 처리를 적용했습니다.")
                    except Exception as e:
                        st.error(f"결측치 처리 중 오류: {e}")

    # -----------------------------
    # 이상치
    # -----------------------------
    with tab_outlier:
        df = get_current_df()
        numeric_cols = [
            c
            for c in df.select_dtypes(include=np.number).columns.tolist()
            if c != st.session_state.target_col
        ]

        if not numeric_cols:
            st.warning("처리 가능한 수치형 독립변수가 없습니다.")
        else:
            selected_cols = st.multiselect(
                "이상치를 처리할 수치형 변수",
                numeric_cols,
                key="outlier_cols",
            )

            outlier_method = st.selectbox(
                "이상치 처리 방법",
                ["IQR 기준 행 제거", "IQR Winsorizing(Capping)", "Z-score 기준 행 제거"],
            )

            if outlier_method.startswith("IQR"):
                iqr_k = st.slider("IQR 배수", 1.0, 3.0, 1.5, 0.1)
            else:
                z_threshold = st.slider("Z-score 임계값", 2.0, 5.0, 3.0, 0.1)

            if st.button("이상치 처리 적용", key="apply_outlier"):
                if not selected_cols:
                    st.warning("처리할 변수를 선택하세요.")
                else:
                    new_df = df.copy()
                    before_n = len(new_df)

                    if outlier_method == "IQR 기준 행 제거":
                        keep_mask = pd.Series(True, index=new_df.index)

                        for col in selected_cols:
                            q1 = new_df[col].quantile(0.25)
                            q3 = new_df[col].quantile(0.75)
                            iqr = q3 - q1
                            low = q1 - iqr_k * iqr
                            high = q3 + iqr_k * iqr
                            keep_mask &= new_df[col].between(low, high) | new_df[col].isna()

                        new_df = new_df.loc[keep_mask].copy()

                    elif outlier_method == "IQR Winsorizing(Capping)":
                        for col in selected_cols:
                            q1 = new_df[col].quantile(0.25)
                            q3 = new_df[col].quantile(0.75)
                            iqr = q3 - q1
                            low = q1 - iqr_k * iqr
                            high = q3 + iqr_k * iqr
                            new_df[col] = new_df[col].clip(low, high)

                    elif outlier_method == "Z-score 기준 행 제거":
                        keep_mask = pd.Series(True, index=new_df.index)

                        for col in selected_cols:
                            mean = new_df[col].mean()
                            std = new_df[col].std()

                            if std and not np.isnan(std):
                                z = (new_df[col] - mean) / std
                                keep_mask &= (z.abs() <= z_threshold) | new_df[col].isna()

                        new_df = new_df.loc[keep_mask].copy()

                    update_working_df(
                        new_df,
                        f"이상치 처리: {outlier_method} / {selected_cols}",
                    )

                    st.success(
                        f"이상치 처리를 적용했습니다. 행 수: {before_n:,} → {len(new_df):,}"
                    )

    # -----------------------------
    # 인코딩
    # -----------------------------
    with tab_encoding:
        df = get_current_df()
        target_col = st.session_state.target_col

        categorical_cols = [
            c
            for c in df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            if c != target_col
        ]

        if not categorical_cols:
            st.success("현재 인코딩이 필요한 범주형 독립변수가 없습니다.")
        else:
            encode_cols = st.multiselect(
                "인코딩할 범주형 변수",
                categorical_cols,
                default=categorical_cols,
            )

            encoding_method = st.selectbox(
                "인코딩 방식",
                ["One-Hot Encoding", "Ordinal/Label Encoding"],
            )

            drop_first = False
            if encoding_method == "One-Hot Encoding":
                drop_first = st.checkbox("첫 번째 더미 범주 제거(drop_first)", value=False)

            if st.button("인코딩 적용", key="apply_encoding"):
                if not encode_cols:
                    st.warning("인코딩할 변수를 선택하세요.")
                else:
                    new_df = df.copy()

                    if encoding_method == "One-Hot Encoding":
                        new_df = pd.get_dummies(
                            new_df,
                            columns=encode_cols,
                            drop_first=drop_first,
                            dtype=int,
                        )

                    else:
                        for col in encode_cols:
                            categories = sorted(new_df[col].dropna().astype(str).unique())
                            mapping = {cat: i for i, cat in enumerate(categories)}
                            st.session_state.encoding_maps[col] = mapping

                            new_df[col] = (
                                new_df[col]
                                .astype("string")
                                .map(mapping)
                                .astype("Float64")
                            )

                    update_working_df(
                        new_df,
                        f"인코딩: {encoding_method} / {encode_cols}",
                    )
                    st.success("인코딩을 적용했습니다.")

                    if st.session_state.encoding_maps:
                        with st.expander("Label Encoding 매핑 확인"):
                            st.json(st.session_state.encoding_maps)

    # -----------------------------
    # Feature selection
    # -----------------------------
    with tab_fs:
        df = get_current_df()
        target_col = st.session_state.target_col

        if target_col not in df.columns:
            st.error("선택한 Target 변수가 현재 데이터에 없습니다.")
        else:
            feature_cols = [c for c in df.columns if c != target_col]

            fs_method = st.selectbox(
                "Feature Selection 방식",
                ["사용자 직접 선택", "SelectKBest (Mutual Information)"],
            )

            if fs_method == "사용자 직접 선택":
                selected_features = st.multiselect(
                    "모델에 사용할 Feature",
                    feature_cols,
                    default=feature_cols,
                )

                if st.button("선택한 Feature만 유지", key="manual_fs"):
                    if not selected_features:
                        st.warning("최소 1개 이상의 Feature를 선택하세요.")
                    else:
                        new_df = df[selected_features + [target_col]].copy()
                        st.session_state.feature_selection_info = {
                            "method": "Manual",
                            "features": selected_features,
                        }
                        update_working_df(
                            new_df,
                            f"Feature Selection(Manual): {selected_features}",
                        )
                        st.success(f"{len(selected_features)}개 Feature를 선택했습니다.")

            else:
                non_numeric = [
                    c
                    for c in feature_cols
                    if not pd.api.types.is_numeric_dtype(df[c])
                ]

                if non_numeric:
                    st.warning(
                        "SelectKBest 전에 모든 독립변수를 수치형으로 인코딩해야 합니다. "
                        f"현재 비수치형 변수: {non_numeric}"
                    )
                elif df[feature_cols].isna().any().any() or df[target_col].isna().any():
                    st.warning("SelectKBest 전에 결측치를 처리하세요.")
                else:
                    max_k = len(feature_cols)
                    k = st.slider(
                        "선택할 Feature 수(k)",
                        1,
                        max_k,
                        min(10, max_k),
                    )

                    if st.button("SelectKBest 실행", key="kbest_fs"):
                        y_temp = pd.factorize(df[target_col])[0]
                        selector = SelectKBest(
                            score_func=mutual_info_classif,
                            k=k,
                        )
                        selector.fit(df[feature_cols], y_temp)
                        selected_features = np.array(feature_cols)[selector.get_support()].tolist()

                        score_df = pd.DataFrame(
                            {
                                "Feature": feature_cols,
                                "MI Score": selector.scores_,
                                "Selected": selector.get_support(),
                            }
                        ).sort_values("MI Score", ascending=False)

                        st.session_state.feature_selection_info = {
                            "method": "SelectKBest-MI",
                            "features": selected_features,
                            "scores": score_df,
                        }

                        new_df = df[selected_features + [target_col]].copy()
                        update_working_df(
                            new_df,
                            f"Feature Selection(SelectKBest-MI, k={k}): {selected_features}",
                        )

                        st.success(f"{k}개 Feature를 선택했습니다.")
                        st.dataframe(score_df, use_container_width=True)

    # -----------------------------
    # Data Partitioning / Oversampling
    # -----------------------------
    with tab_partition:
        df = get_current_df()
        target_col = st.session_state.target_col

        st.markdown("### Train / Validation / Test 분할")

        if target_col not in df.columns:
            st.error("Target 변수를 다시 설정하세요.")
        else:
            X = df.drop(columns=[target_col])
            y_raw = df[target_col]

            non_numeric = [
                c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
            ]
            has_missing = X.isna().any().any() or y_raw.isna().any()

            if non_numeric:
                st.warning(
                    "Data Partitioning 전에 범주형 독립변수를 인코딩해야 합니다. "
                    f"비수치형 변수: {non_numeric}"
                )

            if has_missing:
                st.warning("Data Partitioning 전에 결측치를 처리하세요.")

            unique_target = list(pd.Series(y_raw.dropna().unique()).tolist())

            if len(unique_target) != 2:
                st.warning(
                    "현재 코드는 이진 개인신용평가 모델을 기준으로 합니다. "
                    f"Target의 고유값이 {len(unique_target)}개입니다: {unique_target}"
                )
            else:
                positive_class = st.selectbox(
                    "Positive class (부실/연체 등 1로 평가할 클래스)",
                    unique_target,
                    index=1 if len(unique_target) > 1 else 0,
                )

                c1, c2 = st.columns(2)
                with c1:
                    test_ratio = st.slider(
                        "Test 비율",
                        0.05,
                        0.40,
                        0.20,
                        0.05,
                    )
                with c2:
                    val_ratio = st.slider(
                        "Validation 비율",
                        0.05,
                        0.40,
                        0.20,
                        0.05,
                    )

                train_ratio = 1.0 - test_ratio - val_ratio
                st.info(
                    f"Train {train_ratio:.0%} / Validation {val_ratio:.0%} / Test {test_ratio:.0%}"
                )

                stratify = st.checkbox("Target 비율을 유지하여 층화 분할", value=True)

                if st.button("Data Partitioning 실행", key="partition"):
                    if non_numeric:
                        st.error("먼저 모든 독립변수를 수치형으로 인코딩하세요.")
                    elif has_missing:
                        st.error("먼저 결측치를 처리하세요.")
                    elif train_ratio <= 0:
                        st.error("Train 비율이 0보다 커야 합니다.")
                    else:
                        y = (y_raw == positive_class).astype(int)

                        stratify_y = y if stratify else None
                        temp_ratio = val_ratio + test_ratio

                        try:
                            X_train, X_temp, y_train, y_temp = train_test_split(
                                X,
                                y,
                                test_size=temp_ratio,
                                random_state=RANDOM_STATE,
                                stratify=stratify_y,
                            )

                            test_fraction_of_temp = test_ratio / temp_ratio
                            stratify_temp = y_temp if stratify else None

                            X_val, X_test, y_val, y_test = train_test_split(
                                X_temp,
                                y_temp,
                                test_size=test_fraction_of_temp,
                                random_state=RANDOM_STATE,
                                stratify=stratify_temp,
                            )

                            st.session_state.splits = {
                                "X_train": X_train.reset_index(drop=True),
                                "y_train": y_train.reset_index(drop=True),
                                "X_val": X_val.reset_index(drop=True),
                                "y_val": y_val.reset_index(drop=True),
                                "X_test": X_test.reset_index(drop=True),
                                "y_test": y_test.reset_index(drop=True),
                                "target_col": target_col,
                                "positive_class": positive_class,
                                "feature_names": X.columns.tolist(),
                                "ratios": {
                                    "train": train_ratio,
                                    "validation": val_ratio,
                                    "test": test_ratio,
                                },
                            }
                            st.session_state.resampled = None
                            st.session_state.model_results = {}

                            st.success("Train / Validation / Test 분할을 완료했습니다.")
                        except Exception as e:
                            st.error(
                                "분할 중 오류가 발생했습니다. 각 클래스의 데이터 수가 너무 적은지 확인하세요. "
                                f"\n\n오류: {e}"
                            )

                if st.session_state.splits is not None:
                    splits = st.session_state.splits

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.write("**Train**")
                        st.dataframe(
                            class_count_table(splits["y_train"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with c2:
                        st.write("**Validation**")
                        st.dataframe(
                            class_count_table(splits["y_val"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with c3:
                        st.write("**Test**")
                        st.dataframe(
                            class_count_table(splits["y_test"]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.divider()
                    st.markdown("### Train set 오버샘플링")
                    st.warning(
                        "오버샘플링은 Train set에만 적용합니다. "
                        "Validation/Test set은 원래 분포를 유지합니다."
                    )

                    oversampling_method = st.selectbox(
                        "오버샘플링 방법",
                        ["사용 안 함", "SMOTE", "CTGAN", "Generative AI (OpenAI API)"],
                    )

                    if oversampling_method == "사용 안 함":
                        if st.button("원본 Train set 사용", key="no_sampling"):
                            st.session_state.resampled = {
                                "X_train": splits["X_train"].copy(),
                                "y_train": splits["y_train"].copy(),
                                "method": "No oversampling",
                                "synthetic_rows": 0,
                            }
                            st.session_state.model_results = {}
                            st.success("오버샘플링 없이 원본 Train set을 사용합니다.")

                    elif oversampling_method == "SMOTE":
                        y_train = splits["y_train"]
                        counts = y_train.value_counts()
                        current_ratio = counts.min() / counts.max()

                        target_ratio = st.slider(
                            "목표 Minority / Majority 비율",
                            0.10,
                            1.00,
                            max(0.50, round(float(current_ratio), 2)),
                            0.05,
                            key="smote_ratio",
                        )

                        max_k = max(1, min(10, int(counts.min()) - 1))
                        k_neighbors = st.slider(
                            "SMOTE k_neighbors",
                            1,
                            max_k,
                            min(5, max_k),
                        )

                        if st.button("SMOTE 적용", key="run_smote"):
                            if counts.min() < 2:
                                st.error("SMOTE를 적용하려면 Minority class가 최소 2개 이상 필요합니다.")
                            elif target_ratio <= current_ratio:
                                st.error(
                                    f"목표 비율은 현재 비율({current_ratio:.3f})보다 커야 합니다."
                                )
                            else:
                                try:
                                    smote = SMOTE(
                                        sampling_strategy=target_ratio,
                                        random_state=RANDOM_STATE,
                                        k_neighbors=k_neighbors,
                                    )

                                    X_res, y_res = smote.fit_resample(
                                        splits["X_train"],
                                        splits["y_train"],
                                    )

                                    X_res = pd.DataFrame(
                                        X_res,
                                        columns=splits["feature_names"],
                                    )
                                    y_res = pd.Series(y_res, name=target_col)

                                    n_synthetic = len(X_res) - len(splits["X_train"])

                                    st.session_state.resampled = {
                                        "X_train": X_res,
                                        "y_train": y_res,
                                        "method": f"SMOTE (ratio={target_ratio:.2f})",
                                        "synthetic_rows": n_synthetic,
                                    }
                                    st.session_state.model_results = {}

                                    st.success(
                                        f"SMOTE 완료: 합성 데이터 {n_synthetic:,}행 추가"
                                    )
                                except Exception as e:
                                    st.error(f"SMOTE 적용 중 오류: {e}")

                    elif oversampling_method == "CTGAN":
                        y_train = splits["y_train"]
                        counts = y_train.value_counts()
                        minority_label = counts.idxmin()
                        current_ratio = counts.min() / counts.max()

                        target_ratio = st.slider(
                            "목표 Minority / Majority 비율",
                            0.10,
                            1.00,
                            max(0.50, round(float(current_ratio), 2)),
                            0.05,
                            key="ctgan_ratio",
                        )

                        ctgan_epochs = st.number_input(
                            "CTGAN Epochs",
                            min_value=50,
                            max_value=2000,
                            value=300,
                            step=50,
                        )

                        st.caption(
                            "CTGAN은 Train set의 Minority class 데이터만 학습한 뒤 "
                            "부족한 수만큼 합성 Feature 행을 생성합니다."
                        )

                        if st.button("CTGAN 적용", key="run_ctgan"):
                            if target_ratio <= current_ratio:
                                st.error(
                                    f"목표 비율은 현재 비율({current_ratio:.3f})보다 커야 합니다."
                                )
                            elif counts.min() < 10:
                                st.error(
                                    "CTGAN 학습을 위해 Minority 표본이 너무 적습니다. "
                                    "최소 수십 개 이상의 실제 Minority 표본을 권장합니다."
                                )
                            else:
                                try:
                                    from sdv.metadata import Metadata
                                    from sdv.single_table import CTGANSynthesizer

                                    majority_n = counts.max()
                                    minority_n = counts.min()
                                    target_minority_n = int(np.ceil(majority_n * target_ratio))
                                    n_to_generate = target_minority_n - minority_n

                                    real_minority = (
                                        splits["X_train"]
                                        .loc[y_train == minority_label]
                                        .reset_index(drop=True)
                                    )

                                    metadata = Metadata.detect_from_dataframe(real_minority)

                                    synthesizer = CTGANSynthesizer(
                                        metadata,
                                        enforce_rounding=False,
                                        epochs=int(ctgan_epochs),
                                        verbose=False,
                                    )

                                    with st.spinner(
                                        "CTGAN 학습 및 합성 데이터 생성 중입니다..."
                                    ):
                                        synthesizer.fit(real_minority)
                                        synthetic_X = synthesizer.sample(
                                            num_rows=n_to_generate
                                        )

                                    synthetic_X = synthetic_X[
                                        splits["feature_names"]
                                    ].copy()

                                    synthetic_X = snap_synthetic_to_training_domain(
                                        synthetic_X,
                                        real_minority,
                                    )

                                    X_res = pd.concat(
                                        [splits["X_train"], synthetic_X],
                                        ignore_index=True,
                                    )
                                    y_syn = pd.Series(
                                        [minority_label] * n_to_generate
                                    )
                                    y_res = pd.concat(
                                        [y_train, y_syn],
                                        ignore_index=True,
                                    )

                                    X_res, y_res = shuffle(
                                        X_res,
                                        y_res,
                                        random_state=RANDOM_STATE,
                                    )
                                    X_res = X_res.reset_index(drop=True)
                                    y_res = y_res.reset_index(drop=True)

                                    st.session_state.resampled = {
                                        "X_train": X_res,
                                        "y_train": y_res,
                                        "method": f"CTGAN (ratio={target_ratio:.2f}, epochs={ctgan_epochs})",
                                        "synthetic_rows": n_to_generate,
                                        "synthetic_sample": synthetic_X.head(20),
                                    }
                                    st.session_state.model_results = {}

                                    st.success(
                                        f"CTGAN 완료: 합성 데이터 {n_to_generate:,}행 추가"
                                    )
                                except ImportError:
                                    st.error(
                                        "SDV가 설치되어 있지 않습니다. "
                                        "`pip install sdv` 후 다시 실행하세요."
                                    )
                                except Exception as e:
                                    st.error(f"CTGAN 적용 중 오류: {e}")

                    elif oversampling_method == "Generative AI (OpenAI API)":
                        y_train = splits["y_train"]
                        counts = y_train.value_counts()
                        minority_label = counts.idxmin()
                        current_ratio = counts.min() / counts.max()

                        st.info(
                            "API Key는 password 입력창으로 받고 코드/CSV에 저장하지 않습니다. "
                            "기본 설정에서는 실제 고객 행 대신 Minority class의 통계 요약만 API로 전송합니다."
                        )

                        api_key = st.text_input(
                            "OpenAI API Key",
                            type="password",
                            key="openai_api_key_input",
                            help="이 값은 현재 Streamlit 세션에서 API 호출에만 사용합니다.",
                        )
                        model_name = st.text_input(
                            "OpenAI 모델명",
                            value="gpt-5.6",
                            key="openai_model_name",
                        )

                        target_ratio = st.slider(
                            "목표 Minority / Majority 비율",
                            0.10,
                            1.00,
                            max(0.50, round(float(current_ratio), 2)),
                            0.05,
                            key="genai_ratio",
                        )

                        majority_n = int(counts.max())
                        minority_n = int(counts.min())
                        target_minority_n = int(np.ceil(majority_n * target_ratio))
                        n_to_generate = max(0, target_minority_n - minority_n)

                        st.metric("필요한 AI 합성 행 수", f"{n_to_generate:,}")

                        batch_size = st.slider(
                            "API 1회당 생성 행 수",
                            5,
                            50,
                            20,
                            5,
                            key="genai_batch_size",
                        )
                        if n_to_generate > 0:
                            estimated_calls = int(np.ceil(n_to_generate / batch_size))
                            st.caption(f"예상 API 호출 횟수: 약 {estimated_calls:,}회")

                        real_minority = (
                            splits["X_train"]
                            .loc[y_train == minority_label]
                            .reset_index(drop=True)
                        )

                        likely_identifier_keywords = [
                            "id", "name", "phone", "mobile", "address", "email",
                            "resident", "ssn", "주민", "이름", "전화", "주소", "메일", "고객번호"
                        ]
                        default_excluded = [
                            c for c in real_minority.columns
                            if any(k in c.lower() for k in likely_identifier_keywords)
                        ]

                        excluded_cols = st.multiselect(
                            "API 전송에서 제외할 변수 (제외 변수는 로컬 bootstrap으로 생성)",
                            real_minority.columns.tolist(),
                            default=default_excluded,
                            key="genai_excluded_cols",
                        )

                        include_samples = st.checkbox(
                            "Minority 실제 샘플 일부도 API에 제공 (비식별 데이터일 때만 권장)",
                            value=False,
                            key="genai_include_samples",
                        )
                        sample_rows = 5
                        if include_samples:
                            sample_rows = st.slider(
                                "API에 제공할 예시 행 수",
                                1,
                                min(20, len(real_minority)),
                                min(5, len(real_minority)),
                                key="genai_sample_rows",
                            )

                        remove_duplicates = st.checkbox(
                            "원본 Train과 완전히 동일한 합성행 제거",
                            value=True,
                            key="genai_drop_duplicates",
                        )

                        cost_ack = st.checkbox(
                            "API 사용량에 따라 비용이 발생할 수 있음을 확인했습니다.",
                            value=False,
                            key="genai_cost_ack",
                        )

                        if st.button("Generative AI 오버샘플링 실행", key="run_genai"):
                            if not api_key.strip():
                                st.error("OpenAI API Key를 입력하세요.")
                            elif not model_name.strip():
                                st.error("사용할 OpenAI 모델명을 입력하세요.")
                            elif target_ratio <= current_ratio:
                                st.error(
                                    f"목표 비율은 현재 비율({current_ratio:.3f})보다 커야 합니다."
                                )
                            elif n_to_generate <= 0:
                                st.error("현재 설정에서는 추가 생성할 Minority 데이터가 없습니다.")
                            elif not cost_ack:
                                st.error("API 비용 발생 가능성 확인란을 체크하세요.")
                            elif len(excluded_cols) == real_minority.shape[1]:
                                st.error("모든 변수를 API 전송 제외로 선택할 수는 없습니다.")
                            else:
                                try:
                                    with st.spinner("OpenAI API로 Minority 합성 데이터를 생성 중입니다..."):
                                        synthetic_X = generate_openai_synthetic_rows(
                                            api_key=api_key.strip(),
                                            model_name=model_name.strip(),
                                            real_minority=real_minority,
                                            n_rows=n_to_generate,
                                            batch_size=batch_size,
                                            excluded_cols=excluded_cols,
                                            include_samples=include_samples,
                                            sample_rows=sample_rows,
                                        )

                                    if remove_duplicates:
                                        real_hashable = splits["X_train"].astype(str)
                                        syn_hashable = synthetic_X.astype(str)
                                        real_keys = set(map(tuple, real_hashable.to_numpy()))
                                        keep = [tuple(row) not in real_keys for row in syn_hashable.to_numpy()]
                                        synthetic_X = synthetic_X.loc[keep].reset_index(drop=True)

                                    if synthetic_X.empty:
                                        raise ValueError("검증 후 남은 합성 데이터가 없습니다.")

                                    X_res = pd.concat(
                                        [splits["X_train"], synthetic_X],
                                        ignore_index=True,
                                    )
                                    y_syn = pd.Series(
                                        [minority_label] * len(synthetic_X),
                                        name=target_col,
                                    )
                                    y_res = pd.concat(
                                        [y_train.reset_index(drop=True), y_syn],
                                        ignore_index=True,
                                    )

                                    X_res, y_res = shuffle(
                                        X_res,
                                        y_res,
                                        random_state=RANDOM_STATE,
                                    )
                                    X_res = X_res.reset_index(drop=True)
                                    y_res = y_res.reset_index(drop=True)

                                    st.session_state.resampled = {
                                        "X_train": X_res,
                                        "y_train": y_res,
                                        "method": f"Generative AI/OpenAI ({model_name}, ratio={target_ratio:.2f})",
                                        "synthetic_rows": len(synthetic_X),
                                        "synthetic_sample": synthetic_X.head(20),
                                    }
                                    st.session_state.model_results = {}

                                    st.success(
                                        f"Generative AI 오버샘플링 완료: 합성 데이터 {len(synthetic_X):,}행 추가"
                                    )
                                except ImportError:
                                    st.error(
                                        "OpenAI Python SDK가 설치되어 있지 않습니다. "
                                        "requirements.txt에 `openai`를 포함했는지 확인하세요."
                                    )
                                except Exception as e:
                                    st.error(f"Generative AI 오버샘플링 중 오류: {e}")

                    if st.session_state.resampled is not None:
                        rs = st.session_state.resampled

                        st.write(f"**현재 Train 데이터:** {rs['method']}")
                        c1, c2 = st.columns(2)

                        with c1:
                            st.write("오버샘플링 전")
                            st.dataframe(
                                class_count_table(splits["y_train"]),
                                use_container_width=True,
                                hide_index=True,
                            )

                        with c2:
                            st.write("오버샘플링 후")
                            st.dataframe(
                                class_count_table(rs["y_train"]),
                                use_container_width=True,
                                hide_index=True,
                            )

                        if "synthetic_sample" in rs:
                            with st.expander("CTGAN 합성 데이터 예시"):
                                st.dataframe(
                                    rs["synthetic_sample"],
                                    use_container_width=True,
                                )

        with st.expander("전처리 로그"):
            if st.session_state.preprocessing_log:
                for i, item in enumerate(st.session_state.preprocessing_log, start=1):
                    st.write(f"{i}. {item}")
            else:
                st.write("아직 적용된 전처리가 없습니다.")


# =========================================================
# 6. 모델 학습
# =========================================================
elif page == "4. 모델 학습":
    st.title("4. 모델 학습")

    if st.session_state.splits is None:
        st.warning(
            "먼저 데이터 전처리 페이지에서 Target 설정과 "
            "Train / Validation / Test 분할을 완료하세요."
        )
        st.stop()

    splits = st.session_state.splits

    if st.session_state.resampled is None:
        st.info("현재 오버샘플링이 적용되지 않았습니다. 원본 Train set으로 학습합니다.")
    else:
        st.info(f"현재 학습 데이터: {st.session_state.resampled['method']}")

    selected_models = st.multiselect(
        "학습할 모델 선택",
        [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "DNN",
            "Multilayer Perceptron",
        ],
        default=[
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
        ],
    )

    st.subheader("Hyperparameters")

    with st.expander("Logistic Regression"):
        logistic_c = st.number_input(
            "C (Inverse regularization strength)",
            min_value=0.001,
            max_value=1000.0,
            value=1.0,
            format="%.3f",
        )

    with st.expander("Random Forest"):
        rf_n_estimators = st.slider("n_estimators", 50, 1000, 300, 50)
        rf_max_depth = st.slider(
            "max_depth (0=None)",
            0,
            50,
            0,
            1,
        )

    with st.expander("Decision Tree"):
        dt_max_depth = st.slider(
            "Decision Tree max_depth (0=None)",
            0,
            50,
            5,
            1,
        )
        dt_min_samples_split = st.slider(
            "min_samples_split",
            2,
            50,
            2,
            1,
        )

    with st.expander("Multilayer Perceptron"):
        mlp_layers_text = st.text_input(
            "Hidden layers (쉼표 구분)",
            value="128,64",
        )
        mlp_lr = st.number_input(
            "MLP learning rate",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            format="%.5f",
        )
        mlp_max_iter = st.slider(
            "MLP max_iter",
            100,
            2000,
            500,
            100,
        )

    with st.expander("DNN"):
        dnn_epochs = st.slider("DNN epochs", 20, 1000, 200, 20)
        dnn_batch_size = st.selectbox(
            "DNN batch size",
            [16, 32, 64, 128, 256],
            index=2,
        )
        dnn_lr = st.number_input(
            "DNN learning rate",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            format="%.5f",
        )
        dnn_dropout = st.slider(
            "DNN dropout",
            0.0,
            0.7,
            0.2,
            0.05,
        )

    params = {
        "logistic_c": logistic_c,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
        "dt_max_depth": dt_max_depth,
        "dt_min_samples_split": dt_min_samples_split,
        "mlp_layers": parse_hidden_layers(mlp_layers_text),
        "mlp_lr": mlp_lr,
        "mlp_max_iter": mlp_max_iter,
        "dnn_epochs": dnn_epochs,
        "dnn_batch_size": dnn_batch_size,
        "dnn_lr": dnn_lr,
        "dnn_dropout": dnn_dropout,
    }

    if st.button("선택 모델 학습", type="primary"):
        if not selected_models:
            st.warning("최소 1개 이상의 모델을 선택하세요.")
        else:
            train_and_evaluate_selected_models(selected_models, params)

    if st.session_state.model_results:
        st.subheader("학습 완료 모델")
        result_summary = pd.DataFrame(
            [
                {
                    "Model": name,
                    "Sampling": result["sampling"],
                    **{
                        k: round(v, 4) if pd.notna(v) else np.nan
                        for k, v in result["test_metrics"].items()
                    },
                }
                for name, result in st.session_state.model_results.items()
            ]
        )
        st.dataframe(result_summary, use_container_width=True, hide_index=True)


# =========================================================
# 7. 결과 분석
# =========================================================
elif page == "5. 결과 분석":
    st.title("5. 결과 분석")

    results = st.session_state.model_results

    if not results:
        st.warning("먼저 모델 학습 메뉴에서 모델을 학습하세요.")
        st.stop()

    st.subheader("Test Set 성능 비교")

    metrics_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Sampling": result["sampling"],
                **result["test_metrics"],
            }
            for model_name, result in results.items()
        ]
    )

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    display_df = metrics_df.copy()
    display_df[metric_cols] = display_df[metric_cols].round(4)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("성능 지표 시각화")
    melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    fig_metrics = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        range_y=[0, 1],
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.subheader("ROC Curve")

    fig_roc = go.Figure()

    for model_name, result in results.items():
        auc_value = result["test_metrics"]["ROC-AUC"]
        fig_roc.add_trace(
            go.Scatter(
                x=result["roc_fpr"],
                y=result["roc_tpr"],
                mode="lines",
                name=f"{model_name} (AUC={auc_value:.3f})",
            )
        )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Validation Set 성능")
    val_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                **result["validation_metrics"],
            }
            for model_name, result in results.items()
        ]
    )
    val_df[metric_cols] = val_df[metric_cols].round(4)
    st.dataframe(val_df, use_container_width=True, hide_index=True)

    selected_model = st.selectbox(
        "상세 결과를 확인할 모델",
        list(results.keys()),
    )
    selected_result = results[selected_model]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{selected_result['test_metrics']['Accuracy']:.4f}")
    c2.metric("Precision", f"{selected_result['test_metrics']['Precision']:.4f}")
    c3.metric("Recall", f"{selected_result['test_metrics']['Recall']:.4f}")
    c4.metric("F1-score", f"{selected_result['test_metrics']['F1-score']:.4f}")
    c5.metric("ROC-AUC", f"{selected_result['test_metrics']['ROC-AUC']:.4f}")

    st.download_button(
        "성능 결과 CSV 다운로드",
        data=display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="credit_model_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Accuracy만으로 불균형 신용데이터의 성능을 판단하지 말고 "
        "Precision, Recall, F1-score, ROC-AUC를 함께 확인하세요."
    )
