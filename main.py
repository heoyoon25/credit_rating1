import io
import hashlib
import json
import re
import time
import warnings
import zipfile
from pathlib import Path

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
CHECKPOINT_DIR = Path.cwd() / ".credit_rating_checkpoints"

# OpenAI API standard text-token pricing (USD per 1M tokens).
# Update these values when OpenAI changes API pricing.
OPENAI_MODEL_PRICING = {
    "GPT-5.6 Sol": {
        "id": "gpt-5.6-sol",
        "input_per_million": 5.00,
        "output_per_million": 30.00,
        "description": "최고 성능 / 복잡한 생성 작업",
        "tier": "고성능",
    },
    "GPT-5.6 Terra": {
        "id": "gpt-5.6-terra",
        "input_per_million": 2.00,
        "output_per_million": 12.00,
        "description": "성능과 비용의 균형",
        "tier": "균형형",
    },
    "GPT-5.6 Luna": {
        "id": "gpt-5.6-luna",
        "input_per_million": 0.20,
        "output_per_million": 1.20,
        "description": "5.6 계열의 저비용·대량 생성용",
        "tier": "가성비",
    },
    "GPT-5 mini": {
        "id": "gpt-5-mini",
        "input_per_million": 0.25,
        "output_per_million": 2.00,
        "description": "높은 품질과 낮은 비용의 절충",
        "tier": "가성비",
    },
    "GPT-5 nano": {
        "id": "gpt-5-nano",
        "input_per_million": 0.05,
        "output_per_million": 0.40,
        "description": "매우 저렴한 대량 생성용",
        "tier": "초저비용",
    },
    "GPT-4o mini": {
        "id": "gpt-4o-mini",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
        "description": "검증된 가성비 모델 / 합성 데이터 대량 생성에 적합",
        "tier": "가성비",
    },
    "GPT-4.1 mini": {
        "id": "gpt-4.1-mini",
        "input_per_million": 0.40,
        "output_per_million": 1.60,
        "description": "비추론형 고성능·저비용 모델",
        "tier": "가성비",
    },
}

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


def class_distribution_table(y: pd.Series, stage: str):
    """종속변수 클래스별 개수와 비율을 계산한다."""
    vc = y.value_counts().sort_index()
    total = int(vc.sum())
    rows = []
    for cls, count in vc.items():
        rows.append(
            {
                "구분": stage,
                "클래스": str(cls),
                "개수": int(count),
                "비율(%)": round((int(count) / total * 100) if total else 0.0, 2),
            }
        )
    return pd.DataFrame(rows)


def pair_matching_status(y: pd.Series):
    """이진 종속변수가 정확히 1:1로 균형화되었는지 판정한다."""
    vc = y.value_counts().sort_index()
    if len(vc) != 2:
        return {
            "is_binary": False,
            "is_matched": False,
            "minority": int(vc.min()) if len(vc) else 0,
            "majority": int(vc.max()) if len(vc) else 0,
            "ratio": np.nan,
            "difference": np.nan,
        }

    minority = int(vc.min())
    majority = int(vc.max())
    ratio = minority / majority if majority else np.nan
    difference = majority - minority
    return {
        "is_binary": True,
        "is_matched": minority == majority,
        "minority": minority,
        "majority": majority,
        "ratio": ratio,
        "difference": difference,
    }


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


def detect_identifier_columns(columns):
    """
    API 외부 전송에서 반드시 제외할 직접 식별자/강한 식별자 후보를 변수명으로 탐지한다.

    주의:
    - 단순 substring 'id' 검사는 paid 같은 정상 변수를 오탐할 수 있으므로 사용하지 않는다.
    - 자동 탐지된 변수는 UI에서 해제할 수 없는 보호 변수로 취급한다.
    """
    exact_normalized = {
        # English identifiers
        "id", "userid", "useridentifier", "customerid", "clientid", "memberid",
        "accountid", "loanid", "applicationid", "transactionid", "recordid",
        "name", "fullname", "firstname", "lastname", "middlename", "username",
        "phone", "phonenumber", "mobile", "mobilenumber", "telephone", "tel",
        "address", "homeaddress", "streetaddress", "mailingaddress",
        "email", "emailaddress",
        "ssn", "socialsecuritynumber", "residentregistrationnumber", "rrn",
        "passport", "passportnumber",
        "customernumber", "clientnumber", "membernumber",
        "accountnumber", "bankaccountnumber", "cardnumber", "creditcardnumber",
        "zipcode", "postalcode", "postcode",
        # Korean identifiers
        "아이디", "사용자아이디", "고객아이디", "회원아이디",
        "이름", "성명", "성명정보", "고객명", "회원명",
        "전화번호", "휴대폰번호", "핸드폰번호", "연락처", "휴대전화",
        "주소", "거주지주소", "도로명주소", "상세주소",
        "이메일", "이메일주소", "메일주소",
        "주민등록번호", "주민번호", "외국인등록번호",
        "여권번호", "운전면허번호",
        "고객번호", "회원번호", "계좌번호", "카드번호",
        "우편번호",
    }

    # 이름 자체에 이 토큰이 명확히 포함되면 식별자로 판단한다.
    strong_substrings = (
        "phone", "mobile", "telephone", "email", "address", "passport",
        "socialsecurity", "residentregistration", "accountnumber", "cardnumber",
        "전화번호", "휴대폰", "핸드폰", "연락처", "이메일", "주소",
        "주민등록", "주민번호", "외국인등록", "여권번호", "면허번호",
        "계좌번호", "카드번호", "고객번호", "회원번호",
    )

    protected = []
    for col in columns:
        raw = str(col).strip()
        lower = raw.lower()
        normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", "", lower)
        tokens = [t for t in re.split(r"[^0-9a-zA-Z가-힣]+", lower) if t]

        is_identifier = normalized in exact_normalized

        # id는 독립 토큰 또는 접미/접두 형태일 때만 인식한다.
        if not is_identifier:
            if "id" in tokens or lower.endswith("_id") or lower.startswith("id_"):
                is_identifier = True
            elif normalized.endswith("id") and normalized in {
                "userid", "customerid", "clientid", "memberid", "accountid",
                "loanid", "applicationid", "transactionid", "recordid"
            }:
                is_identifier = True

        if not is_identifier and any(term in normalized for term in strong_substrings):
            is_identifier = True

        # name은 standalone token 또는 명확한 이름 변수에만 적용한다.
        if not is_identifier:
            if "name" in tokens or normalized in {
                "name", "fullname", "firstname", "lastname", "middlename",
                "customername", "clientname", "membername", "username",
                "이름", "성명", "고객명", "회원명"
            }:
                is_identifier = True

        if is_identifier:
            protected.append(col)

    return protected


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


def get_openai_model_price(model_id):
    for display_name, info in OPENAI_MODEL_PRICING.items():
        if info["id"] == model_id:
            return info
    raise ValueError(f"가격 정보가 없는 모델입니다: {model_id}")


def estimate_tokens_from_text(text):
    """UI용 보수적 근사치. 실제 과금 토큰과는 차이가 날 수 있다."""
    if not text:
        return 0
    return max(1, int(np.ceil(len(str(text)) / 4.0)))


def estimate_openai_generation_cost(
    model_id,
    real_minority,
    n_rows,
    batch_size,
    excluded_cols=None,
    include_samples=True,
    sample_rows=10,
):
    """모델 단가와 예상 input/output token을 이용한 생성 전 비용 추정."""
    excluded_cols = excluded_cols or []
    ai_cols = [c for c in real_minority.columns if c not in excluded_cols]
    if not ai_cols or n_rows <= 0:
        return {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    profile = build_minority_profile(real_minority, ai_cols)
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
        safe_sample = real_minority[ai_cols].head(min(sample_rows, len(real_minority)))
        prompt_payload["example_minority_rows"] = safe_sample.to_dict(orient="records")

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

    system_text = (
        "You generate synthetic tabular data for an academic credit-risk "
        "classification experiment. Follow the supplied statistical constraints "
        "and return only data conforming to the schema."
    )
    base_input_text = (
        system_text
        + json.dumps(prompt_payload, ensure_ascii=False, default=str)
        + json.dumps(schema, ensure_ascii=False)
    )
    input_tokens_per_call = estimate_tokens_from_text(base_input_text) + 30
    calls = int(np.ceil(n_rows / max(1, int(batch_size))))
    total_input_tokens = calls * input_tokens_per_call

    # 한 행의 JSON 직렬화 길이를 사용해 출력 토큰을 근사한다.
    prototype = {}
    for c in ai_cols:
        s = pd.to_numeric(real_minority[c], errors="coerce").dropna()
        prototype[c] = float(s.median()) if not s.empty else 0.0
    row_text = json.dumps(prototype, ensure_ascii=False)
    output_tokens_per_row = estimate_tokens_from_text(row_text) + 2
    total_output_tokens = int(n_rows) * output_tokens_per_row + calls * 12

    price = get_openai_model_price(model_id)
    input_cost = total_input_tokens / 1_000_000 * price["input_per_million"]
    output_cost = total_output_tokens / 1_000_000 * price["output_per_million"]

    return {
        "calls": calls,
        "input_tokens": int(total_input_tokens),
        "output_tokens": int(total_output_tokens),
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(input_cost + output_cost),
    }


def calculate_openai_cost_from_usage(model_id, input_tokens, output_tokens):
    price = get_openai_model_price(model_id)
    input_cost = input_tokens / 1_000_000 * price["input_per_million"]
    output_cost = output_tokens / 1_000_000 * price["output_per_million"]
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "input_cost": float(input_cost),
        "output_cost": float(output_cost),
        "total_cost": float(input_cost + output_cost),
    }



def dataframe_signature(df: pd.DataFrame):
    """현재 Train minority 데이터가 같은 작업인지 확인하기 위한 안정적인 해시."""
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    h = hashlib.sha256()
    h.update(hashed)
    h.update("|".join(map(str, df.columns)).encode("utf-8"))
    h.update("|".join(map(str, df.dtypes)).encode("utf-8"))
    return h.hexdigest()


def build_genai_job_id(
    real_minority,
    model_name,
    target_rows,
    batch_size,
    excluded_cols,
    include_samples,
    sample_rows,
    target_col,
    minority_label,
):
    """같은 데이터/설정의 GPT 생성 작업을 식별하는 ID."""
    payload = {
        "data_signature": dataframe_signature(real_minority),
        "model_name": model_name,
        "target_rows": int(target_rows),
        "batch_size": int(batch_size),
        "excluded_cols": sorted(list(excluded_cols or [])),
        "include_samples": bool(include_samples),
        "sample_rows": int(sample_rows),
        "target_col": str(target_col),
        "minority_label": str(minority_label),
        "columns": list(map(str, real_minority.columns)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24], payload


def _checkpoint_paths(job_id):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CHECKPOINT_DIR / f"{job_id}.pkl",
        CHECKPOINT_DIR / f"{job_id}.json",
    )


def save_genai_checkpoint(job_id, synthetic_df, metadata):
    """배치가 끝날 때마다 로컬 디스크에 원자적으로 체크포인트 저장."""
    data_path, meta_path = _checkpoint_paths(job_id)
    metadata = dict(metadata)
    metadata["job_id"] = job_id
    metadata["generated_rows"] = int(len(synthetic_df))
    metadata["updated_at"] = pd.Timestamp.now().isoformat()

    tmp_data = data_path.with_suffix('.pkl.tmp')
    tmp_meta = meta_path.with_suffix('.json.tmp')
    synthetic_df.to_pickle(tmp_data)
    tmp_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    tmp_data.replace(data_path)
    tmp_meta.replace(meta_path)


def load_genai_checkpoint(job_id):
    data_path, meta_path = _checkpoint_paths(job_id)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        synthetic_df = pd.read_pickle(data_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        return {"synthetic_df": synthetic_df, "metadata": metadata}
    except Exception:
        return None


def delete_genai_checkpoint(job_id):
    data_path, meta_path = _checkpoint_paths(job_id)
    for path in (data_path, meta_path):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def build_checkpoint_zip_bytes(job_id, synthetic_df, metadata):
    """사용자가 보관할 수 있는 안전한 ZIP(CSV + JSON) 체크포인트 생성."""
    meta = dict(metadata)
    meta["job_id"] = job_id
    meta["generated_rows"] = int(len(synthetic_df))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            'metadata.json',
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        )
        zf.writestr(
            'synthetic_rows.csv',
            synthetic_df.to_csv(index=False).encode('utf-8-sig'),
        )
    return buffer.getvalue()


def import_checkpoint_zip_bytes(zip_bytes, expected_job_id, expected_columns):
    """사용자가 저장해 둔 ZIP 체크포인트를 다시 읽는다. Pickle은 보안상 업로드받지 않는다."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
        names = set(zf.namelist())
        if 'metadata.json' not in names or 'synthetic_rows.csv' not in names:
            raise ValueError('체크포인트 ZIP에 metadata.json 또는 synthetic_rows.csv가 없습니다.')
        metadata = json.loads(zf.read('metadata.json').decode('utf-8'))
        if metadata.get('job_id') != expected_job_id:
            raise ValueError(
                '현재 데이터/모델/생성 설정과 다른 체크포인트입니다. '
                '체크포인트를 만든 당시의 설정으로 맞춘 뒤 다시 시도하세요.'
            )
        synthetic_df = pd.read_csv(io.BytesIO(zf.read('synthetic_rows.csv')))

    expected_columns = list(expected_columns)
    missing = [c for c in expected_columns if c not in synthetic_df.columns]
    if missing:
        raise ValueError(f'체크포인트에 필요한 변수들이 없습니다: {missing}')
    synthetic_df = synthetic_df[expected_columns].copy()
    return synthetic_df, metadata


def generate_openai_synthetic_rows(
    api_key,
    model_name,
    real_minority,
    n_rows,
    batch_size=20,
    excluded_cols=None,
    include_samples=True,
    sample_rows=10,
    progress_callback=None,
    checkpoint_callback=None,
    resume_df=None,
    initial_input_tokens=0,
    initial_output_tokens=0,
    initial_batch_no=0,
):
    """
    OpenAI Structured Outputs를 이용해 Minority class의 합성 feature 행을 생성.
    기본값은 통계 프로파일 + 비식별 Minority 예시 10행을 전송한다.
    excluded_cols(자동 식별정보 보호 변수 포함)는 API에 절대 전송하지 않고
    Minority 실제값에서 로컬 bootstrap한다.
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

    if resume_df is not None and len(resume_df) > 0:
        resume_df = resume_df.reindex(columns=all_cols).copy().head(int(n_rows))
        generated_batches = [resume_df]
        already_generated = int(len(resume_df))
    else:
        generated_batches = []
        already_generated = 0

    remaining = max(0, int(n_rows) - already_generated)
    batch_no = int(initial_batch_no or 0)
    estimated_batches = int(np.ceil(int(n_rows) / max(1, int(batch_size))))
    total_input_tokens = int(initial_input_tokens or 0)
    total_output_tokens = int(initial_output_tokens or 0)

    if progress_callback is not None:
        progress_callback({
            "stage": "resuming" if already_generated else "starting",
            "batch_no": batch_no,
            "estimated_batches": estimated_batches,
            "generated_rows": already_generated,
            "target_rows": int(n_rows),
            "progress": already_generated / max(1, int(n_rows)),
        })

    while remaining > 0:
        batch_no += 1
        current_n = min(int(batch_size), remaining)
        rows_before_batch = int(n_rows) - remaining

        if progress_callback is not None:
            progress_callback({
                "stage": "requesting",
                "batch_no": batch_no,
                "estimated_batches": estimated_batches,
                "generated_rows": rows_before_batch,
                "target_rows": int(n_rows),
                "progress": rows_before_batch / max(1, int(n_rows)),
            })

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

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

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
        rows_after_batch = int(n_rows) - remaining

        progress_info = {
            "stage": "completed",
            "batch_no": batch_no,
            "estimated_batches": estimated_batches,
            "generated_rows": rows_after_batch,
            "target_rows": int(n_rows),
            "progress": min(1.0, rows_after_batch / max(1, int(n_rows))),
            "input_tokens": int(total_input_tokens),
            "output_tokens": int(total_output_tokens),
        }

        if progress_callback is not None:
            progress_callback(progress_info)

        if checkpoint_callback is not None:
            current_synthetic = pd.concat(generated_batches, ignore_index=True).head(int(n_rows))
            checkpoint_callback(current_synthetic, progress_info)

    synthetic = pd.concat(generated_batches, ignore_index=True).head(int(n_rows)) if generated_batches else pd.DataFrame(columns=all_cols)
    usage_summary = calculate_openai_cost_from_usage(
        model_name, total_input_tokens, total_output_tokens
    )
    usage_summary["completed_batches"] = int(batch_no)
    usage_summary["generated_rows"] = int(len(synthetic))
    return synthetic, usage_summary


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
    else "➖ 오버샘플링: 사용 안 함 (선택사항)"
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

        st.subheader("총 행 수")
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
                    st.markdown("### 오버샘플링")
                    st.warning(
                        "오버샘플링은 Train set에만 적용합니다. "
                        "Validation/Test set은 원래 분포를 유지합니다."
                    )

                    oversampling_method = st.selectbox(
                        "오버샘플링 방법",
                        ["사용 안 함", "SMOTE", "CTGAN", "Generative AI (OpenAI API)"],
                    )

                    if oversampling_method == "사용 안 함":
                        if st.session_state.resampled is not None:
                            st.session_state.resampled = None
                            st.session_state.model_results = {}
                        st.success(
                            "오버샘플링을 적용하지 않습니다. 원본 Train set으로 바로 모델 학습이 가능합니다."
                        )

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

                        model_options = list(OPENAI_MODEL_PRICING.keys())
                        default_model = "GPT-4o mini"
                        model_display = st.selectbox(
                            "GPT 모델 선택",
                            model_options,
                            index=model_options.index(default_model),
                            format_func=lambda name: name,
                            key="openai_model_select",
                        )
                        model_info = OPENAI_MODEL_PRICING[model_display]
                        model_name = model_info["id"]
                        st.caption(
                            f"{model_info['description']} · API 모델 ID: `{model_name}` · "
                            f"Input ${model_info['input_per_million']:.2f}/1M tokens · "
                            f"Output ${model_info['output_per_million']:.2f}/1M tokens"
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
                            100,
                            25,
                            5,
                            key="genai_batch_size",
                        )

                        real_minority = (
                            splits["X_train"]
                            .loc[y_train == minority_label]
                            .reset_index(drop=True)
                        )

                        # 직접 식별정보는 자동 탐지 후 API 전송에서 강제로 제외한다.
                        # 사용자가 실수로 해제할 수 없도록 보호 목록과 추가 제외 목록을 분리한다.
                        protected_identifier_cols = detect_identifier_columns(
                            real_minority.columns.tolist()
                        )
                        optional_exclusion_cols = [
                            c for c in real_minority.columns
                            if c not in protected_identifier_cols
                        ]

                        st.markdown("#### API 개인정보 보호")
                        if protected_identifier_cols:
                            st.success(
                                "다음 식별정보 변수는 자동 보호되어 OpenAI API에 절대 전송되지 않습니다: "
                                + ", ".join(map(str, protected_identifier_cols))
                            )
                        else:
                            st.info(
                                "변수명 기준으로 자동 탐지된 직접 식별정보가 없습니다. "
                                "실제 데이터의 의미상 식별정보가 있다면 아래 추가 제외 목록에서 선택하세요."
                            )

                        extra_excluded_cols = st.multiselect(
                            "추가로 API 전송에서 제외할 변수",
                            optional_exclusion_cols,
                            default=[],
                            key="genai_extra_excluded_cols",
                            help=(
                                "자동 보호된 ID·이름·주소·전화번호·이메일·주민등록번호·"
                                "계좌번호 등은 이 목록과 관계없이 항상 제외됩니다. "
                                "여기서는 그 외 민감하거나 외부 전송을 원하지 않는 변수만 추가 선택하세요."
                            ),
                        )

                        excluded_cols = list(dict.fromkeys(
                            protected_identifier_cols + extra_excluded_cols
                        ))

                        # 기본 동작: 통계정보 + 비식별 Minority 예시 10개
                        include_samples = st.checkbox(
                            "통계정보와 함께 Minority 실제 예시 행도 API에 제공",
                            value=True,
                            key="genai_include_samples",
                            help=(
                                "기본값은 켜짐입니다. 예시 행에는 자동 보호된 식별정보 변수가 "
                                "포함되지 않습니다."
                            ),
                        )
                        sample_rows = min(10, len(real_minority))
                        if include_samples:
                            max_sample_rows = max(1, min(50, len(real_minority)))
                            sample_rows = st.slider(
                                "API에 제공할 예시 행 수",
                                1,
                                max_sample_rows,
                                min(10, max_sample_rows),
                                key="genai_sample_rows",
                            )
                            st.caption(
                                f"기본 설정: Minority 통계정보 + 비식별 예시 {sample_rows}행 · "
                                f"API 전송 변수 {real_minority.shape[1] - len(excluded_cols)}개 · "
                                f"보호/제외 변수 {len(excluded_cols)}개"
                            )
                        else:
                            sample_rows = 0
                            st.caption(
                                "현재 설정: Minority 통계정보만 API에 전송하며 실제 예시 행은 보내지 않습니다."
                            )

                        if n_to_generate > 0 and len(excluded_cols) < real_minority.shape[1]:
                            estimate = estimate_openai_generation_cost(
                                model_id=model_name,
                                real_minority=real_minority,
                                n_rows=n_to_generate,
                                batch_size=batch_size,
                                excluded_cols=excluded_cols,
                                include_samples=include_samples,
                                sample_rows=sample_rows,
                            )

                            st.markdown("#### 예상 API 비용")
                            c_cost1, c_cost2, c_cost3, c_cost4 = st.columns(4)
                            c_cost1.metric("예상 호출 횟수", f"{estimate['calls']:,}회")
                            c_cost2.metric("예상 Input", f"{estimate['input_tokens']:,} tokens")
                            c_cost3.metric("예상 Output", f"{estimate['output_tokens']:,} tokens")
                            c_cost4.metric("예상 총비용", f"${estimate['total_cost']:.4f}")
                            st.caption(
                                f"입력 약 ${estimate['input_cost']:.4f} + "
                                f"출력 약 ${estimate['output_cost']:.4f}. "
                                "이 값은 변수 수와 JSON 길이를 이용한 추정치이며 실제 API 과금액과 다를 수 있습니다."
                            )
                        else:
                            estimate = None

                        remove_duplicates = st.checkbox(
                            "원본 Train과 완전히 동일한 합성행 제거",
                            value=True,
                            key="genai_drop_duplicates",
                        )

                        cost_ack = st.checkbox(
                            "표시된 예상 비용은 추정치이며 실제 API 사용량에 따라 달라질 수 있음을 확인했습니다.",
                            value=False,
                            key="genai_cost_ack",
                        )

                        # -------------------------------------------------
                        # GPT 생성 체크포인트 / 이어하기
                        # -------------------------------------------------
                        job_id, job_payload = build_genai_job_id(
                            real_minority=real_minority,
                            model_name=model_name,
                            target_rows=n_to_generate,
                            batch_size=batch_size,
                            excluded_cols=excluded_cols,
                            include_samples=include_samples,
                            sample_rows=sample_rows,
                            target_col=target_col,
                            minority_label=minority_label,
                        )

                        uploaded_checkpoint = st.file_uploader(
                            "저장해 둔 GPT 체크포인트 ZIP 불러오기 (선택사항)",
                            type=["zip"],
                            key=f"genai_checkpoint_upload_{job_id}",
                            help=(
                                "Streamlit 서버가 완전히 재시작되면 로컬 체크포인트가 사라질 수 있습니다. "
                                "이전에 다운로드한 체크포인트 ZIP을 올리면 해당 지점부터 이어서 생성할 수 있습니다."
                            ),
                        )
                        if uploaded_checkpoint is not None:
                            if st.button("업로드한 체크포인트 적용", key=f"apply_checkpoint_{job_id}"):
                                try:
                                    imported_df, imported_meta = import_checkpoint_zip_bytes(
                                        uploaded_checkpoint.getvalue(),
                                        expected_job_id=job_id,
                                        expected_columns=real_minority.columns.tolist(),
                                    )
                                    imported_df = snap_synthetic_to_training_domain(
                                        imported_df,
                                        real_minority,
                                    ).head(n_to_generate)
                                    imported_meta.update({
                                        "job_payload": job_payload,
                                        "status": "in_progress" if len(imported_df) < n_to_generate else "completed",
                                    })
                                    save_genai_checkpoint(job_id, imported_df, imported_meta)
                                    st.success(
                                        f"체크포인트를 불러왔습니다: {len(imported_df):,} / {n_to_generate:,}행"
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"체크포인트 불러오기 실패: {e}")

                        checkpoint_state = load_genai_checkpoint(job_id)
                        resume_available = False
                        resume_rows = 0
                        if checkpoint_state is not None:
                            cp_df = checkpoint_state["synthetic_df"].head(n_to_generate)
                            cp_meta = checkpoint_state["metadata"]
                            resume_rows = int(len(cp_df))
                            resume_available = 0 < resume_rows < n_to_generate
                            cp_percent = (resume_rows / n_to_generate * 100) if n_to_generate else 100.0

                            if resume_available:
                                st.success(
                                    f"중단된 GPT 생성 작업을 찾았습니다: "
                                    f"{resume_rows:,} / {n_to_generate:,}행 ({cp_percent:.1f}%) 저장됨"
                                )
                            elif resume_rows >= n_to_generate and n_to_generate > 0:
                                st.info(
                                    f"완료된 체크포인트가 있습니다: {resume_rows:,}행. "
                                    "동일 설정으로 후처리를 다시 실행할 수 있습니다."
                                )

                            cp_zip = build_checkpoint_zip_bytes(job_id, cp_df, cp_meta)
                            cp_c1, cp_c2 = st.columns(2)
                            with cp_c1:
                                st.download_button(
                                    "현재 체크포인트 ZIP 다운로드",
                                    data=cp_zip,
                                    file_name=f"gpt_oversampling_checkpoint_{job_id}.zip",
                                    mime="application/zip",
                                    key=f"download_checkpoint_{job_id}_{resume_rows}",
                                    use_container_width=True,
                                )
                            with cp_c2:
                                if st.button(
                                    "현재 체크포인트 삭제",
                                    key=f"delete_checkpoint_{job_id}",
                                    use_container_width=True,
                                ):
                                    delete_genai_checkpoint(job_id)
                                    st.success("체크포인트를 삭제했습니다.")
                                    st.rerun()

                        start_fresh = False
                        if checkpoint_state is not None and resume_rows > 0:
                            start_fresh = st.checkbox(
                                "저장된 체크포인트를 사용하지 않고 처음부터 다시 생성",
                                value=False,
                                key=f"genai_start_fresh_{job_id}",
                            )

                        run_button_label = (
                            "중단 지점부터 GPT 오버샘플링 이어서 실행"
                            if resume_available and not start_fresh
                            else "Generative AI 오버샘플링 실행"
                        )

                        if st.button(run_button_label, key="run_genai"):
                            if not api_key.strip():
                                st.error("OpenAI API Key를 입력하세요.")
                            elif target_ratio <= current_ratio:
                                st.error(
                                    f"목표 비율은 현재 비율({current_ratio:.3f})보다 커야 합니다."
                                )
                            elif n_to_generate <= 0:
                                st.error("현재 설정에서는 추가 생성할 Minority 데이터가 없습니다.")
                            elif not cost_ack:
                                st.error("API 비용 확인란을 체크하세요.")
                            elif len(excluded_cols) == real_minority.shape[1]:
                                st.error("식별정보 보호 및 추가 제외 설정으로 API에 전달할 변수가 남아 있지 않습니다.")
                            else:
                                try:
                                    if start_fresh:
                                        delete_genai_checkpoint(job_id)
                                        checkpoint_state = None

                                    checkpoint_state = load_genai_checkpoint(job_id)
                                    if checkpoint_state is not None:
                                        initial_synthetic = checkpoint_state["synthetic_df"].head(n_to_generate).copy()
                                        initial_meta = checkpoint_state["metadata"]
                                    else:
                                        initial_synthetic = pd.DataFrame(columns=real_minority.columns)
                                        initial_meta = {}

                                    initial_rows = int(len(initial_synthetic))
                                    initial_fraction = initial_rows / max(1, n_to_generate)
                                    initial_input_tokens = int(initial_meta.get("total_input_tokens", 0) or 0)
                                    initial_output_tokens = int(initial_meta.get("total_output_tokens", 0) or 0)
                                    initial_batch_no = int(initial_meta.get("completed_batches", 0) or 0)

                                    progress_bar = st.progress(
                                        initial_fraction,
                                        text=f"생성 준비 중 — {initial_rows:,} / {n_to_generate:,}행",
                                    )
                                    progress_status = st.empty()
                                    progress_metrics = st.empty()
                                    checkpoint_download_placeholder = st.empty()
                                    started_at = time.monotonic()

                                    def _format_seconds(seconds):
                                        seconds = max(0, int(seconds))
                                        minutes, secs = divmod(seconds, 60)
                                        hours, minutes = divmod(minutes, 60)
                                        if hours:
                                            return f"{hours}시간 {minutes}분 {secs}초"
                                        if minutes:
                                            return f"{minutes}분 {secs}초"
                                        return f"{secs}초"

                                    def _update_generation_progress(info):
                                        elapsed = time.monotonic() - started_at
                                        done = int(info.get("generated_rows", 0))
                                        target = int(info.get("target_rows", n_to_generate))
                                        batch_no_now = int(info.get("batch_no", 0))
                                        total_batches_now = int(info.get("estimated_batches", 0))
                                        fraction = float(info.get("progress", 0.0))
                                        fraction = min(1.0, max(0.0, fraction))
                                        percent = int(round(fraction * 100))

                                        newly_done = max(0, done - initial_rows)
                                        rate = newly_done / elapsed if elapsed > 0 and newly_done > 0 else 0.0
                                        eta_seconds = (target - done) / rate if rate > 0 else None
                                        eta_text = _format_seconds(eta_seconds) if eta_seconds is not None else "계산 중"

                                        progress_bar.progress(
                                            fraction,
                                            text=f"생성 진행률 {percent}% — {done:,} / {target:,}행",
                                        )

                                        stage = info.get("stage")
                                        if stage == "requesting":
                                            progress_status.info(
                                                f"API 배치 {batch_no_now:,} 요청 중 "
                                                f"(예상 총 {total_batches_now:,}회) · OpenAI 응답을 기다리고 있습니다."
                                            )
                                        elif stage == "completed":
                                            progress_status.success(
                                                f"배치 {batch_no_now:,} 완료 · 현재까지 {done:,}행 생성"
                                            )
                                        elif stage == "resuming":
                                            progress_status.info(
                                                f"저장된 체크포인트 {done:,}행부터 이어서 생성합니다."
                                            )
                                        else:
                                            progress_status.info("합성 데이터 생성을 시작합니다.")

                                        with progress_metrics.container():
                                            p1, p2, p3, p4 = st.columns(4)
                                            p1.metric("진행률", f"{percent}%")
                                            p2.metric("생성 완료", f"{done:,} / {target:,}행")
                                            p3.metric(
                                                "API 배치",
                                                f"{batch_no_now:,} / {total_batches_now:,}",
                                            )
                                            p4.metric(
                                                "예상 남은 시간",
                                                eta_text if done < target else "완료",
                                            )

                                    def _save_generation_checkpoint(synthetic_so_far, info):
                                        checkpoint_meta = {
                                            "job_payload": job_payload,
                                            "status": "in_progress",
                                            "model_display": model_display,
                                            "model_name": model_name,
                                            "target_rows": int(n_to_generate),
                                            "batch_size": int(batch_size),
                                            "target_ratio": float(target_ratio),
                                            "target_col": str(target_col),
                                            "minority_label": str(minority_label),
                                            "completed_batches": int(info.get("batch_no", 0)),
                                            "total_input_tokens": int(info.get("input_tokens", 0)),
                                            "total_output_tokens": int(info.get("output_tokens", 0)),
                                            "excluded_cols": list(excluded_cols),
                                            "include_samples": bool(include_samples),
                                            "sample_rows": int(sample_rows),
                                        }
                                        save_genai_checkpoint(job_id, synthetic_so_far, checkpoint_meta)

                                        # 현재 생성분을 사용자가 별도로 보관할 수도 있게 매 배치 갱신
                                        zip_bytes = build_checkpoint_zip_bytes(
                                            job_id,
                                            synthetic_so_far,
                                            checkpoint_meta,
                                        )
                                        with checkpoint_download_placeholder.container():
                                            st.download_button(
                                                f"체크포인트 백업 다운로드 ({len(synthetic_so_far):,}행 저장됨)",
                                                data=zip_bytes,
                                                file_name=f"gpt_oversampling_checkpoint_{job_id}.zip",
                                                mime="application/zip",
                                                key=f"running_checkpoint_{job_id}_{len(synthetic_so_far)}",
                                                use_container_width=True,
                                            )

                                    synthetic_X, actual_usage = generate_openai_synthetic_rows(
                                        api_key=api_key.strip(),
                                        model_name=model_name,
                                        real_minority=real_minority,
                                        n_rows=n_to_generate,
                                        batch_size=batch_size,
                                        excluded_cols=excluded_cols,
                                        include_samples=include_samples,
                                        sample_rows=sample_rows,
                                        progress_callback=_update_generation_progress,
                                        checkpoint_callback=_save_generation_checkpoint,
                                        resume_df=initial_synthetic,
                                        initial_input_tokens=initial_input_tokens,
                                        initial_output_tokens=initial_output_tokens,
                                        initial_batch_no=initial_batch_no,
                                    )

                                    completed_meta = {
                                        "job_payload": job_payload,
                                        "status": "completed",
                                        "model_display": model_display,
                                        "model_name": model_name,
                                        "target_rows": int(n_to_generate),
                                        "batch_size": int(batch_size),
                                        "target_ratio": float(target_ratio),
                                        "target_col": str(target_col),
                                        "minority_label": str(minority_label),
                                        "completed_batches": int(actual_usage.get("completed_batches", initial_batch_no)),
                                        "total_input_tokens": int(actual_usage.get("input_tokens", 0)),
                                        "total_output_tokens": int(actual_usage.get("output_tokens", 0)),
                                        "excluded_cols": list(excluded_cols),
                                        "include_samples": bool(include_samples),
                                        "sample_rows": int(sample_rows),
                                    }
                                    save_genai_checkpoint(job_id, synthetic_X, completed_meta)

                                    progress_bar.progress(
                                        1.0,
                                        text=f"생성 완료 — {n_to_generate:,} / {n_to_generate:,}행",
                                    )
                                    progress_status.success(
                                        f"GPT 합성 데이터 생성이 완료되었습니다. 총 소요 시간: "
                                        f"{_format_seconds(time.monotonic() - started_at)}"
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
                                        "method": f"Generative AI/OpenAI ({model_display}, ratio={target_ratio:.2f})",
                                        "synthetic_rows": len(synthetic_X),
                                        "synthetic_sample": synthetic_X.head(20),
                                        "openai_model": model_name,
                                        "estimated_cost": estimate,
                                        "actual_usage": actual_usage,
                                    }
                                    st.session_state.model_results = {}

                                    st.success(
                                        f"Generative AI 오버샘플링 완료: 합성 데이터 {len(synthetic_X):,}행 추가"
                                    )
                                    if actual_usage["input_tokens"] + actual_usage["output_tokens"] > 0:
                                        st.info(
                                            f"이번 실행 API 사용량: Input {actual_usage['input_tokens']:,} tokens / "
                                            f"Output {actual_usage['output_tokens']:,} tokens / "
                                            f"계산상 비용 약 ${actual_usage['total_cost']:.4f}"
                                        )
                                except ImportError:
                                    st.error(
                                        "OpenAI Python SDK가 설치되어 있지 않습니다. "
                                        "requirements.txt에 `openai`를 포함했는지 확인하세요."
                                    )
                                except Exception as e:
                                    saved_state = load_genai_checkpoint(job_id)
                                    if saved_state is not None and len(saved_state["synthetic_df"]) > 0:
                                        saved_n = len(saved_state["synthetic_df"])
                                        st.error(
                                            f"Generative AI 오버샘플링 중 오류: {e}\n\n"
                                            f"체크포인트에 {saved_n:,}행까지 저장되어 있습니다. "
                                            "같은 설정으로 다시 실행하면 저장된 지점부터 이어집니다."
                                        )
                                    else:
                                        st.error(f"Generative AI 오버샘플링 중 오류: {e}")

                    if st.session_state.resampled is not None:
                        rs = st.session_state.resampled

                        st.divider()
                        st.markdown("### 오버샘플링 후 데이터 분포")
                        st.write(f"**현재 Train 데이터:** {rs['method']}")

                        before_dist = class_distribution_table(
                            splits["y_train"], "오버샘플링 전"
                        )
                        after_dist = class_distribution_table(
                            rs["y_train"], "오버샘플링 후"
                        )
                        distribution_df = pd.concat(
                            [before_dist, after_dist], ignore_index=True
                        )

                        before_status = pair_matching_status(splits["y_train"])
                        after_status = pair_matching_status(rs["y_train"])
                        target_name = splits.get("target_col", st.session_state.target_col)
                        positive_class = splits.get("positive_class", None)

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("종속변수", str(target_name))
                        m2.metric(
                            "Positive class (1)",
                            str(positive_class) if positive_class is not None else "1",
                        )
                        m3.metric(
                            "오버샘플링 후 비율",
                            f"{after_status['ratio']:.3f}"
                            if pd.notna(after_status["ratio"])
                            else "N/A",
                            help="Minority / Majority 비율입니다. 1.000이면 정확한 1:1입니다.",
                        )
                        m4.metric(
                            "Pair Matching",
                            "완료 (1:1)" if after_status["is_matched"] else "미완료",
                        )

                        if not after_status["is_binary"]:
                            st.warning(
                                "선택한 종속변수가 현재 이진 클래스가 아니므로 "
                                "1:1 Pair Matching 여부를 판정할 수 없습니다."
                            )
                        elif after_status["is_matched"]:
                            st.success(
                                f"✅ Pair Matching 완료: 종속변수 `{target_name}`의 두 클래스가 "
                                f"각각 {after_status['majority']:,}행으로 정확히 1:1입니다."
                            )
                        else:
                            st.warning(
                                f"⚠️ Pair Matching 미완료: Minority {after_status['minority']:,}행 / "
                                f"Majority {after_status['majority']:,}행으로 "
                                f"{after_status['difference']:,}행 차이가 있습니다. "
                                f"현재 비율은 {after_status['ratio']:.3f}:1입니다."
                            )

                        c1, c2 = st.columns(2)

                        with c1:
                            st.write("**오버샘플링 전 분포**")
                            st.dataframe(
                                before_dist[["클래스", "개수", "비율(%)"]],
                                use_container_width=True,
                                hide_index=True,
                            )

                        with c2:
                            st.write("**오버샘플링 후 분포**")
                            st.dataframe(
                                after_dist[["클래스", "개수", "비율(%)"]],
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.write("**종속변수 분포 비교**")
                        fig_distribution = px.bar(
                            distribution_df,
                            x="클래스",
                            y="개수",
                            color="구분",
                            barmode="group",
                            text="개수",
                            title=f"{target_name} 클래스 분포: 오버샘플링 전 vs 후",
                        )
                        fig_distribution.update_traces(textposition="outside")
                        st.plotly_chart(fig_distribution, use_container_width=True)

                        st.caption(
                            "Pair Matching은 오버샘플링 후 두 클래스의 개수가 정확히 같은 1:1 상태인지 확인합니다. "
                            "목표 Minority / Majority 비율을 1.00으로 설정해야 일반적으로 1:1이 됩니다."
                        )

                        with st.expander("오버샘플링 후 Train 데이터 확인"):
                            resampled_preview = rs["X_train"].copy()
                            resampled_preview[target_name] = rs["y_train"].to_numpy()
                            st.dataframe(
                                resampled_preview.head(100),
                                use_container_width=True,
                            )
                            st.download_button(
                                "오버샘플링 후 Train 데이터 CSV 다운로드",
                                data=resampled_preview.to_csv(index=False).encode("utf-8-sig"),
                                file_name="resampled_train_data.csv",
                                mime="text/csv",
                                key="download_resampled_train",
                            )

                        if "synthetic_sample" in rs:
                            sample_label = (
                                "Generative AI 합성 데이터 예시"
                                if "Generative AI" in rs.get("method", "")
                                else "CTGAN 합성 데이터 예시"
                            )
                            with st.expander(sample_label):
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
        st.success(
            "오버샘플링 미적용 상태입니다. 원본 Train set으로 바로 모델 학습을 진행합니다."
        )
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

