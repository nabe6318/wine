import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions

st.set_page_config(page_title="Wine Logistic Regression Demo", layout="centered")

# ===== セッション初期化 =====
for k, v in {
    "trained": False,
    "clf": None,
    "scaler": None,
    "le": None,
    "feat_cols_trained": None,
    "standardize_trained": None,
    "metrics": None,
    "X_train_std": None,
    "X_test_std": None,
    "y_train": None,
    "y_test": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🍷 ロジスティック回帰（2特徴 × 多クラス）デモ")
st.write(
    "CSV をアップロードして、**2つの特徴量**と**目的変数**を選び、"
    "ロジスティック回帰で学習・評価・決定境界の可視化＆未知サンプル予測を行います。"
)

# -----------------------------
# 1) データ入力
# -----------------------------
st.header("1) データをアップロード")
uploaded = st.file_uploader("CSVファイルを選択（UTF-8想定）", type=["csv"])

sample_hint_expander = st.expander("ワインデータの列名（参考）")
with sample_hint_expander:
    st.code(
        """['Class label', 'Alcohol', 'Malic acid', 'Ash',
 'Alcalinity of ash', 'Magnesium', 'Total phenols',
 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
 'Proline']""",
        language="text",
    )
    st.write("※ UCI Wine と同じ並びの場合、`Color intensity` と `Proline` を使うと元コードと同条件。")

if uploaded is None:
    st.info("CSV をアップロードしてください。")
    st.stop()

# CSV 読み込み（header=0 を試し、失敗したら header=None）
try:
    df = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, header=None)

st.subheader("先頭5行")
st.dataframe(df.head(), use_container_width=True)
st.write("データ形状:", df.shape)

# -----------------------------
# 2) 列の指定
# -----------------------------
st.header("2) 列の指定")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

default_label = "Class label" if "Class label" in df.columns else all_cols[0]
label_col = st.selectbox("目的変数（クラス）列", options=all_cols, index=all_cols.index(default_label))

default_feat1 = "Color intensity" if "Color intensity" in df.columns else (numeric_cols[1] if len(numeric_cols) > 1 else all_cols[1])
default_feat2 = "Proline" if "Proline" in df.columns else (numeric_cols[2] if len(numeric_cols) > 2 else all_cols[2])

feat_cols = st.multiselect(
    "特徴量（ちょうど2列）",
    options=(numeric_cols if len(numeric_cols) >= 2 else all_cols),
    default=[c for c in [default_feat1, default_feat2] if c in all_cols]
)
if len(feat_cols) != 2:
    st.error("特徴量は **2列** だけ選んでください。")
    st.stop()

# 作業用DF（数値化＆NaN除外）
work_df = df[[label_col] + feat_cols].copy()
for c in feat_cols:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
before_drop = len(work_df)
work_df = work_df.dropna(subset=feat_cols)
dropped = before_drop - len(work_df)
if dropped > 0:
    st.warning(f"特徴量に NaN があったため **{dropped} 行** を除外しました。")

# -----------------------------
# 2.5) 特徴量の分布（ヒストグラム）
# -----------------------------
st.header("3) 特徴量の分布（ヒストグラム）")
st.write("※ サンプル値入力の前に、選択した2変数の分布を表示。")
for col in feat_cols:
    vals = pd.to_numeric(work_df[col], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    fig = plt.figure(figsize=(6, 3.5))
    plt.hist(vals, bins=30)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(f"Histogram: {col}")
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# 3) 前処理と分割
# -----------------------------
st.header("4) 前処理とデータ分割")
col_a, col_b, col_c = st.columns(3)
with col_a:
    test_size = st.slider("テストサイズ", 0.1, 0.5, 0.2, 0.05)
with col_b:
    random_state = st.number_input("random_state", min_value=0, max_value=9999, value=0, step=1)
with col_c:
    standardize = st.checkbox("標準化（StandardScaler）を使う", value=True)

X = work_df[feat_cols].to_numpy()
y_raw = work_df[label_col]
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
)

if standardize:
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
else:
    sc = None
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

st.write(
    f"**X_train**: {X_train_std.shape} / **y_train**: {y_train.shape}  |  "
    f"**X_test**: {X_test_std.shape} / **y_test**: {y_test.shape}"
)

# -----------------------------
# 4) モデル設定と学習
# -----------------------------
st.header("5) モデル設定と学習（LogisticRegression）")
col1, col2, col3, col4 = st.columns(4)
with col1:
    C = st.select_slider("正則化強度 C", options=[0.01, 0.1, 0.5, 1.0, 2.0, 10.0], value=1.0)
with col2:
    penalty = st.selectbox("ペナルティ", options=["l2"], index=0)
with col3:
    multi_class = st.selectbox("multi_class", options=["ovr", "multinomial"], index=0)
with col4:
    max_iter = st.number_input("max_iter", min_value=100, max_value=5000, value=200, step=50)

solver = "liblinear" if multi_class == "ovr" else "lbfgs"
model = LogisticRegression(
    max_iter=int(max_iter),
    multi_class=multi_class,
    solver=solver,
    C=float(C),
    penalty=penalty,
    random_state=int(random_state),
)

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    train_button = st.button("学習・評価を実行", type="primary")
with col_btn2:
    reset_button = st.button("🧹 モデルをリセット", type="secondary")

if reset_button:
    for k in ["trained","clf","scaler","le","feat_cols_trained","standardize_trained",
              "metrics","X_train_std","X_test_std","y_train","y_test"]:
        st.session_state[k] = None if k != "trained" else False
    st.success("学習状態をリセットしました。")

if train_button:
    # 学習
    model.fit(X_train_std, y_train)

    # 精度
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # セッションに保存（再実行でも保持）
    st.session_state.trained = True
    st.session_state.clf = model
    st.session_state.scaler = sc
    st.session_state.le = le
    st.session_state.feat_cols_trained = feat_cols.copy()
    st.session_state.standardize_trained = standardize
    st.session_state.metrics = {"train_acc": float(train_acc), "test_acc": float(test_acc)}
    st.session_state.X_train_std = X_train_std
    st.session_state.X_test_std = X_test_std
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

# ===== 学習後の可視化／情報 =====
if st.session_state.trained:
    # 列が変わってないかチェック
    if st.session_state.feat_cols_trained != feat_cols or st.session_state.standardize_trained != standardize:
        st.warning("学習時の設定と異なります。**再学習**してください。")
    else:
        st.success(f"学習完了 ✅ | 訓練: **{st.session_state.metrics['train_acc']:.3f}** / テスト: **{st.session_state.metrics['test_acc']:.3f}**")

        # 決定境界（訓練）
        st.subheader("決定境界（訓練データ）")
        fig1 = plt.figure(figsize=(7, 4))
        plot_decision_regions(st.session_state.X_train_std, st.session_state.y_train, clf=st.session_state.clf)
        plt.xlabel(feat_cols[0] + (" (std)" if st.session_state.standardize_trained else ""))
        plt.ylabel(feat_cols[1] + (" (std)" if st.session_state.standardize_trained else ""))
        plt.title("Decision Regions - Train")
        st.pyplot(fig1, clear_figure=True)

        # 決定境界（テスト）
        st.subheader("決定境界（テストデータ）")
        fig2 = plt.figure(figsize=(7, 4))
        plot_decision_regions(st.session_state.X_test_std, st.session_state.y_test, clf=st.session_state.clf)
        plt.xlabel(feat_cols[0] + (" (std)" if st.session_state.standardize_trained else ""))
        plt.ylabel(feat_cols[1] + (" (std)" if st.session_state.standardize_trained else ""))
        plt.title("Decision Regions - Test")
        st.pyplot(fig2, clear_figure=True)

        # 係数・切片
        st.subheader("モデル係数と切片")
        coef_df = pd.DataFrame(
            st.session_state.clf.coef_,
            columns=[f"{feat_cols[0]}(coef)", f"{feat_cols[1]}(coef)"]
        )
        coef_df.insert(0, "class_index", np.arange(coef_df.shape[0]))
        coef_df["original_label"] = [
            st.session_state.le.classes_[i] if i < len(st.session_state.le.classes_) else None
            for i in coef_df["class_index"]
        ]
        st.dataframe(coef_df, use_container_width=True)

        intercept_df = pd.DataFrame({"intercept": st.session_state.clf.intercept_})
        intercept_df["class_index"] = np.arange(intercept_df.shape[0])
        intercept_df["original_label"] = [
            st.session_state.le.classes_[i] if i < len(st.session_state.le.classes_) else None
        for i in intercept_df["class_index"]]
        st.dataframe(intercept_df, use_container_width=True)

        with st.expander("NumPy配列（print 相当）"):
            st.write("`model.coef_`")
            st.write(st.session_state.clf.coef_)
            st.write("`model.intercept_`")
            st.write(st.session_state.clf.intercept_)

        # -----------------------------
        # 6) 未知データ 1点の予測（学習済みを使用）
        # -----------------------------
        st.header("6) 未知データ（1サンプル）の予測")
        def safe_stats(series: pd.Series):
            vals = pd.to_numeric(series, errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan, np.nan, np.nan
            return np.nanmin(vals), np.nanmax(vals), np.nanmedian(vals)

        def number_input_safe(label: str, default: float, vmin: float, vmax: float):
            kwargs = {"value": float(default) if np.isfinite(default) else 0.0, "format": "%.6f"}
            if np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin):
                step = (vmax - vmin) / 100.0
                if step <= 0 or not np.isfinite(step):
                    step = 1.0
                kwargs.update(min_value=float(vmin), max_value=float(vmax), step=float(step))
            else:
                kwargs.update(step=1.0)
            return st.number_input(label, **kwargs)

        f1_min, f1_max, f1_med = safe_stats(work_df[feat_cols[0]])
        f2_min, f2_max, f2_med = safe_stats(work_df[feat_cols[1]])
        st.write("※ 入力は **生の値**（標準化前）。内部で学習時と同じスケーラーを適用します。")

        u_f1 = number_input_safe(f"{feat_cols[0]} (sample)", f1_med, f1_min, f1_max)
        u_f2 = number_input_safe(f"{feat_cols[1]} (sample)", f2_med, f2_min, f2_max)

        predict_btn = st.button("未知サンプルを予測", type="primary")
        if predict_btn:
            try:
                unknown_raw = np.array([[float(u_f1), float(u_f2)]], dtype=float)
                if st.session_state.standardize_trained and st.session_state.scaler is not None:
                    unknown_std = st.session_state.scaler.transform(unknown_raw)
                else:
                    unknown_std = unknown_raw

                pred_idx = st.session_state.clf.predict(unknown_std)
                pred_prob = st.session_state.clf.predict_proba(unknown_std)
                pred_label = st.session_state.le.inverse_transform(pred_idx)

                result_df = pd.DataFrame({
                    "sample": ["sample_1"],
                    feat_cols[0]: [unknown_raw[0, 0]],
                    feat_cols[1]: [unknown_raw[0, 1]],
                    "pred_class": [pred_label[0]],
                    "pred_index": [int(pred_idx[0])]
                })

                st.subheader("予測結果（クラス）")
                st.dataframe(result_df, use_container_width=True)

                prob_cols = [f"proba_{c}" for c in st.session_state.le.classes_]
                prob_df = pd.DataFrame(pred_prob, columns=prob_cols, index=["sample_1"])
                st.subheader("予測確率")
                st.dataframe(prob_df, use_container_width=True)
            except Exception as e:
                st.error("未知サンプルの予測でエラーが発生しました。")
                st.exception(e)
else:
    st.info("まだ学習していません。「学習・評価を実行」をクリックしてください。")

st.caption("※ 決定境界プロット（mlxtend）は**2次元特徴量のみ対応**です。3列以上は選ばないでください。")
