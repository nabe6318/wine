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

st.title("🍷 ロジスティック回帰（2特徴 × 多クラス）デモ")
st.write(
    "CSV をアップロードして、**2つの特徴量**と**目的変数**を選び、"
    "ロジスティック回帰で学習・評価・決定境界の可視化を行います。"
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
    st.write("※ UCI Wine データと同じ並びの場合、"
             "`Color intensity`（色; 10列）と `Proline`（13列）を使うと、"
             "元コードと同じ条件になります。")

if uploaded is None:
    st.info("CSV をアップロードしてください。")
    st.stop()

# CSV 読み込み（まず header=0 を試し、ダメなら header=None）
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

# 数値列のみ候補にする（特徴量は数値が望ましいため）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
all_cols = df.columns.tolist()

# 目的変数（ラベル）
default_label = "Class label" if "Class label" in df.columns else all_cols[0]
label_col = st.selectbox("目的変数（クラス）列を選択", options=all_cols, index=all_cols.index(default_label))

# 特徴量（2列のみ）
# デフォルトは Wine と同様に Color intensity, Proline があればそれにする
default_feat1 = "Color intensity" if "Color intensity" in df.columns else (numeric_cols[1] if len(numeric_cols) > 1 else all_cols[1])
default_feat2 = "Proline" if "Proline" in df.columns else (numeric_cols[2] if len(numeric_cols) > 2 else all_cols[2])

feat_cols = st.multiselect(
    "特徴量（ちょうど2列を選択）", options=numeric_cols if len(numeric_cols) >= 2 else all_cols,
    default=[c for c in [default_feat1, default_feat2] if c in all_cols]
)

if len(feat_cols) != 2:
    st.error("特徴量は **2列** だけ選んでください。")
    st.stop()

# -----------------------------
# 3) 前処理と分割
# -----------------------------
st.header("3) 前処理とデータ分割")

col_a, col_b, col_c = st.columns(3)
with col_a:
    test_size = st.slider("テストサイズ", 0.1, 0.5, 0.2, 0.05)
with col_b:
    random_state = st.number_input("random_state", min_value=0, max_value=9999, value=0, step=1)
with col_c:
    standardize = st.checkbox("標準化（StandardScaler）を使う", value=True)

# X, y 準備
X = df[feat_cols].to_numpy()
y_raw = df[label_col]

# y を 0..K-1 の整数にエンコード（元のコードの「-1」相当の整形を安全に）
# 文字列や1,2,3以外でも OK にする
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 学習/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
)

# 標準化
if standardize:
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
else:
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()

st.write(
    f"**X_train**: {X_train_std.shape} / **y_train**: {y_train.shape}  |  "
    f"**X_test**: {X_test_std.shape} / **y_test**: {y_test.shape}"
)

# -----------------------------
# 4) モデル設定と学習
# -----------------------------
st.header("4) モデル設定と学習（LogisticRegression）")

col1, col2, col3, col4 = st.columns(4)
with col1:
    C = st.select_slider("正則化強度 C", options=[0.01, 0.1, 0.5, 1.0, 2.0, 10.0], value=1.0)
with col2:
    penalty = st.selectbox("ペナルティ", options=["l2"], index=0)
with col3:
    multi_class = st.selectbox("multi_class", options=["ovr", "multinomial"], index=0)
with col4:
    max_iter = st.number_input("max_iter", min_value=100, max_value=5000, value=200, step=50)

# solver は liblinear（ovr & l2 と相性良）
solver = "liblinear" if multi_class == "ovr" else "lbfgs"

model = LogisticRegression(
    max_iter=int(max_iter),
    multi_class=multi_class,
    solver=solver,
    C=float(C),
    penalty=penalty,
    random_state=int(random_state),
)

train_button = st.button("学習・評価を実行", type="primary")

if train_button:
    # 学習
    model.fit(X_train_std, y_train)

    # 学習データ精度
    y_train_pred = model.predict(X_train_std)
    train_acc = accuracy_score(y_train, y_train_pred)

    # テストデータ精度
    y_test_pred = model.predict(X_test_std)
    test_acc = accuracy_score(y_test, y_test_pred)

    st.success(f"学習完了 ✅  |  訓練正解率: **{train_acc:.3f}**  /  テスト正解率: **{test_acc:.3f}**")

    with st.expander("詳しいレポート（テスト）"):
        st.text("Classification report (test):")
        st.text(classification_report(y_test, y_test_pred, target_names=[str(c) for c in le.classes_]))
        st.write("混同行列（test）")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_test_pred),
                                  index=[f"true_{c}" for c in le.classes_],
                                  columns=[f"pred_{c}" for c in le.classes_]))

    # 決定境界（訓練）
    st.subheader("決定境界（訓練データ）")
    fig1 = plt.figure(figsize=(7, 4))
    plot_decision_regions(X_train_std, y_train, clf=model)
    plt.xlabel(feat_cols[0] + (" (std)" if standardize else ""))
    plt.ylabel(feat_cols[1] + (" (std)" if standardize else ""))
    plt.title("Decision Regions - Train")
    st.pyplot(fig1, clear_figure=True)

    # 決定境界（テスト）
    st.subheader("決定境界（テストデータ）")
    fig2 = plt.figure(figsize=(7, 4))
    plot_decision_regions(X_test_std, y_test, clf=model)
    plt.xlabel(feat_cols[0] + (" (std)" if standardize else ""))
    plt.ylabel(feat_cols[1] + (" (std)" if standardize else ""))
    plt.title("Decision Regions - Test")
    st.pyplot(fig2, clear_figure=True)

    # 係数と切片（クラスごと）
    st.subheader("モデル係数と切片")
    coef_df = pd.DataFrame(model.coef_, columns=[f"{feat_cols[0]}(coef)", f"{feat_cols[1]}(coef)"])
    coef_df.insert(0, "class_index", np.arange(coef_df.shape[0]))
    coef_df["class_label (encoded)"] = coef_df["class_index"]
    coef_df["original_label"] = [le.classes_[i] if i < len(le.classes_) else None for i in coef_df["class_index"]]
    st.dataframe(coef_df, use_container_width=True)

    intercept_df = pd.DataFrame({"intercept": model.intercept_})
    intercept_df["class_index"] = np.arange(intercept_df.shape[0])
    intercept_df["original_label"] = [le.classes_[i] if i < len(le.classes_) else None for i in intercept_df["class_index"]]
    st.dataframe(intercept_df, use_container_width=True)

    with st.expander("NumPy配列で出力（print と同等）"):
        st.write("`model.coef_`")
        st.write(model.coef_)
        st.write("`model.intercept_`")
        st.write(model.intercept_)

st.caption("※ 決定境界プロット（mlxtend）は**2次元特徴量のみ対応**です。3列以上は選ばないでください。")
