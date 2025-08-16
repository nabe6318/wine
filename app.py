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

st.title("🍷 ロジスティック回帰（2特徴 × 多クラス）高冷地先端デモ　信大雑草研作成")
st.write(
    "CSV をアップロードして、**2つの特徴量**と**目的変数**を選び、"
    "ロジスティック回帰で学習・評価・決定境界の可視化＆未知データ予測を行います。"
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
    st.write("※ UCI Wine と同じ並びの場合、"
             "`Color intensity` と `Proline` を使うと、元コードと同じ条件です。")

if uploaded is None:
    st.info("CSV をアップロードしてください。")
    st.stop()

# CSV 読み込み
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
    "特徴量（ちょうど2列）", options=numeric_cols if len(numeric_cols) >= 2 else all_cols,
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

X = df[feat_cols].to_numpy()
y_raw = df[label_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
)

if standardize:
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
else:
    sc = None  # 標準化なし
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
    # ===== 学習 =====
    model.fit(X_train_std, y_train)

    # ===== 精度 =====
    y_train_pred = model.predict(X_train_std)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test_std)
    test_acc = accuracy_score(y_test, y_test_pred)

    st.success(f"学習完了 ✅  |  訓練: **{train_acc:.3f}**  /  テスト: **{test_acc:.3f}**")

    with st.expander("詳しいレポート（テスト）"):
        st.text("Classification report (test):")
        st.text(classification_report(y_test, y_test_pred, target_names=[str(c) for c in le.classes_]))
        st.write("混同行列（test）")
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_test_pred),
                                  index=[f"true_{c}" for c in le.classes_],
                                  columns=[f"pred_{c}" for c in le.classes_]))

    # ===== 決定境界（訓練） =====
    st.subheader("決定境界（訓練データ）")
    fig1 = plt.figure(figsize=(7, 4))
    plot_decision_regions(X_train_std, y_train, clf=model)
    plt.xlabel(feat_cols[0] + (" (std)" if standardize else ""))
    plt.ylabel(feat_cols[1] + (" (std)" if standardize else ""))
    plt.title("Decision Regions - Train")
    st.pyplot(fig1, clear_figure=True)

    # ===== 決定境界（テスト） =====
    st.subheader("決定境界（テストデータ）")
    fig2 = plt.figure(figsize=(7, 4))
    plot_decision_regions(X_test_std, y_test, clf=model)
    plt.xlabel(feat_cols[0] + (" (std)" if standardize else ""))
    plt.ylabel(feat_cols[1] + (" (std)" if standardize else ""))
    plt.title("Decision Regions - Test")
    st.pyplot(fig2, clear_figure=True)

    # ===== 係数と切片 =====
    st.subheader("モデル係数と切片")
    coef_df = pd.DataFrame(model.coef_, columns=[f"{feat_cols[0]}(coef)", f"{feat_cols[1]}(coef)"])
    coef_df.insert(0, "class_index", np.arange(coef_df.shape[0]))
    coef_df["original_label"] = [le.classes_[i] if i < len(le.classes_) else None for i in coef_df["class_index"]]
    st.dataframe(coef_df, use_container_width=True)

    intercept_df = pd.DataFrame({"intercept": model.intercept_})
    intercept_df["class_index"] = np.arange(intercept_df.shape[0])
    intercept_df["original_label"] = [le.classes_[i] if i < len(le.classes_) else None for i in intercept_df["class_index"]]
    st.dataframe(intercept_df, use_container_width=True)

    with st.expander("NumPy配列（print 相当）"):
        st.write("`model.coef_`")
        st.write(model.coef_)
        st.write("`model.intercept_`")
        st.write(model.intercept_)

    # -----------------------------
    # 5) 未知データ 2点の予測
    # -----------------------------
    st.header("5) 未知データ（2サンプル）の予測")

    # 入力補助のために、学習データのmin/max/medianを取得
    train_df = pd.DataFrame(X_train, columns=feat_cols)
    f1_min, f1_max, f1_med = float(train_df[feat_cols[0]].min()), float(train_df[feat_cols[0]].max()), float(train_df[feat_cols[0]].median())
    f2_min, f2_max, f2_med = float(train_df[feat_cols[1]].min()), float(train_df[feat_cols[1]].max()), float(train_df[feat_cols[1]].median())

    st.write("※ 入力は **生の値**（標準化前）を入れてください。必要に応じて内部で同じスケーラーを適用します。")

    c1, c2 = st.columns(2, vertical_alignment="center")
    with c1:
        st.markdown("**サンプル 1**")
        u1_f1 = st.number_input(f"{feat_cols[0]} (sample 1)", value=f1_med, min_value=f1_min, max_value=f1_max, step=(f1_max-f1_min)/100 if f1_max>f1_min else 1.0, format="%.6f")
        u1_f2 = st.number_input(f"{feat_cols[1]} (sample 1)", value=f2_med, min_value=f2_min, max_value=f2_max, step=(f2_max-f2_min)/100 if f2_max>f2_min else 1.0, format="%.6f")
    with c2:
        st.markdown("**サンプル 2**")
        u2_f1 = st.number_input(f"{feat_cols[0]} (sample 2)", value=f1_med, min_value=f1_min, max_value=f1_max, step=(f1_max-f1_min)/100 if f1_max>f1_min else 1.0, format="%.6f")
        u2_f2 = st.number_input(f"{feat_cols[1]} (sample 2)", value=f2_med, min_value=f2_min, max_value=f2_max, step=(f2_max-f2_min)/100 if f2_max>f2_min else 1.0, format="%.6f")

    predict_btn = st.button("未知サンプルを予測", type="primary")

    if predict_btn:
        # 2サンプルを配列化
        unknown_raw = np.array([[u1_f1, u1_f2],
                                [u2_f1, u2_f2]], dtype=float)

        # 学習時と同じ前処理（標準化）
        if standardize and sc is not None:
            unknown_std = sc.transform(unknown_raw)
        else:
            unknown_std = unknown_raw

        # 予測
        pred_idx = model.predict(unknown_std)
        pred_prob = model.predict_proba(unknown_std)

        # ラベル名へ戻す
        pred_label = le.inverse_transform(pred_idx)

        # 結果表
        result_df = pd.DataFrame({
            "sample": ["sample_1", "sample_2"],
            feat_cols[0]: unknown_raw[:,0],
            feat_cols[1]: unknown_raw[:,1],
            "pred_class": pred_label,
            "pred_index": pred_idx
        })

        st.subheader("予測結果（クラス）")
        st.dataframe(result_df, use_container_width=True)

        # 確率も表示（クラスごとに列展開）
        prob_cols = [f"proba_{c}" for c in le.classes_]
        prob_df = pd.DataFrame(pred_prob, columns=prob_cols, index=["sample_1","sample_2"])
        st.subheader("予測確率")
        st.dataframe(prob_df, use_container_width=True)

st.caption("※ 決定境界プロット（mlxtend）は**2次元特徴量のみ対応**です。3列以上は選ばないでください。")

