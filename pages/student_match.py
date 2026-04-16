import streamlit as st
import pandas as pd
from io import BytesIO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sheet(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")


def find_name_column(df: pd.DataFrame) -> str | None:
    """Auto-detect the most likely name column."""
    for col in df.columns:
        col_lower = col.lower().strip()
        if any(k in col_lower for k in ["name", "student", "full name", "fullname"]):
            return col
    return None


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def find_common_students(
    enrolled: pd.DataFrame,
    dropped: pd.DataFrame,
    enrolled_col: str,
    dropped_col: str,
) -> pd.DataFrame:
    enrolled_norm = normalize(enrolled[enrolled_col])
    dropped_norm = normalize(dropped[dropped_col])

    common_names = set(enrolled_norm) & set(dropped_norm)

    matched = enrolled[enrolled_norm.isin(common_names)].copy()
    matched = matched.rename(columns={enrolled_col: "Student Name"})
    matched["Student Name"] = matched["Student Name"].str.strip()
    return matched.reset_index(drop=True)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Common Students")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render():
    st.title("🎓 Student Match")
    st.caption("Find students who appear in both the enrolled sheet and the dropped-off sheet.")

    st.markdown(
        """
        **How it works:**
        1. Upload the **Enrolled** sheet (Sheet 1)
        2. Upload the **Dropped-off** sheet (Sheet 2)
        3. Select the name column in each sheet
        4. Click **Find Common Students**

        Supports `.xlsx`, `.xls`, and `.csv` files.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Sheet 1 — Enrolled Students")
        enrolled_file = st.file_uploader(
            "Upload enrolled sheet",
            type=["xlsx", "xls", "csv"],
            key="enrolled_file",
        )

    with col2:
        st.markdown("### 📋 Sheet 2 — Dropped-off Students")
        dropped_file = st.file_uploader(
            "Upload dropped-off sheet",
            type=["xlsx", "xls", "csv"],
            key="dropped_file",
        )

    enrolled_df = None
    dropped_df = None
    enrolled_col = None
    dropped_col = None

    if enrolled_file:
        try:
            enrolled_df = load_sheet(enrolled_file)
            st.success(f"Enrolled sheet loaded — {len(enrolled_df):,} rows, {len(enrolled_df.columns)} columns")
            auto_col = find_name_column(enrolled_df)
            enrolled_col = st.selectbox(
                "Select the student name column (enrolled)",
                options=enrolled_df.columns.tolist(),
                index=enrolled_df.columns.tolist().index(auto_col) if auto_col else 0,
                key="enrolled_col",
            )
            with st.expander("Preview enrolled sheet"):
                st.dataframe(enrolled_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read enrolled sheet: {e}")

    if dropped_file:
        try:
            dropped_df = load_sheet(dropped_file)
            st.success(f"Dropped-off sheet loaded — {len(dropped_df):,} rows, {len(dropped_df.columns)} columns")
            auto_col = find_name_column(dropped_df)
            dropped_col = st.selectbox(
                "Select the student name column (dropped-off)",
                options=dropped_df.columns.tolist(),
                index=dropped_df.columns.tolist().index(auto_col) if auto_col else 0,
                key="dropped_col",
            )
            with st.expander("Preview dropped-off sheet"):
                st.dataframe(dropped_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Could not read dropped-off sheet: {e}")

    ready = (
        enrolled_df is not None
        and dropped_df is not None
        and enrolled_col is not None
        and dropped_col is not None
    )

    if st.button("🔍 Find Common Students", use_container_width=True, disabled=not ready):
        with st.spinner("Matching..."):
            result = find_common_students(enrolled_df, dropped_df, enrolled_col, dropped_col)

        st.divider()

        total_enrolled = len(enrolled_df)
        total_dropped = len(dropped_df)
        total_common = len(result)

        m1, m2, m3 = st.columns(3)
        m1.metric("Enrolled", f"{total_enrolled:,}")
        m2.metric("Dropped-off", f"{total_dropped:,}")
        m3.metric("Common Students Found", f"{total_common:,}")

        if total_common == 0:
            st.info("No common students found. Check that both sheets use the same name format.")
        else:
            st.markdown(f"### ✅ {total_common} Students Found in Both Sheets")
            st.dataframe(result, use_container_width=True)

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "⬇️ Download as Excel",
                    data=to_excel_bytes(result),
                    file_name="common_students.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "⬇️ Download as CSV",
                    data=result.to_csv(index=False).encode(),
                    file_name="common_students.csv",
                    mime="text/csv",
                    use_container_width=True,
                )