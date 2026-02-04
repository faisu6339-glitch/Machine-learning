import streamlit as st

st.title("Learning st.columns() – Button Color Change")

# Initialize session state
if "col1_clicked" not in st.session_state:
    st.session_state.col1_clicked = False

if "col2_clicked" not in st.session_state:
    st.session_state.col2_clicked = False


# Create columns
col1, col2 = st.columns(2)

# ---------- COLUMN 1 ----------
with col1:
    bg1 = "#90EE90" if st.session_state.col1_clicked else "#F0F0F0"

    st.markdown(
        f"""
        <div style="background-color:{bg1};
                    padding:20px;
                    border-radius:10px;">
            <h3>Column 1</h3>
            <p>This is the left column</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Button 1"):
        st.session_state.col1_clicked = not st.session_state.col1_clicked


# ---------- COLUMN 2 ----------
with col2:
    bg2 = "#ADD8E6" if st.session_state.col2_clicked else "#F0F0F0"

    st.markdown(
        f"""
        <div style="background-color:{bg2};
                    padding:20px;
                    border-radius:10px;">
            <h3>Column 2</h3>
            <p>This is the right column</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Button 2"):
        st.session_state.col2_clicked = not st.session_state.col2_clicked
