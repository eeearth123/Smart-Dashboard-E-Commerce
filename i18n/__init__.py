# ============================================================
# i18n/__init__.py
# ============================================================
import streamlit as st
from i18n.translations import TRANSLATIONS


def t(key: str, **kwargs) -> str:
    """
    ดึงข้อความตามภาษาที่เลือกใน session_state
    รองรับ format string เช่น  t("data_loaded", n="1,234")
    """
    lang = st.session_state.get("lang", "th")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["th"]).get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def render_lang_toggle() -> None:
    """วาดปุ่ม 🇹🇭 / 🇬🇧 ใน sidebar — เรียกจาก app.py เท่านั้น"""
    if "lang" not in st.session_state:
        st.session_state.lang = "th"

    st.sidebar.markdown(f"**{t('language')}**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button(
            "🇹🇭 ไทย",
            use_container_width=True,
            type="primary" if st.session_state.lang == "th" else "secondary",
            key="lang_th",
        ):
            st.session_state.lang = "th"
            st.rerun()
    with col2:
        if st.button(
            "🇬🇧 English",
            use_container_width=True,
            type="primary" if st.session_state.lang == "en" else "secondary",
            key="lang_en",
        ):
            st.session_state.lang = "en"
            st.rerun()
    st.sidebar.markdown("---")
