# ============================================================
# pages/p3_action.py — Action Plan & Simulator
# ============================================================
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from i18n import t
from utils.helpers import safe_cats, status_display_options


def render(df: pd.DataFrame, model, feature_names: list) -> None:
    st.title(t("page_action"))
    st.caption(t("p3_caption"))

    # ── Target selection ──────────────────────────────────────
    display_list, to_internal, _ = status_display_options(t)
    # ตัด Active ออก (ไม่สมเหตุสมผลที่จะ target)
    risk_display_opts = [d for d in display_list if d != t("status_active")]

    with st.expander(t("p3_target_exp"), expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            sel_display = st.multiselect(
                t("p3_risk_seg"),
                risk_display_opts,
                default=risk_display_opts[:2],
            )
            sel_statuses = [to_internal[s] for s in sel_display if s in to_internal]
        with f2:
            sel_cats = st.multiselect(t("cat_label"), safe_cats(df), key="p3_cat")

    df_p3 = df.copy()
    if sel_statuses: df_p3 = df_p3[df_p3["status"].isin(sel_statuses)]
    if sel_cats:     df_p3 = df_p3[df_p3["product_category_name"].isin(sel_cats)]

    filter_msg = (
        ", ".join(sel_display[:2]) + ("..." if len(sel_display) > 2 else "")
        if sel_display else t("p3_all_groups")
    )
    total_pop = len(df_p3)
    avg_ltv   = float(df_p3["payment_value"].mean()) if "payment_value" in df_p3.columns else 150.0

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: st.info(t("p3_analyzing", g=filter_msg))
    with c2: st.metric(t("p3_target_pop"), f"{total_pop:,}{t('people_unit')}")
    with c3: st.metric(t("p3_avg_ltv"),    f"R$ {avg_ltv:,.0f}")
    st.markdown("---")

    # ── 4 Tabs ────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([t("p3_tab1"), t("p3_tab2"), t("p3_tab3"), t("p3_tab4")])

    with tab1:
        st.subheader(t("p3_t1_title"))
        if "freight_ratio" not in df_p3.columns:
            st.error(t("p3_no_freight"))
        else:
            target = df_p3[df_p3["freight_ratio"] > 0.2].copy()
            avg_fr = float(target["freight_value"].mean()) if not target.empty and "freight_value" in target.columns else 15.0
            _run_simulation(
                target, {"freight_value": ("set", 0), "freight_ratio": ("set", 0)},
                avg_fr, "tab1", t("p3_t1_strategy"), t("p3_t1_rec", avg=avg_fr),
                total_pop, avg_ltv, model, feature_names,
            )

    with tab2:
        st.subheader(t("p3_t2_title"))
        disc_pct = st.radio(t("p3_t2_disc"), [10, 20], horizontal=True, key="disc_t2")
        if "price" not in df_p3.columns:
            st.error(t("p3_no_price"))
        else:
            target = df_p3[df_p3["churn_probability"] > 0.5].copy()
            _run_simulation(
                target,
                {"price": ("multiply", 1 - disc_pct/100), "payment_value": ("multiply", 1 - disc_pct/100)},
                float(avg_ltv * disc_pct / 100), "tab2",
                t("p3_t2_strategy", d=disc_pct), t("p3_t2_rec", d=disc_pct),
                total_pop, avg_ltv, model, feature_names,
            )

    with tab3:
        st.subheader(t("p3_t3_title"))
        if "delay_days" not in df_p3.columns:
            st.error(t("p3_no_delay"))
        else:
            target = df_p3[df_p3["delay_days"] > 0].copy()
            _run_simulation(
                target, {"delay_days": ("set", 0), "delivery_vs_estimated": ("clip_upper", 0)},
                15.0, "tab3", t("p3_t3_strategy"), t("p3_t3_rec"),
                total_pop, avg_ltv, model, feature_names,
            )

    with tab4:
        st.subheader(t("p3_t4_title"))
        if "cat_churn_risk" not in df_p3.columns:
            st.error(t("p3_no_cat_risk"))
        else:
            target = df_p3[df_p3["cat_churn_risk"] > 0.8].copy()
            _run_simulation(
                target,
                {"cat_churn_risk": ("multiply", 0.6), "payment_installments": ("add", 2)},
                10.0, "tab4", t("p3_t4_strategy"), t("p3_t4_rec"),
                total_pop, avg_ltv, model, feature_names,
            )


# ── Simulation engine ─────────────────────────────────────────

def _run_simulation(
    target_df: pd.DataFrame,
    feature_changes: dict,
    cost_per_head: float,
    tab_key: str,
    strategy_name: str,
    rec_text: str,
    total_pop: int,
    avg_ltv: float,
    model,
    feature_names: list,
) -> None:
    n_target    = len(target_df)
    pct_problem = (n_target / total_pop * 100) if total_pop > 0 else 0

    c_prob, c_sol, c_res = st.columns([1, 1.3, 1])

    with c_prob:
        st.info(t("p3_problem", n=f"{n_target:,}", pct=pct_problem))
        st.progress(min(pct_problem / 100, 1.0))
        if not target_df.empty:
            st.markdown(t("p3_feat_avg"))
            for col in list(feature_changes.keys())[:3]:
                if col in target_df.columns:
                    st.caption(f"• {col}: {target_df[col].mean():.2f}")

    with c_sol:
        st.markdown(t("p3_strategy", name=strategy_name))
        st.write(rec_text)
        st.markdown("---")
        cost = st.number_input(
            t("p3_cost_lbl"), value=float(cost_per_head),
            min_value=0.0, max_value=500.0, step=0.5, key=f"cost_{tab_key}",
        )
        be_rate = cost / avg_ltv if avg_ltv > 0 else 0
        st.caption(t("p3_breakeven", r=be_rate))

        if model is None or not feature_names:
            max_pot   = 15
            realistic = min(max_pot, 10) if cost >= 15 else min(max_pot, 5)
            st.caption(t("p3_no_model"))
            lift = st.slider(t("p3_lift_lbl"), 1, 100, realistic, key=f"lift_{tab_key}")
            sim_success_rate = lift / 100
            sim_mode = "manual"
        else:
            sim_mode = "model"
            lift     = None

    with c_res:
        with st.spinner(t("p3_simulating")):
            time.sleep(0.3)

            if sim_mode == "model" and not target_df.empty:
                sim_success_rate = _model_uplift(target_df, feature_changes, model, feature_names)
                _render_uplift_chart(target_df, feature_changes, model, feature_names)
            else:
                sim_success_rate = lift / 100 if lift else 0.1

            _render_roi(n_target, sim_success_rate, cost, avg_ltv, be_rate)


def _apply_feature_changes(df_sim: pd.DataFrame, feature_changes: dict) -> pd.DataFrame:
    for col, (op, val) in feature_changes.items():
        if col in df_sim.columns:
            if op == "set":          df_sim[col] = val
            elif op == "multiply":   df_sim[col] = df_sim[col] * val
            elif op == "clip_upper": df_sim[col] = df_sim[col].clip(upper=val)
            elif op == "add":        df_sim[col] = df_sim[col] + val
    # recalc freight_ratio ถ้ามีการเปลี่ยน freight หรือ price
    if "freight_value" in df_sim.columns and "price" in df_sim.columns:
        df_sim["freight_ratio"] = (df_sim["freight_value"] / df_sim["price"].replace(0, np.nan)).fillna(0)
    return df_sim


def _model_uplift(target_df, feature_changes, model, feature_names) -> float:
    X_orig    = target_df.reindex(columns=feature_names, fill_value=0).fillna(0)
    prob_orig = model.predict_proba(X_orig)[:, 1]

    df_sim    = _apply_feature_changes(target_df.copy(), feature_changes)
    X_sim     = df_sim.reindex(columns=feature_names, fill_value=0).fillna(0)
    prob_sim  = model.predict_proba(X_sim)[:, 1]

    uplift_arr = prob_orig - prob_sim
    return (uplift_arr > 0.08).mean()


def _render_uplift_chart(target_df, feature_changes, model, feature_names) -> None:
    X_orig    = target_df.reindex(columns=feature_names, fill_value=0).fillna(0)
    prob_orig = model.predict_proba(X_orig)[:, 1]
    df_sim    = _apply_feature_changes(target_df.copy(), feature_changes)
    X_sim     = df_sim.reindex(columns=feature_names, fill_value=0).fillna(0)
    prob_sim  = model.predict_proba(X_sim)[:, 1]
    uplift    = prob_orig - prob_sim

    dist = {
        t("p3_resp_high"): int((uplift > 0.15).sum()),
        t("p3_resp_mid"):  int(((uplift > 0.08) & (uplift <= 0.15)).sum()),
        t("p3_resp_low"):  int(((uplift > 0) & (uplift <= 0.08)).sum()),
        t("p3_resp_none"): int((uplift <= 0).sum()),
    }
    dist_df = pd.DataFrame({"Group": list(dist.keys()), "Count": list(dist.values())})
    st.altair_chart(
        alt.Chart(dist_df).mark_bar().encode(
            x=alt.X("Group", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count"),
            color=alt.Color("Group",
                scale=alt.Scale(domain=list(dist.keys()), range=["#2ecc71","#f1c40f","#e67e22","#95a5a6"]),
                legend=None,
            ),
            tooltip=["Group", "Count"],
        ).properties(height=160, title=t("p3_uplift_chart")),
        use_container_width=True,
    )


def _render_roi(n_target, sim_success_rate, cost, avg_ltv, be_final) -> None:
    budget      = n_target * cost
    saved_users = int(n_target * sim_success_rate)
    revenue     = saved_users * avg_ltv
    profit      = revenue - budget
    roi         = (profit / budget * 100) if budget > 0 else 0

    st.markdown(t("p3_results"))
    st.metric(t("p3_success"),  f"{sim_success_rate:.1%}", delta=t("p3_be_delta", r=be_final))
    st.metric(t("p3_saved"),    f"{saved_users:,}{t('people_unit')}")
    st.metric(t("p3_budget"),   f"R$ {budget:,.0f}")

    if profit > 0:
        st.metric(t("p3_profit"), f"R$ {profit:,.0f}", f"+{roi:.1f}%")
        st.success(t("p3_worthit"))
    else:
        st.metric(t("p3_loss"), f"R$ {profit:,.0f}", f"{roi:.1f}%")
        gap = be_final - sim_success_rate
        st.error(t("p3_not_worth", be=be_final, sr=sim_success_rate, gap=gap))
        st.caption(t("p3_reduce_cost", c=avg_ltv * sim_success_rate))
