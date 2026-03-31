from pathlib import Path

def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f'missing pattern for {label}')
    return text.replace(old, new, 1)

# quant_store.py
path = Path('quant_store.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """def _live_archive_validation_summary(live_archive: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'games': 0,
        'settled_days': 0,
        'log_loss': None,
        'accuracy': None,
        'date_span': 'N/A',
        'latest_settled_date': None,
    }
""",
    """def _live_archive_validation_summary(live_archive: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        'games': 0,
        'settled_days': 0,
        'log_loss': None,
        'accuracy': None,
        'date_span': 'N/A',
        'latest_settled_date': None,
    }
""",
    'anchor live archive summary',
)
insert_after = """    return summary


def _run_snapshot(report_date: dt.date, phase: str, weights: Dict[str, Any], state: Dict[str, Any], predictions: List[Dict[str, Any]], validation: Dict[str, Any], live_validation: Dict[str, Any], benchmark_ladder: Dict[str, Any], market_proof: Dict[str, Any], warnings: List[str], report_path: str, workbook_path: str, run_id: str) -> pd.DataFrame:
"""
helper = """    return summary


def _market_proof_lookup(market_proof: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    by_market = (market_proof or {}).get('by_market') or []
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in by_market:
        key = str((row or {}).get('market_type') or '').lower().strip()
        if key:
            lookup[key] = dict(row or {})
    return lookup


def _market_proof_frames(report_date: dt.date, run_id: str, market_proof: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_iso = report_date.isoformat()
    base_summary = {
        'run_id': run_id,
        'as_of_report_date': report_iso,
        'available': bool((market_proof or {}).get('available')),
        'settled_bets': _safe_int((market_proof or {}).get('settled_bets')),
        'units': _safe_float((market_proof or {}).get('units')),
        'roi': _safe_float((market_proof or {}).get('roi')),
        'avg_clv': _safe_float((market_proof or {}).get('avg_clv')),
    }
    summary_frame = pd.DataFrame([base_summary])
    by_market_rows = []
    for row in ((market_proof or {}).get('by_market') or []):
        by_market_rows.append({
            'run_id': run_id,
            'as_of_report_date': report_iso,
            'market_type': str((row or {}).get('market_type') or ''),
            'bets': _safe_int((row or {}).get('bets')),
            'units': _safe_float((row or {}).get('units')),
            'roi': _safe_float((row or {}).get('roi')),
            'avg_clv': _safe_float((row or {}).get('avg_clv')),
        })
    by_market_frame = pd.DataFrame(by_market_rows)
    return summary_frame, by_market_frame


def _run_snapshot(report_date: dt.date, phase: str, weights: Dict[str, Any], state: Dict[str, Any], predictions: List[Dict[str, Any]], validation: Dict[str, Any], live_validation: Dict[str, Any], benchmark_ladder: Dict[str, Any], market_proof: Dict[str, Any], warnings: List[str], report_path: str, workbook_path: str, run_id: str) -> pd.DataFrame:
"""
text = replace_once(text, insert_after, helper, 'market proof helpers')
text = replace_once(
    text,
    """    probability_shrink = ((state.get('probability_shrinkage') or {}).get(phase) or {})
    total_shrink = ((state.get('total_shrinkage') or {}).get(phase) or {})
    margin_shrink = ((state.get('margin_shrinkage') or {}).get(phase) or {})
    row = {
""",
    """    probability_shrink = ((state.get('probability_shrinkage') or {}).get(phase) or {})
    total_shrink = ((state.get('total_shrinkage') or {}).get(phase) or {})
    margin_shrink = ((state.get('margin_shrinkage') or {}).get(phase) or {})
    market_proof_lookup = _market_proof_lookup(market_proof)
    moneyline_proof = market_proof_lookup.get('moneyline') or {}
    total_proof = market_proof_lookup.get('total') or {}
    run_line_proof = market_proof_lookup.get('run_line') or {}
    row = {
""",
    'run snapshot market proof lookup',
)
text = replace_once(
    text,
    """        'market_proof_available': bool(market_proof.get('available')),
        'market_proof_settled_bets': _safe_int(market_proof.get('settled_bets')),
        'market_proof_roi': _safe_float(market_proof.get('roi')),
        'market_proof_avg_clv': _safe_float(market_proof.get('avg_clv')),
        'warnings_json': _safe_json(warnings or []),
""",
    """        'market_proof_available': bool(market_proof.get('available')),
        'market_proof_settled_bets': _safe_int(market_proof.get('settled_bets')),
        'market_proof_roi': _safe_float(market_proof.get('roi')),
        'market_proof_avg_clv': _safe_float(market_proof.get('avg_clv')),
        'market_proof_moneyline_bets': _safe_int(moneyline_proof.get('bets')),
        'market_proof_moneyline_units': _safe_float(moneyline_proof.get('units')),
        'market_proof_moneyline_roi': _safe_float(moneyline_proof.get('roi')),
        'market_proof_moneyline_avg_clv': _safe_float(moneyline_proof.get('avg_clv')),
        'market_proof_total_bets': _safe_int(total_proof.get('bets')),
        'market_proof_total_units': _safe_float(total_proof.get('units')),
        'market_proof_total_roi': _safe_float(total_proof.get('roi')),
        'market_proof_total_avg_clv': _safe_float(total_proof.get('avg_clv')),
        'market_proof_run_line_bets': _safe_int(run_line_proof.get('bets')),
        'market_proof_run_line_units': _safe_float(run_line_proof.get('units')),
        'market_proof_run_line_roi': _safe_float(run_line_proof.get('roi')),
        'market_proof_run_line_avg_clv': _safe_float(run_line_proof.get('avg_clv')),
        'warnings_json': _safe_json(warnings or []),
""",
    'run snapshot market proof fields',
)
text = replace_once(
    text,
    """    policy_snapshot = policy_research.build_policy_research_snapshot(
        live_archive,
        bet_journal,
        phase,
        run_id,
        report_date,
    )
""",
    """    policy_snapshot = policy_research.build_policy_research_snapshot(
        live_archive,
        bet_journal,
        phase,
        run_id,
        report_date,
    )
    market_proof_summary_frame, market_proof_by_market_frame = _market_proof_frames(report_date, run_id, market_proof)
""",
    'export market proof frames',
)
text = replace_once(
    text,
    """        _append_table(con, 'policy_research_market_type_recommendations', policy_snapshot.get('market_type_recommendations'))
        _append_table(con, 'policy_research_window_rankings', policy_snapshot.get('window_rankings'))
        _append_table(con, 'policy_research_recommendations', policy_recommendations)
        _sync_reference_tables(con)
""",
    """        _append_table(con, 'policy_research_market_type_recommendations', policy_snapshot.get('market_type_recommendations'))
        _append_table(con, 'policy_research_window_rankings', policy_snapshot.get('window_rankings'))
        _append_table(con, 'policy_research_recommendations', policy_recommendations)
        _append_table(con, 'market_proof_summary', market_proof_summary_frame)
        _append_table(con, 'market_proof_by_market', market_proof_by_market_frame)
        _sync_reference_tables(con)
""",
    'append market proof tables',
)
text = replace_once(
    text,
    """        'policy_research_market_type_history': pd.DataFrame(),
        'model_state': _load_json(MODEL_STATE_PATH),
    }
""",
    """        'policy_research_market_type_history': pd.DataFrame(),
        'market_proof_summary': pd.DataFrame(),
        'market_proof_by_market': pd.DataFrame(),
        'market_proof_by_market_history': pd.DataFrame(),
        'model_state': _load_json(MODEL_STATE_PATH),
    }
""",
    'bundle defaults market proof',
)
text = replace_once(
    text,
    """            'policy_research_market_type_history': query_quant('SELECT * FROM policy_research_market_type_recommendations ORDER BY as_of_report_date DESC, window_days ASC NULLS LAST, market_type, run_id DESC'),
        })
""",
    """            'policy_research_market_type_history': query_quant('SELECT * FROM policy_research_market_type_recommendations ORDER BY as_of_report_date DESC, window_days ASC NULLS LAST, market_type, run_id DESC'),
            'market_proof_summary': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM market_proof_summary WHERE run_id = ?', [run_id]),
            'market_proof_by_market': pd.DataFrame() if run_id is None else query_quant('SELECT * FROM market_proof_by_market WHERE run_id = ? ORDER BY market_type', [run_id]),
            'market_proof_by_market_history': query_quant('SELECT * FROM market_proof_by_market ORDER BY as_of_report_date DESC, market_type, run_id DESC'),
        })
""",
    'bundle load market proof',
)
path.write_text(text, encoding='utf-8')

# quant_dashboard.py
path = Path('quant_dashboard.py')
text = path.read_text(encoding='utf-8')
text = replace_once(
    text,
    """    policy_research_market_policy_daily = _frame_or_empty(bundle.get('policy_research_market_policy_daily'))
    policy_research_market_policy_summary = _frame_or_empty(bundle.get('policy_research_market_policy_summary'))
    policy_research_market_type_history = _frame_or_empty(bundle.get('policy_research_market_type_history'))
    model_state = bundle.get('model_state') or {}
""",
    """    policy_research_market_policy_daily = _frame_or_empty(bundle.get('policy_research_market_policy_daily'))
    policy_research_market_policy_summary = _frame_or_empty(bundle.get('policy_research_market_policy_summary'))
    policy_research_market_type_history = _frame_or_empty(bundle.get('policy_research_market_type_history'))
    market_proof_summary_bundle = _frame_or_empty(bundle.get('market_proof_summary'))
    market_proof_by_market_bundle = _frame_or_empty(bundle.get('market_proof_by_market'))
    market_proof_by_market_history = _frame_or_empty(bundle.get('market_proof_by_market_history'))
    model_state = bundle.get('model_state') or {}
""",
    'dashboard bundle market proof',
)
text = replace_once(
    text,
    """def _build_market_realized_summary(bet_journal: pd.DataFrame):
    if bet_journal.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = {'report_date', 'result_units'}
    if not required_columns.issubset(bet_journal.columns):
        return pd.DataFrame(), pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    work = work.dropna(subset=['report_date', 'result_units'])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    if 'is_settled' in work.columns:
        settled_mask = work['is_settled'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        work = work[settled_mask | work['result_units'].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work['roi_pct'] = np.where(work['stake_units'].abs() > 1e-9, work['result_units'] / work['stake_units'], np.nan)
    daily = work.groupby('report_date', dropna=True).agg(
        bets=('result_units', 'size'),
        stake_units=('stake_units', 'sum'),
        result_units=('result_units', 'sum'),
    ).reset_index()
    daily['roi_pct'] = np.where(daily['stake_units'].abs() > 1e-9, daily['result_units'] / daily['stake_units'], np.nan)
    summary_rows = [{
        'settled_bets': int(len(work)),
        'stake_units': float(work['stake_units'].sum()),
        'result_units': float(work['result_units'].sum()),
        'roi_pct': float(work['result_units'].sum() / work['stake_units'].sum()) if abs(float(work['stake_units'].sum())) > 1e-9 else None,
        'avg_clv': float(pd.to_numeric(work.get('clv'), errors='coerce').dropna().mean()) if 'clv' in work.columns else None,
        'max_drawdown_units': abs(float((daily['result_units'].cumsum() - daily['result_units'].cumsum().cummax()).min() or 0.0)),
    }]
    by_market = pd.DataFrame()
    if 'market_type' in work.columns:
        by_market = work.groupby('market_type', dropna=False).agg(
            bets=('result_units', 'size'),
            stake_units=('stake_units', 'sum'),
            result_units=('result_units', 'sum'),
        ).reset_index()
        by_market['roi_pct'] = np.where(by_market['stake_units'].abs() > 1e-9, by_market['result_units'] / by_market['stake_units'], np.nan)
    return pd.DataFrame(summary_rows), by_market
""",
    """def _build_market_realized_summary(bet_journal: pd.DataFrame):
    if bet_journal.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_columns = {'report_date', 'result_units'}
    if not required_columns.issubset(bet_journal.columns):
        return pd.DataFrame(), pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    clv_source = 'clv_value' if 'clv_value' in work.columns else ('clv' if 'clv' in work.columns else None)
    if clv_source is not None:
        work['clv'] = pd.to_numeric(work.get(clv_source), errors='coerce')
    work = work.dropna(subset=['report_date', 'result_units'])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    if 'is_settled' in work.columns:
        settled_mask = work['is_settled'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        work = work[settled_mask | work['result_units'].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work['roi_pct'] = np.where(work['stake_units'].abs() > 1e-9, work['result_units'] / work['stake_units'], np.nan)
    daily = work.groupby('report_date', dropna=True).agg(
        bets=('result_units', 'size'),
        stake_units=('stake_units', 'sum'),
        result_units=('result_units', 'sum'),
    ).reset_index()
    if 'clv' in work.columns:
        clv_daily = work.groupby('report_date', dropna=True)['clv'].mean().reset_index(name='avg_clv')
        daily = daily.merge(clv_daily, on='report_date', how='left')
    daily['cumulative_units'] = daily['result_units'].cumsum()
    daily['rolling_peak_units'] = daily['cumulative_units'].cummax()
    daily['drawdown_units'] = daily['cumulative_units'] - daily['rolling_peak_units']
    daily['roi_pct'] = np.where(daily['stake_units'].abs() > 1e-9, daily['result_units'] / daily['stake_units'], np.nan)
    summary_rows = [{
        'settled_bets': int(len(work)),
        'stake_units': float(work['stake_units'].sum()),
        'result_units': float(work['result_units'].sum()),
        'roi_pct': float(work['result_units'].sum() / work['stake_units'].sum()) if abs(float(work['stake_units'].sum())) > 1e-9 else None,
        'avg_clv': float(work['clv'].dropna().mean()) if 'clv' in work.columns and not work['clv'].dropna().empty else None,
        'max_drawdown_units': abs(float((daily['drawdown_units'].min()) or 0.0)),
    }]
    by_market = pd.DataFrame()
    if 'market_type' in work.columns:
        market_rows = []
        for market_type, grp in work.groupby('market_type', dropna=False):
            grp = grp.sort_values('report_date').copy()
            grp['cumulative_units'] = grp['result_units'].cumsum()
            grp['rolling_peak_units'] = grp['cumulative_units'].cummax()
            grp['drawdown_units'] = grp['cumulative_units'] - grp['rolling_peak_units']
            stake_units = float(grp['stake_units'].sum())
            market_rows.append({
                'market_type': market_type,
                'bets': int(len(grp)),
                'stake_units': stake_units,
                'result_units': float(grp['result_units'].sum()),
                'roi_pct': float(grp['result_units'].sum() / stake_units) if abs(stake_units) > 1e-9 else None,
                'avg_clv': float(grp['clv'].dropna().mean()) if 'clv' in grp.columns and not grp['clv'].dropna().empty else None,
                'max_drawdown_units': abs(float((grp['drawdown_units'].min()) or 0.0)),
                'settled_days': int(grp['report_date'].nunique()),
            })
        by_market = pd.DataFrame(market_rows)
    return pd.DataFrame(summary_rows), by_market
""",
    'dashboard realized summary helper',
)
text = replace_once(
    text,
    """def _build_market_realized_daily(bet_journal: pd.DataFrame) -> pd.DataFrame:
    if bet_journal.empty or 'report_date' not in bet_journal.columns or 'result_units' not in bet_journal.columns:
        return pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    if 'clv' in work.columns:
        work['clv'] = pd.to_numeric(work.get('clv'), errors='coerce')
""",
    """def _build_market_realized_daily(bet_journal: pd.DataFrame) -> pd.DataFrame:
    if bet_journal.empty or 'report_date' not in bet_journal.columns or 'result_units' not in bet_journal.columns:
        return pd.DataFrame()
    work = bet_journal.copy()
    work['report_date'] = pd.to_datetime(work.get('report_date'), errors='coerce').dt.date
    work['result_units'] = pd.to_numeric(work.get('result_units'), errors='coerce')
    work['stake_units'] = pd.to_numeric(work.get('stake_units'), errors='coerce').fillna(1.0)
    clv_source = 'clv_value' if 'clv_value' in work.columns else ('clv' if 'clv' in work.columns else None)
    if clv_source is not None:
        work['clv'] = pd.to_numeric(work.get(clv_source), errors='coerce')
""",
    'dashboard realized daily clv source',
)
text = replace_once(
    text,
    """        if not realized_uncertainty_frame.empty:
            st.markdown('**Realized Uncertainty Learning**')
            st.caption('Technical: this shows the market-level bias learned from settled live predictions and settled priced bets. Plain English: it tells you where the model has learned to trust itself less or a little more based on actual outcomes and closing-line behavior.')
            st.dataframe(realized_uncertainty_frame, use_container_width=True, hide_index=True)

        if not run_history.empty:
""",
    """        if not realized_uncertainty_frame.empty:
            st.markdown('**Realized Uncertainty Learning**')
            st.caption('Technical: this shows the market-level bias learned from settled live predictions and settled priced bets. Plain English: it tells you where the model has learned to trust itself less or a little more based on actual outcomes and closing-line behavior.')
            st.dataframe(realized_uncertainty_frame, use_container_width=True, hide_index=True)

        st.markdown('**Totals Market Proof Track**')
        totals_cols = st.columns(4)
        totals_cols[0].metric('Totals settled bets', int(latest.get('market_proof_total_bets') or 0))
        totals_cols[1].metric('Totals P&L', _fmt_num(latest.get('market_proof_total_units'), 2) + 'u')
        totals_cols[2].metric('Totals ROI', _fmt_pct(latest.get('market_proof_total_roi')))
        totals_cols[3].metric('Totals Avg CLV', _fmt_num(latest.get('market_proof_total_avg_clv'), 3))
        st.caption('Technical: this is the totals-only real-market proof line. Plain English: it shows whether totals themselves are earning trust once priced bets settle, instead of being buried inside the combined market scorecard.')
        if not run_history.empty and 'market_proof_total_roi' in run_history.columns and pd.to_numeric(run_history['market_proof_total_roi'], errors='coerce').notna().any():
            totals_history = run_history.copy()
            totals_history['report_date'] = pd.to_datetime(totals_history['report_date'], errors='coerce')
            left, right = st.columns(2)
            with left:
                _render_line_chart(totals_history.sort_values('report_date'), 'report_date', 'market_proof_total_roi', 'Totals Market Proof ROI')
            with right:
                _render_line_chart(totals_history.sort_values('report_date'), 'report_date', 'market_proof_total_avg_clv', 'Totals Market Proof CLV')
        elif not market_proof_by_market_bundle.empty:
            totals_market_bundle = market_proof_by_market_bundle[market_proof_by_market_bundle['market_type'].astype(str).str.lower() == 'total'].copy()
            if not totals_market_bundle.empty:
                st.dataframe(totals_market_bundle, use_container_width=True, hide_index=True)

        if not run_history.empty:
""",
    'overview totals proof block',
)
text = replace_once(
    text,
    """            realized_summary, realized_by_market = _build_market_realized_summary(bet_journal)
            if not realized_summary.empty:
                realized_row = realized_summary.iloc[0]
                realized_cols = st.columns(4)
                realized_cols[0].metric('Settled market bets', int(realized_row.get('settled_bets') or 0))
                realized_cols[1].metric('Realized P&L', _fmt_num(realized_row.get('result_units'), 2) + 'u')
                realized_cols[2].metric('Realized ROI', _fmt_pct(realized_row.get('roi_pct')))
                realized_cols[3].metric('Avg CLV', _fmt_num(realized_row.get('avg_clv'), 3))
                if not realized_by_market.empty:
                    st.dataframe(realized_by_market, use_container_width=True, hide_index=True)
            else:
                st.info('No settled priced bets are available yet, so this section will turn on once regular-season market journaling starts filling in.')
            if not bet_journal.empty and 'result_units' in bet_journal.columns:

                settled = bet_journal.copy()
""",
    """            realized_summary, realized_by_market = _build_market_realized_summary(bet_journal)
            if not realized_summary.empty:
                realized_row = realized_summary.iloc[0]
                realized_cols = st.columns(4)
                realized_cols[0].metric('Settled market bets', int(realized_row.get('settled_bets') or 0))
                realized_cols[1].metric('Realized P&L', _fmt_num(realized_row.get('result_units'), 2) + 'u')
                realized_cols[2].metric('Realized ROI', _fmt_pct(realized_row.get('roi_pct')))
                realized_cols[3].metric('Avg CLV', _fmt_num(realized_row.get('avg_clv'), 3))
                if not realized_by_market.empty:
                    st.dataframe(realized_by_market, use_container_width=True, hide_index=True)
                    totals_realized = realized_by_market[realized_by_market['market_type'].astype(str).str.lower() == 'total'].copy()
                    if not totals_realized.empty:
                        totals_row = totals_realized.iloc[0]
                        st.markdown('**Totals-specific real-market proof**')
                        st.caption('Technical: this isolates settled totals bets only. Plain English: it tells you whether the totals side of the engine is actually making money and beating close on its own.')
                        totals_cols = st.columns(4)
                        totals_cols[0].metric('Totals bets', int(totals_row.get('bets') or 0))
                        totals_cols[1].metric('Totals ROI', _fmt_pct(totals_row.get('roi_pct')))
                        totals_cols[2].metric('Totals Avg CLV', _fmt_num(totals_row.get('avg_clv'), 3))
                        totals_cols[3].metric('Totals Max DD', _fmt_num(totals_row.get('max_drawdown_units'), 2) + 'u')
                        totals_daily = _build_market_realized_daily(bet_journal[bet_journal.get('market_type', pd.Series(dtype=str)).astype(str).str.lower() == 'total'].copy()) if 'market_type' in bet_journal.columns else pd.DataFrame()
                        if not totals_daily.empty:
                            left, right = st.columns(2)
                            with left:
                                _render_line_chart(totals_daily, 'report_date', 'cumulative_units', 'Totals Settled Equity Curve')
                            with right:
                                if 'avg_clv' in totals_daily.columns and pd.to_numeric(totals_daily['avg_clv'], errors='coerce').notna().any():
                                    _render_line_chart(totals_daily, 'report_date', 'avg_clv', 'Totals Daily Average CLV')
            else:
                st.info('No settled priced bets are available yet, so this section will turn on once regular-season market journaling starts filling in.')
            if not bet_journal.empty and 'result_units' in bet_journal.columns:

                settled = bet_journal.copy()
""",
    'portfolio totals proof block',
)
path.write_text(text, encoding='utf-8')

print('updated quant_store.py and quant_dashboard.py with totals market proof persistence and views')
