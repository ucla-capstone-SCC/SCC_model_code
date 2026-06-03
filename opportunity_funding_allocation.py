"""
Opportunity Score-Based Funding Allocation from Longitudinal LME Model
- Gap target:       County average (2027) — bring below-average ZIPs up to the mean
- Population weight: senior_households_total
- Breakdown:        County-wide % of total budget by variable
- Data:             arima_forecast_values_standardized_v3.csv (Year <= 2027)
- Output:           opportunity_funding_allocation.docx
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\emily\Downloads\arimax_forecast_TS10_qol_indices.csv')
df = df[df['Year'] <= 2027].copy().reset_index(drop=True)
df['Pandemic'] = (df['Pandemic'] > 0).astype(int)

OUTCOME    = 'qol_composite_index'
GROUP_VAR  = 'zip_code'
TIME_VAR   = 'Year'
PREDICTORS = [c for c in df.columns if c not in [GROUP_VAR, TIME_VAR, OUTCOME]]

PLAIN = {
    'Pandemic':                                'Pandemic indicator',
    'ACS_PCT_HH_ABOVE65_ZC':                   'Households with someone 65+ (%)',
    'CCBP_TOT_CHS_ZP':                         'Community health services',
    'CCBP_TOT_CS_ZP':                          'Community social services',
    'CCBP_TOT_FF_ZP':                          'Fast food establishments',
    'CCBP_TOT_FSR_ZP':                         'Full-service restaurants',
    'CCBP_TOT_RET_ZP':                         'Retail establishments',
    'drive_time_hospital_min':                 'Hospital proximity (reversed)',
    'green_coverage_pct':                      'Green space coverage (%)',
    'healthcare_facilities_per_10k_older_adults': 'Healthcare facilities per 10k seniors',
    'pct_asian_60_plus':                       'Asian older adults (%)',
    'pct_black_60_plus':                       'Black older adults (%)',
    'pct_disabled_60_plus':                    'Lower disability burden (reversed)',
    'pct_hispanic_60_plus':                    'Hispanic older adults (%)',
    'pct_limited_english_60_plus':             'Lower language barrier (reversed)',
    'pct_no_computer':                         'Higher digital access (reversed)',
    'pct_no_internet':                         'Higher internet access (reversed)',
    'pct_pop_60_plus':                         'Population aged 60+ (%)',
    'pct_snap_households_60_plus':             'Lower SNAP dependency (reversed)',
    'pct_white_non_hispanic_60_plus':          'Non-Hispanic White older adults (%)',
    'senior_households_total':                 'Total senior households',
    'transit_coverage_pct':                    'Transit coverage (%)',
}

ACTIONABLE = {
    'CCBP_TOT_CHS_ZP':                         'Community health services investment',
    'CCBP_TOT_CS_ZP':                          'Community social services investment',
    'drive_time_hospital_min':                 'Senior transportation programs',
    'green_coverage_pct':                      'Green space & parks development',
    'healthcare_facilities_per_10k_older_adults': 'Healthcare facility access',
    'pct_no_computer':                         'Digital equity & device programs',
    'pct_no_internet':                         'Broadband connectivity expansion',
    'pct_snap_households_60_plus':             'Food security & nutrition programs',
    'pct_limited_english_60_plus':             'Language access & translation services',
    'transit_coverage_pct':                    'Public transit expansion',
}

# ── Refit LME ─────────────────────────────────────────────────────────────────
print("Fitting LME model...")
lme_res = MixedLM(df[OUTCOME], sm.add_constant(df[PREDICTORS]),
                  groups=df[GROUP_VAR],
                  exog_re=np.ones((len(df), 1))).fit(reml=True, method='lbfgs')
print("LME done.")

lme_coefs = {v: lme_res.params[v]   for v in PREDICTORS if v in lme_res.params.index}
lme_pvals = {v: lme_res.pvalues[v]  for v in PREDICTORS if v in lme_res.pvalues.index}

# ── 2027 snapshot ─────────────────────────────────────────────────────────────
df_2027 = df[df['Year'] == 2027].copy().reset_index(drop=True)

# County average per variable in 2027 (the target each ZIP should reach)
county_mean_2027 = {v: df_2027[v].mean() for v in PREDICTORS}

# ── Opportunity Score ─────────────────────────────────────────────────────────
# For each ZIP z and actionable variable v (all vars have higher = better):
#   gap(v, z)      = max(0, county_mean(v) - x_v(z, 2027))
#   opp(v, z)      = |β_v| × gap(v, z) × senior_households_total(z)
#
# ZIP funding share  = sum_v opp(v, z) / total opp across all ZIPs
# Variable % budget  = sum_z opp(v, z) / total opp across all ZIPs & vars

opp_rows = []
for _, row in df_2027.iterrows():
    zc      = int(row[GROUP_VAR])
    seniors = row['senior_households_total']
    rec     = {'zip_code': zc, 'qol_2027': round(row[OUTCOME], 4),
               'senior_households': int(seniors) if not np.isnan(seniors) else 0}

    total_opp = 0.0
    for v in ACTIONABLE:
        if v not in lme_coefs:
            rec[f'opp_{v}'] = 0.0
            continue
        beta      = lme_coefs[v]
        x_zip     = row[v]
        x_mean    = county_mean_2027[v]
        gap       = max(0.0, x_mean - x_zip)          # below-average gap only
        opp       = abs(beta) * gap * max(seniors, 0) # population-weighted
        rec[f'opp_{v}'] = round(opp, 4)
        total_opp += opp

    rec['total_opp'] = round(total_opp, 4)
    opp_rows.append(rec)

df_opp = pd.DataFrame(opp_rows)

# ZIP-level funding share
grand_total = df_opp['total_opp'].sum()
df_opp['funding_pct'] = (df_opp['total_opp'] / grand_total * 100).round(2)
df_opp = df_opp.sort_values('funding_pct', ascending=False).reset_index(drop=True)
df_opp['rank'] = range(1, len(df_opp) + 1)

# County-wide variable budget breakdown
var_budget = {}
for v in ACTIONABLE:
    col = f'opp_{v}'
    var_budget[v] = df_opp[col].sum()

var_budget_df = pd.DataFrame({
    'variable':        list(var_budget.keys()),
    'total_opp':       list(var_budget.values()),
}).sort_values('total_opp', ascending=False).reset_index(drop=True)
var_budget_df['pct_of_budget'] = (var_budget_df['total_opp'] / grand_total * 100).round(2)

# Per-ZIP variable breakdown (% within each ZIP)
for v in ACTIONABLE:
    col = f'opp_{v}'
    df_opp[f'zip_pct_{v}'] = (df_opp[col] / df_opp['total_opp'].replace(0, np.nan) * 100).round(1).fillna(0)

# ── Tier labels ───────────────────────────────────────────────────────────────
def tier(pct):
    if pct >= 5:   return 'High Priority'
    if pct >= 2:   return 'Moderate Priority'
    if pct >= 0.5: return 'Low Priority'
    return 'Minimal / No Gap'

n_high = (df_opp['funding_pct'] >= 5).sum()
n_mod  = ((df_opp['funding_pct'] >= 2) & (df_opp['funding_pct'] < 5)).sum()
n_low  = ((df_opp['funding_pct'] >= 0.5) & (df_opp['funding_pct'] < 2)).sum()
n_min  = (df_opp['funding_pct'] < 0.5).sum()

# ── Word doc helpers ───────────────────────────────────────────────────────────
def shade_row(row, hex_color):
    for cell in row.cells:
        tc = cell._tc; tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), hex_color)
        tcPr.append(shd)

def shade_cell(cell, hex_color):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def center(cell):
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

def pct_color(pct):
    if pct >= 5:   return 'FFC7CE'
    if pct >= 2:   return 'FFEB9C'
    if pct >= 0.5: return 'C6EFCE'
    return 'F2F2F2'

def bar(pct, width=20):
    filled = round(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)

# ── Build document ────────────────────────────────────────────────────────────
doc = Document()
doc.styles['Normal'].font.name = 'Calibri'
doc.styles['Normal'].font.size = Pt(11)

tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run('Opportunity Score-Based Funding Allocation\nfor Senior Services in Santa Clara County')
r.bold = True; r.font.size = Pt(16)
doc.add_paragraph(
    f'Based on Longitudinal LME Model  |  2027 Forecast Data  |  Population-Weighted\n'
    f'Prepared: {datetime.date.today().strftime("%B %d, %Y")}'
).alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ── 1. Methodology ────────────────────────────────────────────────────────────
doc.add_heading('1. Methodology', level=1)
doc.add_paragraph(
    'This report allocates funding using an Opportunity Score that combines three factors '
    'for each ZIP code and each actionable variable: (1) how much the predictor matters '
    'for QoL (model coefficient), (2) how far below the county average a ZIP currently is '
    '(the gap), and (3) how many seniors live in that ZIP (population weight). '
    'This ensures funding flows where it can have the greatest QoL impact on the most people.'
)

doc.add_heading('Opportunity Score formula', level=2)
doc.add_paragraph('For each ZIP z and actionable variable v:', style='Normal')
doc.add_paragraph(
    '    Gap(v, z)          =  max(0,  county_mean(v, 2027) − x_v(z, 2027))\n'
    '    Opp(v, z)          =  |β_v|  ×  Gap(v, z)  ×  senior_households(z)\n'
    '    ZIP funding share   =  Σ_v Opp(v, z)  /  Σ_z Σ_v Opp(v, z)  × 100%\n'
    '    Variable % budget   =  Σ_z Opp(v, z)  /  Σ_z Σ_v Opp(v, z)  × 100%',
    style='Intense Quote'
)
doc.add_paragraph(
    'Key design decisions:\n'
    '• Gap target = county average (2027): ZIPs performing above the county mean on a '
    'variable receive zero gap — they do not need investment in that area.\n'
    '• All variables are z-scored and sign-reversed where needed so higher always means better. '
    'Gaps are therefore directly comparable across variables.\n'
    '• |β_v| (absolute coefficient) weights each gap by how strongly that variable '
    'improves QoL — investing in high-impact variables is prioritized.\n'
    '• Population weight (senior_households_total) ensures larger senior communities '
    'receive proportionally more resources for the same per-capita gap.\n'
    '• Pandemic and demographic variables (race/ethnicity, age share) are excluded '
    'as they reflect structural conditions, not addressable gaps.'
)

doc.add_heading('Actionable variables and funding levers', level=2)
t = doc.add_table(rows=1, cols=3)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
for h, cell in zip(['Variable', 'Plain Name', 'Funding Lever'], t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True
for v, lever in ACTIONABLE.items():
    r = t.add_row()
    r.cells[0].text = v
    r.cells[1].text = PLAIN.get(v, v)
    r.cells[2].text = lever
doc.add_paragraph()

# ── 2. LME Coefficients ───────────────────────────────────────────────────────
doc.add_heading('2. LME Coefficients Used for Opportunity Scoring', level=1)
doc.add_paragraph(
    'All variables are z-scored, so each β represents the change in QoL (in SD units) '
    'per 1 SD increase in the predictor. |β| is used as the impact weight in the '
    'opportunity score. ✓ marks actionable variables included in the funding formula.'
)
t = doc.add_table(rows=1, cols=5)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
for h, cell in zip(['Variable', 'Plain Name', 'β (LME)', 'p-value', '|β| Weight'], t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)

for v in PREDICTORS:
    if v not in lme_coefs: continue
    b   = lme_coefs[v]
    p   = lme_pvals.get(v, np.nan)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
    flag = ' ✓' if v in ACTIONABLE else ''
    r = t.add_row()
    r.cells[0].text = v + flag
    r.cells[1].text = PLAIN.get(v, v)
    r.cells[2].text = f'{b:+.4f} {sig}'; center(r.cells[2])
    r.cells[3].text = f'{p:.4f}';         center(r.cells[3])
    r.cells[4].text = f'{abs(b):.4f}';    center(r.cells[4])
    if v in ACTIONABLE:
        shade_cell(r.cells[0], 'EBF3FB')
        shade_cell(r.cells[4], 'EBF3FB')
doc.add_paragraph('✓ = included in opportunity score')
doc.add_paragraph()

# ── 3. County-Wide Variable Budget Breakdown ──────────────────────────────────
doc.add_heading('3. County-Wide Budget Allocation by Intervention Type', level=1)
doc.add_paragraph(
    'The table below answers: of the total funding pool, what percentage should be '
    'directed to each type of intervention across all ZIP codes? '
    'Variables with a large |β| and widespread below-average gaps attract the most funding.'
)
t = doc.add_table(rows=1, cols=5)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
for h, cell in zip(['Rank', 'Variable', 'Funding Lever', '% of Total Budget', 'Visual'], t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)

for rank_i, row in var_budget_df.iterrows():
    v   = row['variable']
    pct = row['pct_of_budget']
    r   = t.add_row()
    r.cells[0].text = str(rank_i + 1);      center(r.cells[0])
    r.cells[1].text = v
    r.cells[2].text = ACTIONABLE.get(v, v)
    r.cells[3].text = f'{pct:.2f}%';        center(r.cells[3])
    r.cells[4].text = bar(pct)
    if pct >= 20:   shade_cell(r.cells[3], 'FFC7CE')
    elif pct >= 10: shade_cell(r.cells[3], 'FFEB9C')
    else:           shade_cell(r.cells[3], 'C6EFCE')

# Verify sums to 100
total_check = var_budget_df['pct_of_budget'].sum()
doc.add_paragraph(f'Total: {total_check:.1f}% (rounding may cause minor deviation from 100%)')
doc.add_paragraph()

# ── 4. ZIP-Level Funding Allocation ───────────────────────────────────────────
doc.add_heading('4. ZIP-Level Funding Allocation (Ranked by Priority)', level=1)
doc.add_paragraph(
    f'Tier summary:  High Priority (≥5%): {n_high} ZIPs  |  '
    f'Moderate (2–5%): {n_mod} ZIPs  |  '
    f'Low (0.5–2%): {n_low} ZIPs  |  '
    f'Minimal/No Gap (<0.5%): {n_min} ZIPs\n'
    'Color: red ≥ 5% | yellow ≥ 2% | green ≥ 0.5% | grey < 0.5%'
)

t = doc.add_table(rows=1, cols=6)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
for h, cell in zip(['Rank', 'ZIP', 'QoL 2027 (z)', 'Senior HH', 'Funding Share (%)', 'Priority Tier'],
                   t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)

for _, row in df_opp.iterrows():
    pct = row['funding_pct']
    r   = t.add_row()
    r.cells[0].text = str(int(row['rank']));            center(r.cells[0])
    r.cells[1].text = str(int(row['zip_code']));        center(r.cells[1])
    r.cells[2].text = f"{row['qol_2027']:+.3f}";       center(r.cells[2])
    r.cells[3].text = f"{int(row['senior_households']):,}"; center(r.cells[3])
    r.cells[4].text = f'{pct:.2f}%';                   center(r.cells[4])
    r.cells[5].text = tier(pct)
    shade_cell(r.cells[4], pct_color(pct))
    shade_cell(r.cells[5], pct_color(pct))
doc.add_paragraph()

# ── 5. Top Priority ZIPs — Variable-Level Breakdown ───────────────────────────
doc.add_heading('5. Top Priority ZIPs — Where to Spend Within Each ZIP', level=1)
doc.add_paragraph(
    'For each high/moderate priority ZIP, the table shows what percentage of that '
    'ZIP\'s allocation should go to each intervention type, based on which gaps are '
    'largest relative to the county average.'
)

top_zips = df_opp[df_opp['funding_pct'] >= 2.0].copy()

t = doc.add_table(rows=1, cols=len(ACTIONABLE) + 3)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
headers = ['ZIP', 'Fund %', 'Senior HH'] + [PLAIN.get(v, v)[:20] for v in ACTIONABLE]
for h, cell in zip(headers, t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)
    cell.paragraphs[0].runs[0].font.size = Pt(8)

for _, row in top_zips.iterrows():
    r = t.add_row()
    r.cells[0].text = str(int(row['zip_code']));           center(r.cells[0])
    r.cells[1].text = f"{row['funding_pct']:.2f}%";       center(r.cells[1])
    shade_cell(r.cells[1], pct_color(row['funding_pct']))
    r.cells[2].text = f"{int(row['senior_households']):,}"; center(r.cells[2])
    for i, v in enumerate(ACTIONABLE, start=3):
        pct_v = row[f'zip_pct_{v}']
        r.cells[i].text = f'{pct_v:.0f}%' if pct_v > 0 else '—'; center(r.cells[i])
        cell = r.cells[i]
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.size = Pt(8)
        if pct_v >= 30:   shade_cell(cell, 'FFC7CE')
        elif pct_v >= 15: shade_cell(cell, 'FFEB9C')
        elif pct_v > 0:   shade_cell(cell, 'C6EFCE')
doc.add_paragraph('Values show % of that ZIP\'s allocation directed to each intervention. Red ≥ 30% | Yellow ≥ 15% | Green > 0%')
doc.add_paragraph()

# ── 6. Gap Analysis Table ─────────────────────────────────────────────────────
doc.add_heading('6. Gap Analysis — Below-Average ZIPs by Variable', level=1)
doc.add_paragraph(
    'For each actionable variable, this table shows how many ZIPs fall below the '
    'county average and the average gap size — identifying where needs are most widespread.'
)
gap_rows = []
for v in ACTIONABLE:
    vals_2027 = df_2027[v]
    mean_val  = vals_2027.mean()
    gaps      = (mean_val - vals_2027).clip(lower=0)
    n_below   = (gaps > 0).sum()
    avg_gap   = gaps[gaps > 0].mean() if n_below > 0 else 0
    gap_rows.append({
        'variable':   v,
        'plain_name': PLAIN.get(v, v),
        'lever':      ACTIONABLE[v],
        'coef_abs':   round(abs(lme_coefs.get(v, 0)), 4),
        'n_below_avg': n_below,
        'avg_gap':    round(avg_gap, 4),
        'budget_pct': var_budget_df.loc[var_budget_df['variable'] == v, 'pct_of_budget'].values[0]
                      if v in var_budget_df['variable'].values else 0,
    })
gap_df = pd.DataFrame(gap_rows).sort_values('budget_pct', ascending=False)

t = doc.add_table(rows=1, cols=6)
t.style = 'Table Grid'
shade_row(t.rows[0], 'BDD7EE')
for h, cell in zip(['Variable', 'Funding Lever', '|β|', 'ZIPs Below Avg', 'Avg Gap (z)', '% of Budget'],
                   t.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)

for _, row in gap_df.iterrows():
    pct = row['budget_pct']
    r = t.add_row()
    r.cells[0].text = row['variable']
    r.cells[1].text = row['lever']
    r.cells[2].text = f"{row['coef_abs']:.4f}"; center(r.cells[2])
    r.cells[3].text = f"{int(row['n_below_avg'])} / {len(df_2027)}"; center(r.cells[3])
    r.cells[4].text = f"{row['avg_gap']:.4f}";  center(r.cells[4])
    r.cells[5].text = f"{pct:.2f}%";            center(r.cells[5])
    if pct >= 20:   shade_cell(r.cells[5], 'FFC7CE')
    elif pct >= 10: shade_cell(r.cells[5], 'FFEB9C')
    else:           shade_cell(r.cells[5], 'C6EFCE')
doc.add_paragraph()

# ── 7. Notes ──────────────────────────────────────────────────────────────────
doc.add_heading('7. Notes & Limitations', level=1)
doc.add_paragraph(
    '• Gap is measured against the 2027 county average (across 63 ZIPs). ZIPs already '
    'at or above the county average on a given variable receive no funding for that dimension.\n'
    '• 2025–2027 values are ARIMA forecasts — forecast uncertainty increases for later years.\n'
    '• All variables are z-scored: gaps and coefficients are on a standardized scale, '
    'making cross-variable comparisons meaningful.\n'
    '• Population weight uses senior_households_total (2027 forecast). Larger ZIP populations '
    'receive proportionally more resources for equal per-capita gaps.\n'
    '• The opportunity score captures potential QoL improvement from closing gaps. '
    'It does not account for implementation costs, which vary by intervention type.\n'
    '• Results should be used alongside local program capacity, political feasibility, '
    'and cost-effectiveness data for final allocation decisions.'
)

out_path = r'C:\Users\emily\Downloads\opportunity_funding_allocation.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
print(f'\nCounty-wide variable budget breakdown:')
print(var_budget_df[['variable','pct_of_budget']].to_string(index=False))
print(f'\nTop 15 ZIPs by funding priority:')
print(df_opp[['rank','zip_code','qol_2027','senior_households','funding_pct']].head(15).to_string(index=False))
print(f'\nTier counts: High={n_high}, Moderate={n_mod}, Low={n_low}, Minimal={n_min}')
