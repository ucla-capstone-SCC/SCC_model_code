import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import linregress
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(r'C:\Users\emily\Downloads\arima_forecast_values_standardized_v3.csv')
df = df[df['Year'] <= 2027].copy().reset_index(drop=True)
df['Pandemic'] = (df['Pandemic'] > 0).astype(int)   # recode to binary: 0=no, 1=yes

OUTCOME    = 'qol_composite_index'
GROUP_VAR  = 'zip_code'
TIME_VAR   = 'Year'
PREDICTORS = [c for c in df.columns if c not in [GROUP_VAR, TIME_VAR, OUTCOME]]

N_obs  = len(df)
N_zip  = df[GROUP_VAR].nunique()
N_year = df[TIME_VAR].nunique()

print(f"N={N_obs}, ZIPs={N_zip}, Years={N_year}")
print(f"Predictors ({len(PREDICTORS)}): {PREDICTORS}")

PLAIN_NAMES = {
    'Pandemic':                                'Pandemic (binary: 0=no pandemic, 1=pandemic)',
    'ACS_PCT_HH_ABOVE65_ZC':                   'Households with someone 65+ (%)',
    'CCBP_TOT_CHS_ZP':                         'Community health services',
    'CCBP_TOT_CS_ZP':                          'Community social services',
    'CCBP_TOT_FF_ZP':                          'Fast food establishments',
    'CCBP_TOT_FSR_ZP':                         'Full-service restaurants',
    'CCBP_TOT_RET_ZP':                         'Retail establishments',
    'drive_time_hospital_min':                 'Hospital proximity (reversed: higher = closer)',
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
    'share_older_adult_living_alone_households': 'Lower social isolation (reversed)',
    'transit_coverage_pct':                    'Transit coverage (%)',
}

INTERP = {
    'Pandemic':                                'Pandemic period effect on QoL for older adults',
    'ACS_PCT_HH_ABOVE65_ZC':                   'Concentration of 65+ households shapes resource needs',
    'CCBP_TOT_CHS_ZP':                         'More community health services supports QoL',
    'CCBP_TOT_CS_ZP':                          'More community social services supports QoL',
    'CCBP_TOT_FF_ZP':                          'Fast food density reflects food environment quality',
    'CCBP_TOT_FSR_ZP':                         'Full-service restaurant availability reflects economic vitality',
    'CCBP_TOT_RET_ZP':                         'Retail density reflects neighborhood economic activity',
    'drive_time_hospital_min':                 'Closer proximity to hospital improves healthcare access',
    'green_coverage_pct':                      'More green space supports physical and mental wellbeing',
    'healthcare_facilities_per_10k_older_adults': 'More nearby facilities improves healthcare access',
    'pct_asian_60_plus':                       'Reflects QoL conditions for Asian older adult communities',
    'pct_black_60_plus':                       'Reflects racial equity gaps in QoL for Black seniors',
    'pct_disabled_60_plus':                    'Lower disability burden is linked to higher QoL',
    'pct_hispanic_60_plus':                    'Reflects QoL conditions for Hispanic older adult communities',
    'pct_limited_english_60_plus':             'Fewer language barriers improves service access and QoL',
    'pct_no_computer':                         'Greater digital access is associated with higher QoL',
    'pct_no_internet':                         'Greater internet access is associated with higher QoL',
    'pct_pop_60_plus':                         'Larger older adult share affects community resource demands',
    'pct_snap_households_60_plus':             'Lower reliance on food assistance signals economic stability',
    'pct_white_non_hispanic_60_plus':          'Reflects structural conditions for White non-Hispanic seniors',
    'senior_households_total':                 'More senior households reflects a larger aging population',
    'share_older_adult_living_alone_households': 'Lower social isolation is linked to higher QoL',
    'transit_coverage_pct':                    'Better transit access supports older adult independence',
}

# ── Panel setup ───────────────────────────────────────────────────────────────
df_pan = df.set_index([GROUP_VAR, TIME_VAR])
X = sm.add_constant(df_pan[PREDICTORS])
y = df_pan[OUTCOME]

# ── Models ────────────────────────────────────────────────────────────────────
fe_res = PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(
    cov_type='clustered', cluster_entity=True)
re_res = RandomEffects(y, X).fit(cov_type='robust')

df3 = df.copy()
lme_res = MixedLM(df3[OUTCOME], sm.add_constant(df3[PREDICTORS]),
                  groups=df3[GROUP_VAR],
                  exog_re=np.ones((len(df3), 1))).fit(reml=True, method='lbfgs')

print("All models done")

# Hausman
common = [v for v in fe_res.params.index if v in re_res.params.index and v != 'Intercept']
try:
    diff   = fe_res.params[common].values - re_res.params[common].values
    V_diff = fe_res.cov.loc[common, common].values - re_res.cov.loc[common, common].values
    H_stat = float(diff @ np.linalg.pinv(V_diff) @ diff)
    H_df   = len(common)
    H_pval = 1 - stats.chi2.cdf(H_stat, H_df)
except:
    H_stat, H_df, H_pval = np.nan, len(common), np.nan

# ── Collect results ───────────────────────────────────────────────────────────
def collect(res):
    rows = []
    for v in PREDICTORS:
        if v not in res.params.index:
            continue
        b   = res.params[v]
        se  = res.std_errors[v] if hasattr(res, 'std_errors') else res.bse[v]
        t   = res.tstats[v]    if hasattr(res, 'tstats')     else res.tvalues[v]
        p   = res.pvalues[v]
        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '.' if p<0.1 else ''
        rows.append({'Variable': v, 'Coef': b, 'SE': se, 't': t, 'p': p, 'Sig': sig})
    return pd.DataFrame(rows)

fe_tbl  = collect(fe_res)
re_tbl  = collect(re_res)
lme_tbl = collect(lme_res)

def get_val(tbl, var, col):
    row = tbl[tbl['Variable'] == var]
    return row[col].values[0] if len(row) else np.nan

# ── Cross-model significance ──────────────────────────────────────────────────
cross = []
for v in PREDICTORS:
    fe_p  = get_val(fe_tbl,  v, 'p');   fe_b  = get_val(fe_tbl,  v, 'Coef')
    re_p  = get_val(re_tbl,  v, 'p');   re_b  = get_val(re_tbl,  v, 'Coef')
    lme_p = get_val(lme_tbl, v, 'p');   lme_b = get_val(lme_tbl, v, 'Coef')
    n_sig = sum(p < 0.05 for p in [fe_p, re_p, lme_p] if not np.isnan(p))
    signs = [np.sign(b) for b in [fe_b, re_b, lme_b] if not np.isnan(b)]
    cross.append({'Variable': v,
                  'FE_b': fe_b, 'FE_p': fe_p,
                  'RE_b': re_b, 'RE_p': re_p,
                  'LME_b': lme_b, 'LME_p': lme_p,
                  'n_sig': n_sig,
                  'sign_consistent': len(set(signs)) == 1 if signs else False})
df_cross = pd.DataFrame(cross)

sig_all3 = df_cross[df_cross['n_sig'] == 3].sort_values('LME_b', key=abs, ascending=False)
sig_2of3 = df_cross[df_cross['n_sig'] == 2].sort_values('LME_b', key=abs, ascending=False)
sig_1of3 = df_cross[df_cross['n_sig'] == 1].sort_values('LME_b', key=abs, ascending=False)
sig_none = df_cross[df_cross['n_sig'] == 0]

# LME random intercepts
re_intercepts = pd.DataFrame(
    [(zc, vals[0]) for zc, vals in lme_res.random_effects.items()],
    columns=['zip_code', 'random_intercept']
).sort_values('random_intercept', ascending=False).reset_index(drop=True)

# Population and poverty trends
def var_trend_table(df_raw, var, zip_list):
    rows = []
    for zc in zip_list:
        sub = df_raw[df_raw['zip_code'] == zc][['Year', var]].dropna().sort_values('Year')
        if len(sub) >= 4:
            slope, _, _, p, _ = linregress(sub['Year'], sub[var])
            direction = 'Increasing' if slope > 0.005 else 'Decreasing' if slope < -0.005 else 'Stable'
        else:
            slope = np.nan; direction = 'N/A'
        rows.append({'ZIP': zc, 'Slope/yr': round(slope, 4) if not np.isnan(slope) else np.nan,
                     'Trend': direction})
    return pd.DataFrame(rows)

all_zips = sorted(df['zip_code'].unique())
pop_trend_df = var_trend_table(df, 'pct_pop_60_plus', all_zips)

print("Analysis computed")

# ── Word helpers ──────────────────────────────────────────────────────────────
def shade_row(row, hex_color):
    for cell in row.cells:
        tc = cell._tc; tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_color)
        tcPr.append(shd)

def shade_cell(cell, hex_color):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def center(cell):
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

def sig_color(p):
    if np.isnan(p): return 'F2F2F2'
    if p < 0.001:   return 'C6EFCE'
    if p < 0.05:    return 'FFEB9C'
    if p < 0.10:    return 'FCE4D6'
    return 'F2F2F2'

def coef_color(b, p):
    if np.isnan(p) or p >= 0.05: return 'F2F2F2'
    return 'C6EFCE' if b > 0 else 'FFC7CE'

def add_results_table(doc, tbl_df, caption):
    doc.add_paragraph(caption, style='Intense Quote')
    headers = ['Variable', 'Plain Name', 'Coef', 'SE', 't-stat', 'p-value', 'Sig']
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    shade_row(t.rows[0], 'BDD7EE')
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = h
        c.paragraphs[0].runs[0].bold = True; center(c)
    for _, row in tbl_df.iterrows():
        r = t.add_row()
        vals = [row['Variable'], PLAIN_NAMES.get(row['Variable'], row['Variable']),
                f"{row['Coef']:.4f}", f"{row['SE']:.4f}",
                f"{row['t']:.3f}", f"{row['p']:.4f}", row['Sig']]
        for i, v in enumerate(vals):
            r.cells[i].text = str(v)
            if i != 1: center(r.cells[i])
            if i == 2: shade_cell(r.cells[i], coef_color(row['Coef'], row['p']))
            if i == 5: shade_cell(r.cells[i], sig_color(row['p']))
    doc.add_paragraph()

def sig_label(p):
    if np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '.'
    return ''

def add_effect_bullets(doc, df_sub, caption):
    if len(df_sub) == 0:
        doc.add_paragraph(f'{caption}: None.'); return
    doc.add_paragraph(caption, style='Intense Quote')
    for _, row in df_sub.iterrows():
        v = row['Variable']
        fe_b  = row['FE_b'];  fe_p  = row['FE_p']
        re_b  = row['RE_b'];  re_p  = row['RE_p']
        lme_b = row['LME_b']; lme_p = row['LME_p']
        fe_str  = f"{fe_b:+.3f}{sig_label(fe_p)}"   if not np.isnan(fe_b)  else 'N/A'
        re_str  = f"{re_b:+.3f}{sig_label(re_p)}"   if not np.isnan(re_b)  else 'N/A'
        lme_str = f"{lme_b:+.3f}{sig_label(lme_p)}" if not np.isnan(lme_b) else 'N/A'
        sign_note = 'consistent signs' if row['sign_consistent'] else 'inconsistent signs'
        interp = INTERP.get(v, '')
        p = doc.add_paragraph(style='List Bullet')
        run = p.add_run(f'{v}')
        run.bold = True
        p.add_run(f'  —  FE: {fe_str}  |  RE: {re_str}  |  LME: {lme_str}  |  {sign_note}')
        if interp:
            p.add_run(f'\n    {interp}').italic = True
    doc.add_paragraph()

def add_pos_neg_bullets(doc, tbl_df, model_name):
    sig = tbl_df[tbl_df['p'] < 0.05].copy()
    pos = sig[sig['Coef'] > 0].sort_values('Coef', ascending=False)
    neg = sig[sig['Coef'] < 0].sort_values('Coef')

    doc.add_paragraph(f'Significant positive effects — {model_name}', style='Intense Quote')
    if len(pos):
        for _, row in pos.iterrows():
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(f"{row['Variable']}").bold = True
            p.add_run(f"  —  {row['Coef']:+.3f}{row['Sig']}  (p={row['p']:.4f})")
            interp = INTERP.get(row['Variable'], '')
            if interp:
                p.add_run(f'\n    {interp}').italic = True
    else:
        doc.add_paragraph('None.')
    doc.add_paragraph()

    doc.add_paragraph(f'Significant negative effects — {model_name}', style='Intense Quote')
    if len(neg):
        for _, row in neg.iterrows():
            p = doc.add_paragraph(style='List Bullet')
            p.add_run(f"{row['Variable']}").bold = True
            p.add_run(f"  —  {row['Coef']:+.3f}{row['Sig']}  (p={row['p']:.4f})")
            interp = INTERP.get(row['Variable'], '')
            if interp:
                p.add_run(f'\n    {interp}').italic = True
    else:
        doc.add_paragraph('None.')
    doc.add_paragraph()

# ── Build document ────────────────────────────────────────────────────────────
doc = Document()
doc.styles['Normal'].font.name = 'Calibri'
doc.styles['Normal'].font.size = Pt(11)

tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = tp.add_run('Longitudinal Analysis of Quality-of-Life for Older Adults\nSanta Clara County ZIP Codes — 2015 to 2027')
run.bold = True; run.font.size = Pt(16)
doc.add_paragraph(f'Prepared: {datetime.date.today().strftime("%B %d, %Y")}').alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()

# ── 1. Overview ───────────────────────────────────────────────────────────────
doc.add_heading('1. Overview', level=1)
doc.add_paragraph(
    f'This analysis examines how 23 community-level predictors are associated with a '
    f'composite Quality-of-Life (QoL) index for older adults across {N_zip} ZIP codes '
    f'in Santa Clara County over {N_year} years (2015–2029), combining observed data '
    f'with ARIMA-based forecasts. Missing values were handled with MICE imputation prior '
    f'to analysis. Seven predictors with negative directionality (e.g. drive time, '
    f'disability rate) had their signs reversed so that higher values consistently '
    f'indicate better conditions. All variables including the outcome are standardized '
    f'(z-scored), so all coefficients are directly comparable — each represents the '
    f'change in QoL in standard deviation units per one SD change in the predictor.'
)
doc.add_paragraph(
    f'Three complementary panel models are estimated:\n'
    f'  • Fixed Effects (FE): Controls for all stable ZIP characteristics; captures within-ZIP changes over time.\n'
    f'  • Random Effects (RE): Uses both within- and between-ZIP variation; more efficient under valid assumptions.\n'
    f'  • Mixed Effects (LME): Random intercept per ZIP; most flexible specification.\n\n'
    f'Hausman test (FE vs RE): statistic = {H_stat:.3f}, p = {H_pval:.4f}. '
    + ('Fixed Effects preferred — random effects assumptions are violated.'
       if not np.isnan(H_pval) and H_pval < 0.05
       else 'Random Effects not rejected — both FE and RE are valid.')
)

# ── 2. Model Fit ──────────────────────────────────────────────────────────────
doc.add_heading('2. Model Fit Summary', level=1)
fit_tbl = doc.add_table(rows=1, cols=4)
fit_tbl.style = 'Table Grid'
shade_row(fit_tbl.rows[0], 'BDD7EE')
for h, cell in zip(['Metric', 'Fixed Effects', 'Random Effects', 'Mixed Effects (LME)'],
                   fit_tbl.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)
for row_data in [
    ('N observations',  str(N_obs),                      str(N_obs),                      str(N_obs)),
    ('N ZIP codes',     str(N_zip),                      str(N_zip),                      str(N_zip)),
    ('N years',         str(N_year),                     str(N_year),                     str(N_year)),
    ('R²',              f'{fe_res.rsquared:.4f}',        f'{re_res.rsquared:.4f}',        'N/A'),
    ('AIC',             'N/A',                           'N/A',                           f'{lme_res.aic:.2f}'),
    ('BIC',             'N/A',                           'N/A',                           f'{lme_res.bic:.2f}'),
    ('Hausman p-value', f'{H_pval:.4f}',                 f'{H_pval:.4f}',                 'N/A'),
]:
    r = fit_tbl.add_row()
    for i, v in enumerate(row_data):
        r.cells[i].text = v
        if i > 0: center(r.cells[i])
doc.add_paragraph()

# ── 3. Cross-Model Consistency ────────────────────────────────────────────────
doc.add_heading('3. Cross-Model Consistency', level=1)
doc.add_paragraph(
    'A predictor is most reliable when it reaches statistical significance (p < 0.05) '
    'in all three models. Results are grouped by how many models each predictor is '
    'significant in. Green coefficient = positive effect on QoL; Red = negative. '
    'Bold p-values are significant at p < 0.05.'
)

doc.add_heading('3.1 Significant in All Three Models — Strongest Evidence', level=2)
add_effect_bullets(doc, sig_all3, 'Consistent across FE, RE, and LME')

doc.add_heading('3.2 Significant in Two of Three Models', level=2)
add_effect_bullets(doc, sig_2of3, 'Significant in 2 of 3 models')

doc.add_heading('3.3 Significant in One Model Only', level=2)
doc.add_paragraph(
    'These effects may be model-specific. FE captures within-ZIP changes only; '
    'RE/LME also use between-ZIP differences. A predictor significant only in RE/LME '
    'may reflect structural between-ZIP differences rather than within-ZIP dynamics.')
add_effect_bullets(doc, sig_1of3, 'Significant in 1 of 3 models')

doc.add_heading('3.4 Not Significant in Any Model', level=2)
if len(sig_none):
    doc.add_paragraph('The following predictors show no significant association with QoL in any model:')
    p = doc.add_paragraph()
    p.add_run(', '.join([PLAIN_NAMES.get(v, v) for v in sig_none['Variable'].tolist()])).italic = True
doc.add_paragraph()

# ── 4. Fixed Effects ──────────────────────────────────────────────────────────
doc.add_heading('4. Fixed Effects Model — Within-ZIP Dynamics', level=1)
doc.add_paragraph(
    f'Within-ZIP R² = {fe_res.rsquared:.4f}. The FE model controls for all stable '
    f'ZIP-level characteristics. Only factors that change within a ZIP over time can '
    f'show effects here. Standard errors are clustered by ZIP code. '
    f'Interpretation: a coefficient of +0.10 means a 1 SD increase in the predictor '
    f'is associated with a 0.10 SD improvement in QoL within the same ZIP over time.'
)
add_pos_neg_bullets(doc, fe_tbl, 'Fixed Effects')

# ── 5. Random Effects ─────────────────────────────────────────────────────────
doc.add_heading('5. Random Effects Model — Within- and Between-ZIP Patterns', level=1)
doc.add_paragraph(
    f'R² = {re_res.rsquared:.4f}. The RE model pools within-ZIP changes and '
    f'cross-sectional differences across ZIP codes. It captures structural predictors '
    f'of QoL — both what changes over time and what differentiates communities. '
    f'Robust standard errors used.'
)
add_pos_neg_bullets(doc, re_tbl, 'Random Effects')

# ── 6. Mixed Effects ──────────────────────────────────────────────────────────
doc.add_heading('6. Mixed Effects Model — Shared Effects and ZIP Baselines', level=1)
doc.add_paragraph(
    f'AIC = {lme_res.aic:.2f}, BIC = {lme_res.bic:.2f}. '
    f'The LME model estimates shared predictor effects (fixed slopes) while allowing '
    f'each ZIP to have its own baseline QoL level (random intercept). '
    f'This separates two questions: which predictors matter everywhere, and which ZIPs '
    f'start higher or lower than average even after controlling for all predictors.'
)
add_pos_neg_bullets(doc, lme_tbl, 'Mixed Effects')

doc.add_heading('6.1 ZIP-Level Baseline QoL (Random Intercepts)', level=2)
doc.add_paragraph(
    'The random intercept reflects each ZIP\'s baseline QoL relative to the county '
    'average after accounting for all 23 predictors. A positive intercept means the ZIP '
    'performs better than its characteristics would predict; negative means it '
    'underperforms — signaling unmeasured disadvantages.'
)
top5    = re_intercepts.head(5)
bottom5 = re_intercepts.tail(5).sort_values('random_intercept')

ri_tbl = doc.add_table(rows=1, cols=3)
ri_tbl.style = 'Table Grid'
shade_row(ri_tbl.rows[0], 'BDD7EE')
for h, cell in zip(['ZIP Code', 'Random Intercept', 'Interpretation'], ri_tbl.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)
for rows_grp, color, label in [
    (top5,    'C6EFCE', 'Above-average baseline QoL'),
    (bottom5, 'FFC7CE', 'Below-average baseline QoL'),
]:
    for _, row in rows_grp.iterrows():
        r = ri_tbl.add_row()
        r.cells[0].text = str(int(row['zip_code'])); center(r.cells[0])
        val = row['random_intercept']
        r.cells[1].text = f"{'+' if val >= 0 else ''}{val:.3f}"; center(r.cells[1])
        shade_cell(r.cells[1], color)
        r.cells[2].text = label
doc.add_paragraph()

# ── 7. Pandemic Effect ────────────────────────────────────────────────────────
doc.add_heading('7. Pandemic Effect on QoL', level=1)
pan_tbl = doc.add_table(rows=1, cols=4)
pan_tbl.style = 'Table Grid'
shade_row(pan_tbl.rows[0], 'BDD7EE')
for h, cell in zip(['Model', 'Coefficient', 'p-value', 'Interpretation'], pan_tbl.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)
for model_name, tbl in [('Fixed Effects', fe_tbl), ('Random Effects', re_tbl), ('Mixed Effects', lme_tbl)]:
    b = get_val(tbl, 'Pandemic', 'Coef')
    p = get_val(tbl, 'Pandemic', 'p')
    sig = get_val(tbl, 'Pandemic', 'Sig')
    r = pan_tbl.add_row()
    r.cells[0].text = model_name
    r.cells[1].text = f"{b:.4f}" if not np.isnan(b) else 'N/A'; center(r.cells[1])
    shade_cell(r.cells[1], coef_color(b, p))
    r.cells[2].text = f"{p:.4f} {sig}" if not np.isnan(p) else 'N/A'; center(r.cells[2])
    shade_cell(r.cells[2], sig_color(p))
    if not np.isnan(p) and p < 0.05:
        r.cells[3].text = ('Pandemic significantly lowered QoL for older adults.' if b < 0
                           else 'Pandemic was associated with higher QoL scores.')
    else:
        r.cells[3].text = 'No statistically significant pandemic effect on QoL.'
doc.add_paragraph()

# ── 8. Technical Appendix ─────────────────────────────────────────────────────
doc.add_heading('8. Technical Appendix — Full Regression Tables', level=1)
doc.add_paragraph(
    'All variables are standardized (z-scored). Coefficients represent SD-unit changes in QoL. '
    'Green coefficient = positive and significant. Red = negative and significant. '
    'Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10'
)
doc.add_heading('8.1 Fixed Effects', level=2)
add_results_table(doc, fe_tbl, 'Table A1: Fixed Effects (Clustered SE by ZIP)')
doc.add_heading('8.2 Random Effects', level=2)
add_results_table(doc, re_tbl, 'Table A2: Random Effects (Robust SE)')
doc.add_heading('8.3 Mixed Effects', level=2)
add_results_table(doc, lme_tbl, 'Table A3: Mixed Effects (REML, Random Intercept by ZIP)')

doc.add_heading('8.4 Side-by-Side Comparison', level=2)
comp_tbl = doc.add_table(rows=1, cols=7)
comp_tbl.style = 'Table Grid'
shade_row(comp_tbl.rows[0], 'BDD7EE')
for h, cell in zip(['Variable', 'FE Coef', 'FE p', 'RE Coef', 'RE p', 'LME Coef', 'LME p'],
                   comp_tbl.rows[0].cells):
    cell.text = h; cell.paragraphs[0].runs[0].bold = True; center(cell)
for v in PREDICTORS:
    r = comp_tbl.add_row()
    r.cells[0].text = v
    for i, (tbl, col) in enumerate([(fe_tbl,'Coef'),(fe_tbl,'p'),
                                     (re_tbl,'Coef'),(re_tbl,'p'),
                                     (lme_tbl,'Coef'),(lme_tbl,'p')], 1):
        val = get_val(tbl, v, col)
        r.cells[i].text = f'{val:.4f}' if not np.isnan(val) else 'N/A'
        center(r.cells[i])
        if col == 'Coef': shade_cell(r.cells[i], coef_color(val, get_val(tbl, v, 'p')))
        if col == 'p':    shade_cell(r.cells[i], sig_color(val))
doc.add_paragraph()

doc.add_heading('8.5 Notes', level=2)
doc.add_paragraph(
    'Data source: arima_forecast_values_standardized_v3.csv\n'
    'Missing values imputed using MICE (10 iterations) prior to standardization.\n'
    'Seven variables had signs reversed before standardization so higher = better:\n'
    '  drive_time_hospital_min, pct_disabled_60_plus, pct_limited_english_60_plus,\n'
    '  pct_no_computer, pct_no_internet, pct_snap_households_60_plus,\n'
    '  share_older_adult_living_alone_households.\n'
    'Clustered SEs (by ZIP) for FE; robust SEs for RE; REML for LME.\n'
    'Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10'
)

out_path = r'C:\Users\emily\Downloads\longitudinal_analysis_qol_2015_2027.docx'
doc.save(out_path)
print(f'Saved: {out_path}')
