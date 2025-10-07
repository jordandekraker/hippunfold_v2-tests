#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ---------------------------
# Load metrics
# ---------------------------
df = pd.read_csv("metrics_per_subject.csv")

# ---------------------------
# ANOVA for Consistency
# ---------------------------
df_cons = df[df["measure"]=="consistency"].dropna(subset=["value"])
if not df_cons.empty:
    model_cons = smf.ols("value ~ C(condition) * C(dataset)", data=df_cons).fit()
    anova_cons = sm.stats.anova_lm(model_cons, typ=2)
    print("\n=== ANOVA: Consistency ===")
    print(anova_cons)

# ---------------------------
# ANOVA for Identifiability
# ---------------------------
df_ident = df[df["measure"]=="identifiability"].dropna(subset=["value"])
if not df_ident.empty:
    model_ident = smf.ols("value ~ C(condition) * C(dataset)", data=df_ident).fit()
    anova_ident = sm.stats.anova_lm(model_ident, typ=2)
    print("\n=== ANOVA: Identifiability ===")
    print(anova_ident)

# ---------------------------
# t-test for Generalizability (condition: T1w vs synthseg_v0.2)
# ---------------------------
lastcol = df.columns[-1]
mask = df["measure"] == "generalizability"
df.loc[mask, "dataset"] = df.loc[mask, lastcol]
df = df.drop(columns=[lastcol])

df_ident = df[df["measure"]=="generalizability"].dropna(subset=["value"])
if not df_ident.empty:
    model_ident = smf.ols("value ~ C(condition) * C(dataset)", data=df_ident).fit()
    anova_ident = sm.stats.anova_lm(model_ident, typ=2)
    print("\n=== ANOVA: Generalizability ===")
    print(anova_ident)