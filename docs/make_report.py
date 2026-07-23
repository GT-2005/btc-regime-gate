#!/usr/bin/env python3
"""
docs/make_report.py — builds the LATENT technical report as a PDF.

    python docs/make_report.py

Kept as a script rather than a static file so the report can be regenerated
whenever the system changes. A report that drifts out of sync with the code
is worse than no report.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

OUT = Path(__file__).resolve().parent / "LATENT_Technical_Report.pdf"

INK = colors.HexColor("#0F1B2D")
ACCENT = colors.HexColor("#FF5A1F")
SOFT = colors.HexColor("#4C5B73")
RULE = colors.HexColor("#C6D0DC")
FAINT = colors.HexColor("#EEF1F5")

# ── styles ───────────────────────────────────────────────────────────
S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=30,
                            textColor=INK, leading=33, spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=12.5,
                          textColor=SOFT, leading=18, spaceAfter=22),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16,
                         textColor=INK, leading=20, spaceBefore=22, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5,
                         textColor=INK, leading=15, spaceBefore=14, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.6,
                           textColor=INK, leading=14.8, spaceAfter=8,
                           alignment=TA_JUSTIFY),
    "mono": ParagraphStyle("mono", fontName="Courier", fontSize=8.4,
                           textColor=INK, leading=12.5, spaceAfter=8,
                           backColor=FAINT, borderPadding=7),
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.2,
                          textColor=SOFT, leading=12, spaceAfter=12),
    "warn": ParagraphStyle("warn", fontName="Helvetica", fontSize=9,
                           textColor=INK, leading=14, spaceAfter=7),
    "eyebrow": ParagraphStyle("eyebrow", fontName="Courier", fontSize=7.6,
                              textColor=SOFT, leading=11, spaceAfter=16),
}


def P(t, s="body"):
    return Paragraph(t, S[s])


def rule(c=RULE, w=0.7, before=2, after=12):
    return HRFlowable(width="100%", thickness=w, color=c,
                      spaceBefore=before, spaceAfter=after)


def table(rows, widths, header=True, mono_col=None):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7),
            ("TEXTCOLOR", (0, 0), (-1, 0), SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ]
    if mono_col is not None:
        style.append(("FONTNAME", (mono_col, 1), (mono_col, -1), "Courier"))
    t.setStyle(TableStyle(style))
    return t


def box(paras, border=ACCENT):
    t = Table([[p] for p in paras], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), FAINT),
    ]))
    return t


# ── page furniture ───────────────────────────────────────────────────
def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Courier", 7)
    canvas.setFillColor(SOFT)
    canvas.drawString(22 * mm, 12 * mm, "LATENT — TECHNICAL REPORT")
    canvas.drawRightString(A4[0] - 22 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(22 * mm, 16 * mm, A4[0] - 22 * mm, 16 * mm)
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=22 * mm,
        title="LATENT — Technical Report", author="Ganesh Tilekar",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])

    s: list = []

    # ── cover ────────────────────────────────────────────────────────
    s += [
        Spacer(1, 40 * mm),
        P("REGIME-AWARE SIGNAL ENGINE&nbsp;&nbsp;·&nbsp;&nbsp;BTC/USDT&nbsp;&nbsp;·&nbsp;&nbsp;1H", "eyebrow"),
        P("LATENT", "title"),
        P("A technical report on a regime-aware BTC/USDT research system: "
          "Kalman filtering for direction, a hidden-Markov model for market "
          "regime, GARCH for volatility, a stacked ensemble for the decision, "
          "and fractional Kelly for the size.", "sub"),
        rule(INK, 1.4, 0, 18),
        table([
            ["Author", "Ganesh Tilekar"],
            ["Repository", "github.com/GT-2005/btc-regime-gate"],
            ["Licence", "MIT"],
            ["Version", "0.1.0"],
            ["Status", "Research. Backtested, not live traded."],
        ], [38 * mm, 127 * mm], header=False),
        Spacer(1, 16 * mm),
        box([
            P("<b>Read this before anything else</b>", "warn"),
            P("This document describes a research tool. It is not investment "
              "advice, not financial advice, and not a recommendation to buy or "
              "sell any asset. No advisory relationship is created by reading it.", "warn"),
            P("Every performance figure in this report comes from historical "
              "backtesting. There is no live track record. Hypothetical results "
              "are prepared with hindsight, do not represent actual trading, and "
              "cannot account for the effect of financial risk in live markets.", "warn"),
            P("Trading cryptocurrency involves substantial risk of loss, including "
              "loss of your entire capital. You are solely responsible for your own "
              "decisions and for complying with the laws of your jurisdiction.", "warn"),
        ]),
        PageBreak(),
    ]

    # ── 1. introduction ──────────────────────────────────────────────
    s += [
        P("1 &nbsp; Introduction", "h1"), rule(),
        P("Most technical indicators are functions of price that react to price. "
          "A moving average tells you where price has been. An oscillator tells "
          "you whether the recent move was large relative to other recent moves. "
          "Neither attempts to say anything about the <i>state</i> the market is "
          "currently in, and yet the same signal means very different things in a "
          "trending market and a choppy one."),
        P("LATENT is built around the opposite premise: estimate the state first, "
          "then decide. Three unobservable quantities are estimated at every bar — "
          "the underlying trend and its velocity, the prevailing market regime, and "
          "the volatility about to be experienced. Only then does a model decide "
          "whether to act, and a separate layer decides how much."),

        P("1.1 &nbsp; Objectives", "h2"),
        P("&bull; Build a noise-filtered, regime-aware signal pipeline.<br/>"
          "&bull; Engineer a dimensionless feature set from raw OHLCV alone.<br/>"
          "&bull; Train a stacked ensemble producing a calibrated probability per bar.<br/>"
          "&bull; Validate through walk-forward testing so no future information leaks.<br/>"
          "&bull; Size positions using fractional Kelly with hard risk caps.<br/>"
          "&bull; Report through institutional-standard metrics, including the unflattering ones."),

        P("1.2 &nbsp; Architecture", "h2"),
        P("Five layers. Raw candles enter on the left, a sized order leaves on the "
          "right, and nothing downstream can see the future."),
        table([
            ["LAYER", "COMPONENTS", "RESPONSIBILITY"],
            ["Data", "load_data, add_base_features",
             "Ingest hourly OHLCV, compute 30+ normalised columns"],
            ["Signal", "Kalman, HMM, Hurst, OFI, VWAP-z",
             "Directional, regime and statistical state estimates"],
            ["Model", "XGBoost, LightGBM, meta-logistic",
             "One calibrated probability per bar"],
            ["Risk", "Fractional Kelly, GARCH, drawdown brake",
             "How much to commit, and when not to"],
            ["Execution", "6-condition gate, ATR TP/SL, timeout",
             "Order placement, exit, realistic cost simulation"],
        ], [26 * mm, 55 * mm, 84 * mm]),
        P("The separation is deliberate. If any single model fails to converge — "
          "the HMM, most commonly — the pipeline filters to whatever feature columns "
          "actually exist and continues in degraded mode. It would rather run on 25 "
          "features than crash on 29. That matters when real capital is involved.", "cap"),
    ]

    # ── 2. data ──────────────────────────────────────────────────────
    s += [
        P("2 &nbsp; Data pipeline", "h1"), rule(),
        P("2.1 &nbsp; Dependencies", "h2"),
        table([
            ["LIBRARY", "ROLE"],
            ["NumPy", "Array mathematics throughout"],
            ["Pandas", "DataFrames, rolling windows, EWM"],
            ["SciPy", "norm.logpdf for HMM emissions; optimize.minimize (L-BFGS-B) for GARCH MLE"],
            ["XGBoost", "XGBClassifier — 600 trees, depth 5, learning rate 0.03"],
            ["LightGBM", "LGBMClassifier — 600 estimators, 40 leaves, learning rate 0.03"],
            ["scikit-learn", "LogisticRegression meta-learner, StandardScaler, roc_auc_score"],
            ["Matplotlib", "Equity curve and drawdown chart"],
            ["openpyxl", "Reading the Binance .xlsx export"],
        ], [30 * mm, 135 * mm]),
        P("The Kalman filter, the hidden-Markov model and the Hurst exponent are "
          "implemented directly rather than imported. This keeps the dependency "
          "list short enough to install anywhere, and means the mathematics is "
          "readable rather than hidden behind an API."),

        P("2.2 &nbsp; Loading and sorting", "h2"),
        P("The loader accepts .xlsx or .csv, normalises column names against a "
          "table of common aliases, coerces types, drops unparseable rows, and "
          "then does the single most important thing in the entire codebase:"),
        Paragraph("df = df.sort_values(\"date\").drop_duplicates(subset=\"date\", keep=\"last\")", S["mono"]),
        box([P("<b>Why sorting is not a detail.</b> If even a handful of rows sit "
               "out of chronological order, every rolling window and the target "
               "column will silently incorporate future information into past rows. "
               "The backtest will look extraordinary and it will be fiction. This is "
               "the most common way a research result turns out to be worthless.", "warn")]),

        P("2.3 &nbsp; Validation", "h2"),
        P("A separate <font face='Courier'>validate()</font> pass reports problems in plain "
          "language rather than failing silently: too few rows for the folds, gaps "
          "in the series, bars where high sits below low, open or close outside the "
          "high-low range, and an implausible share of zero-volume bars. Running "
          "with <font face='Courier'>--validate</font> performs these checks and exits."),
    ]

    # ── 3. features ──────────────────────────────────────────────────
    s += [
        PageBreak(),
        P("3 &nbsp; Feature engineering", "h1"), rule(),
        P("Wherever possible features are dimensionless. A feature that scales with "
          "price teaches the model that Bitcoin was cheap in 2020, which is true "
          "and useless. Dividing by ATR or by price makes the same information "
          "portable across price levels and across time."),
        table([
            ["GROUP", "COLUMNS", "NOTE"],
            ["Returns", "ret, log_ret",
             "Log returns add over time, which makes multi-period analysis cleaner"],
            ["Momentum", "mom_3, mom_5, mom_10",
             "Three horizons: short, medium, slightly longer"],
            ["Trend", "ema_9/21/50/200, fast_slow, trend_up",
             "fast_slow normalises the EMA gap so it survives a doubling in price"],
            ["Volatility", "atr, atr_pct, realised_vol",
             "atr_pct rather than raw ATR enters the model"],
            ["Oscillators", "rsi, rsi_smooth, macd, macd_hist",
             "MACD divided by price; RSI smoothed over 5 bars to reduce jitter"],
            ["Participation", "vol_ratio, vol_trend",
             "Distinguishes a genuine volume surge from a one-off spike"],
            ["Candle shape", "body_pct, upper_wick, lower_wick, range_pct",
             "All divided by ATR. A long upper wick hints at selling pressure above"],
        ], [26 * mm, 62 * mm, 77 * mm]),

        P("3.1 &nbsp; Target", "h2"),
        P("The supervised target is binary: did the close rise over the following "
          "five bars? Formally <font face='Courier'>target = 1 if close[t+5] &gt; close[t]</font>. "
          "The final five rows have no future to look at and are excluded from "
          "training rather than filled."),
    ]

    # ── 4. signal layer ──────────────────────────────────────────────
    s += [
        P("4 &nbsp; Signal layer", "h1"), rule(),

        P("4.1 &nbsp; Kalman filter", "h2"),
        P("A constant-velocity Kalman filter runs over the close series. The state "
          "is [level, velocity]; the filter observes level only and infers velocity. "
          "Each bar performs a predict step and an update step, weighting the new "
          "observation against the accumulated estimate by the Kalman gain."),
        Paragraph("predict:  x_p = F x            P_p = F P F' + Q\n"
                  "update :  y   = z - H x_p     S   = H P_p H' + R\n"
                  "          K   = P_p H' / S    x   = x_p + K y", S["mono"]),
        P("Two parameters control behaviour. <font face='Courier'>process_var</font> encodes how "
          "much you believe the trend genuinely changes; raising it makes the filter "
          "quicker and noisier. <font face='Courier'>measure_var</font> encodes how noisy you "
          "believe observed price to be; raising it smooths harder. The output "
          "<font face='Courier'>kf_velocity</font> is divided by price so it remains comparable "
          "at any level."),

        P("4.2 &nbsp; Hidden Markov model", "h2"),
        P("Three states with Gaussian emissions over log returns, fitted by "
          "Baum-Welch. The entire forward-backward pass runs in log space; on "
          "sequences of several thousand bars, probabilities in linear space "
          "underflow to zero within a few hundred steps."),
        P("Nothing tells the model what &ldquo;bear&rdquo; means. It finds three return "
          "distributions, and the states are relabelled afterwards in ascending "
          "order of mean return, which is why the labels are stable between runs. "
          "The transition matrix is initialised with 0.9 on the diagonal, encoding "
          "the prior that regimes persist rather than alternate bar to bar."),
        P("Fitting is capped at the most recent 8,000 observations. Baum-Welch is "
          "inherently sequential and its cost grows linearly with sequence length; "
          "prediction still runs over the full history.", "cap"),

        P("4.3 &nbsp; Hurst exponent", "h2"),
        P("Estimated by rescaled range over a rolling window. H above 0.5 indicates "
          "a trending series where momentum logic is appropriate; below 0.5 "
          "indicates mean reversion; around 0.5 is a random walk. Recomputed every "
          "five bars rather than every bar — over a 100-bar window the value barely "
          "moves between adjacent bars, and recomputing it each time costs "
          "substantial runtime for no additional information."),

        P("4.4 &nbsp; Microstructure proxies", "h2"),
        P("True order-flow imbalance requires the order book, which OHLCV does not "
          "contain. The proxy used is where each bar closed within its own range, "
          "weighted by volume: closing near the high means buyers controlled that "
          "bar. <font face='Courier'>vwap_z</font> measures distance from rolling VWAP in "
          "standard deviations, clipped to &plusmn;5 so a single dislocation cannot "
          "dominate the feature."),
    ]

    # ── 5. GARCH ─────────────────────────────────────────────────────
    s += [
        PageBreak(),
        P("5 &nbsp; Volatility modelling", "h1"), rule(),
        P("Volatility clusters. Quiet hours follow quiet hours and violent ones "
          "follow violent ones. A rolling standard deviation reports what "
          "volatility <i>was</i>; GARCH forecasts what it is about to be, which is "
          "the quantity actually needed when deciding position size."),
        Paragraph("sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2", S["mono"]),
        P("Parameters are estimated by maximum likelihood using L-BFGS-B under "
          "bounds that keep the process stationary (alpha + beta &lt; 0.999) and "
          "the variance positive. If the optimiser fails to converge, the model "
          "falls back to sensible defaults rather than raising — the pipeline is "
          "designed to degrade, not die."),
        P("Persistence (alpha + beta) close to 1 means shocks decay slowly. Values "
          "in the region of 0.97 are normal for crypto and are not a sign of "
          "misfit. The derived <font face='Courier'>vol_regime</font> column is forecast "
          "volatility divided by its own expanding average: above 1 means "
          "conditions are hotter than this market's own normal."),
    ]

    # ── 6. ML layer ──────────────────────────────────────────────────
    s += [
        P("6 &nbsp; Model layer", "h1"), rule(),
        P("6.1 &nbsp; Base learners and stacking", "h2"),
        P("XGBoost and LightGBM are trained on the same feature matrix. Both are "
          "gradient-boosted tree ensembles but they split differently — LightGBM "
          "grows leaf-wise and uses histogram binning — so their errors are "
          "usefully decorrelated. A logistic regression sits on top and learns "
          "when to trust which."),
        box([P("<b>The meta-learner must not see its base models' training data.</b> "
               "If the stack is fitted on predictions the base models made about "
               "rows they were trained on, it learns their overfitting rather than "
               "their signal. Here the base models train on the first 75% of the "
               "training slice and the meta-learner is fitted on their predictions "
               "over the held-out remainder.", "warn")]),
        P("Every split is chronological. Shuffling a time series before splitting "
          "places future information in the training set and is the second most "
          "common way a backtest becomes meaningless."),

        P("6.2 &nbsp; Walk-forward validation", "h2"),
        P("The usable data is divided into sequential folds. Fold k trains on "
          "everything preceding its test window and is scored only on that window. "
          "The model is refitted at each fold rather than carried forward, so each "
          "result reflects what would have been known at that time."),
        P("Train once and test once gives you a number. This gives you a range, "
          "and the range is the honest answer.", "cap"),

        P("6.3 &nbsp; Interpreting AUC", "h2"),
        P("On financial data an AUC of 0.60 is a strong result; 0.50 is a coin "
          "flip. The stack typically improves on either base model by 0.01 to 0.03, "
          "which sounds negligible until compounded over a thousand trades. If AUC "
          "sits at 0.50 the model has found nothing, and the correct response is to "
          "distrust the strategy rather than to lower the threshold until it trades."),
    ]

    # ── 7. risk ──────────────────────────────────────────────────────
    s += [
        PageBreak(),
        P("7 &nbsp; Risk layer", "h1"), rule(),
        P("7.1 &nbsp; Fractional Kelly", "h2"),
        Paragraph("f* = (p * b - q) / b       q = 1 - p,  b = reward / risk", S["mono"]),
        P("Kelly is growth-optimal <i>given a known edge</i>. The edge here is an "
          "estimate from a model with an AUC around 0.6, which is emphatically not "
          "known. Half Kelly is the standard compromise: it gives up roughly a "
          "quarter of the growth rate for a large reduction in the probability of "
          "a deep drawdown. Negative edge returns zero — declining to bet is a "
          "valid bet."),

        P("7.2 &nbsp; Caps and the drawdown brake", "h2"),
        table([
            ["PARAMETER", "DEFAULT", "CONTROLS"],
            ["starting_capital", "100,000", "Baseline for all return and drawdown figures"],
            ["max_risk_per_trade", "0.01", "If the stop is hit, the loss cannot exceed 1% of equity"],
            ["max_drawdown_limit", "0.15", "Emergency brake — halts new entries"],
            ["max_position_frac", "0.20", "Concentration cap, even if Kelly asks for more"],
            ["signal_buy_thresh", "0.62", "Confidence required before a long is considered"],
            ["signal_sell_thresh", "0.38", "Probability at or below which a short is considered"],
        ], [40 * mm, 22 * mm, 103 * mm], mono_col=0),
        P("The brake engages when drawdown passes the limit and releases only once "
          "equity recovers to within half of it. A system that keeps trading "
          "through its worst stretch is how a bad month becomes a blown account. "
          "Volatility scaling is applied before the caps: forecast volatility above "
          "its long-run average divides the size, so exposure falls before the "
          "storm rather than after it."),
    ]

    # ── 8. execution ─────────────────────────────────────────────────
    s += [
        P("8 &nbsp; Execution", "h1"), rule(),
        P("8.1 &nbsp; The entry gate", "h2"),
        P("Six conditions must all pass before a position opens:"),
        table([
            ["#", "CONDITION", "PURPOSE"],
            ["1", "probability beyond threshold", "The model is actually confident"],
            ["2", "RSI between 30 and 70", "Do not buy into an exhausted move"],
            ["3", "ATR finite and positive", "Volatility is measurable, so stops can be placed"],
            ["4", "cooldown since last exit", "Prevents re-entering the same noise"],
            ["5", "drawdown brake off", "Risk layer has not halted trading"],
            ["6", "vol_regime below 3", "Not in a volatility spike"],
        ], [10 * mm, 62 * mm, 93 * mm]),

        P("8.2 &nbsp; Exits and cost model", "h2"),
        P("Stop-loss and take-profit are both ATR-scaled, so they widen in volatile "
          "conditions and tighten in quiet ones without any parameter changing. A "
          "trending tape, indicated by Hurst above 0.55, earns a 15% wider target. "
          "Any position still open after 25 bars is closed at market — a thesis "
          "that has not worked within a day has usually stopped being a thesis."),
        P("The simulation is deliberately pessimistic. Commission and slippage are "
          "charged on both entry and exit. If both the stop and the target fall "
          "inside the same bar, the stop is assumed to have been hit first. Real "
          "fills are worse than modelled fills, and a backtest that assumes "
          "otherwise is telling you what you want to hear."),
    ]

    # ── 9. metrics ───────────────────────────────────────────────────
    s += [
        PageBreak(),
        P("9 &nbsp; Metrics and reporting", "h1"), rule(),
        table([
            ["METRIC", "WHAT IT TELLS YOU"],
            ["Sharpe", "Return per unit of total risk, annualised for hourly bars"],
            ["Sortino", "The same, but upside surprises are not counted as risk"],
            ["Calmar", "Annualised return against the worst drawdown endured"],
            ["Max drawdown", "The deepest hole the equity curve digs"],
            ["Profit factor", "Gross wins over gross losses; below 1.3 is not viable"],
            ["VaR 95%", "The loss that should be exceeded on only 5% of trades"],
            ["Longest losing run", "How many consecutive losses to expect. Live will be worse"],
            ["Time under water", "Fraction of the period spent below a previous peak"],
        ], [36 * mm, 129 * mm]),
        P("Time under water and longest losing run are included deliberately. They "
          "are the numbers that determine whether a system can actually be held "
          "through a bad stretch, and they are almost never quoted in marketing "
          "material for precisely that reason."),
        P("Results are additionally split by the regime each trade was opened in. "
          "A system that only works in one regime is a system waiting for that "
          "regime to end, and the per-regime table is where that becomes visible."),

        P("9.1 &nbsp; Output files", "h2"),
        table([
            ["FILE", "CONTENTS"],
            ["metrics.csv", "Headline figures"],
            ["trades.csv", "Every trade: entry, exit, size, regime, exit reason, P&L"],
            ["regime_breakdown.csv", "Metrics split by bear / sideways / bull"],
            ["folds.csv", "Per-fold rows, AUCs, probability ranges, returns"],
            ["feature_importance.csv", "Averaged gain importance across both base models"],
            ["equity_curve.png", "Equity and drawdown"],
        ], [42 * mm, 123 * mm], mono_col=0),
    ]

    # ── 10. results ──────────────────────────────────────────────────
    s += [
        P("10 &nbsp; Results", "h1"), rule(),
        box([P("<b>This section is intentionally left for you to complete.</b>", "warn"),
             P("The numbers below are the design targets taken from the original "
               "system specification, not measured output. Fill them in from your "
               "own run against real market data, and delete this box once you "
               "have. Publishing target figures as though they were measured is "
               "the exact failure this report is written to avoid.", "warn")]),
        table([
            ["METRIC", "DESIGN TARGET", "YOUR RESULT"],
            ["Net return", "~ +20%", "&nbsp;"],
            ["Win rate", "58 – 65%", "&nbsp;"],
            ["Sharpe", "1.4 – 2.0", "&nbsp;"],
            ["Max drawdown", "< 8%", "&nbsp;"],
            ["Profit factor", "> 1.5", "&nbsp;"],
            ["Model AUC", "~ 0.60", "&nbsp;"],
        ], [50 * mm, 45 * mm, 70 * mm]),
        P("Expected regime shape, again as a design target: bear regimes producing "
          "15–25% of trades at 45–55% win rate; sideways 35–45% of trades at 55–65%; "
          "bull 35–45% at 60–70%. If a real run inverts that pattern entirely, "
          "suspect the data before crediting the model.", "cap"),
    ]

    # ── 11. limitations ──────────────────────────────────────────────
    s += [
        P("11 &nbsp; Limitations", "h1"), rule(),
        P("&bull; <b>No live track record.</b> Everything here is backtested.<br/>"
          "&bull; <b>Tuned for BTC/USDT 1H.</b> Output on other markets is untested.<br/>"
          "&bull; <b>Bear regimes are the weak point</b>, close to break-even.<br/>"
          "&bull; <b>The HMM does not always converge</b> on short datasets.<br/>"
          "&bull; <b>Backtests are optimistic by construction.</b> Real fills are worse.<br/>"
          "&bull; <b>Edges decay.</b> A model that tested well may stop working, and this is "
          "the normal life cycle of a strategy rather than a defect in it."),

        P("11.1 &nbsp; On repeated testing", "h2"),
        P("It takes roughly twenty iterations of adjusting and re-testing on the "
          "same data to discover a false strategy at conventional significance "
          "levels. Walk-forward validation reduces this risk but does not remove "
          "it. Feature importance analysis is a better research tool than "
          "repeatedly re-running a backtest until it looks good."),

        rule(INK, 1.2, 20, 14),
        P("<b>Risk disclosure</b>", "h2"),
        P("LATENT is an analytical tool. It is not investment advice, financial "
          "advice, or a recommendation to buy or sell any asset. No advisory or "
          "fiduciary relationship is created by your use of it. Trading "
          "cryptocurrency involves substantial risk of loss including loss of "
          "your entire capital, and leveraged trading can produce losses exceeding "
          "your deposit. All performance figures are derived from historical "
          "backtests; hypothetical results are prepared with hindsight, do not "
          "represent actual trading, and cannot fully account for the effect of "
          "financial risk in live markets. Past performance, real or simulated, "
          "does not indicate future results. You are solely responsible for your "
          "own trading decisions and for complying with the laws and regulations "
          "of your jurisdiction."),
    ]

    doc.build(s)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
