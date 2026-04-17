#!/usr/bin/env python3
"""
FMCG Data Chat System with Interactive AI
Enhanced version with better Groq AI integration
"""

import pandas as pd
import re
import sys
import os
from textwrap import dedent
from groq import Groq

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────

print("Loading data...")

df_main  = pd.read_csv("data/weekly_df_final_for_modeling.csv")
df_fmcg  = pd.read_csv("data/FMCG_2022_2024.csv")
df_mi006 = pd.read_csv("data/df_weekly_MI-006_enriched.csv")

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("⚠️  Warning: GROQ_API_KEY not set. AI analysis features will be limited.")
    client = None
else:
    client = Groq(api_key=GROQ_API_KEY)

# Normalise column names to lowercase
for df in [df_main, df_fmcg, df_mi006]:
    df.columns = [c.lower().strip() for c in df.columns]

# Ensure date columns are parsed
df_main["week"]  = pd.to_datetime(df_main["week"])
df_fmcg["date"]  = pd.to_datetime(df_fmcg["date"])
df_mi006["week"] = pd.to_datetime(df_mi006["week"])

print("Data loaded.\n")

# ──────────────────────────────────────────────
# 2. HELPER UTILITIES
# ──────────────────────────────────────────────

ALL_SKUS      = sorted(df_fmcg["sku"].unique())
ALL_BRANDS    = sorted(df_fmcg["brand"].unique())
ALL_CATEGORIES= sorted(df_fmcg["category"].unique())
ALL_CHANNELS  = sorted(df_fmcg["channel"].unique())
ALL_REGIONS   = sorted(df_fmcg["region"].unique())

def fmt_num(n):
    """Format a number nicely."""
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{n:,}"

def extract_sku(text):
    """Find a SKU mention in the user's text."""
    text_up = text.upper()
    for sku in ALL_SKUS:
        if sku.upper() in text_up:
            return sku
    return None

def extract_channel(text):
    text_l = text.lower()
    for ch in ALL_CHANNELS:
        if ch.lower() in text_l:
            return ch
    return None

def extract_region(text):
    text_l = text.lower()
    for r in ALL_REGIONS:
        if r.lower() in text_l:
            return r
    return None

def extract_year(text):
    m = re.search(r'\b(202[2-4])\b', text)
    return int(m.group(1)) if m else None

def extract_category(text):
    text_l = text.lower()
    for cat in ALL_CATEGORIES:
        if cat.lower() in text_l:
            return cat
    return None

def extract_brand(text):
    text_l = text.lower()
    for b in ALL_BRANDS:
        if b.lower() in text_l:
            return b
    return None

# ──────────────────────────────────────────────
# 3. QUERY HANDLERS
# ──────────────────────────────────────────────

def handle_total_sales(text):
    """Total units sold — optionally filtered."""
    df = df_fmcg.copy()
    filters = []

    sku = extract_sku(text)
    channel = extract_channel(text)
    region = extract_region(text)
    year = extract_year(text)
    category = extract_category(text)
    brand = extract_brand(text)

    if sku:
        df = df[df["sku"] == sku]; filters.append(f"SKU {sku}")
    if channel:
        df = df[df["channel"] == channel]; filters.append(f"channel {channel}")
    if region:
        df = df[df["region"] == region]; filters.append(f"region {region}")
    if year:
        df = df[df["date"].dt.year == year]; filters.append(f"year {year}")
    if category:
        df = df[df["category"] == category]; filters.append(f"category {category}")
    if brand:
        df = df[df["brand"] == brand]; filters.append(f"brand {brand}")

    total = df["units_sold"].sum()
    label = " | ".join(filters) if filters else "all products"
    return f"Total units sold ({label}): {fmt_num(total)}"


def handle_top_skus(text):
    """Top N SKUs by sales."""
    m = re.search(r'\b(\d+)\b', text)
    n = int(m.group(1)) if m else 5
    year = extract_year(text)
    channel = extract_channel(text)

    df = df_fmcg.copy()
    if year:
        df = df[df["date"].dt.year == year]
    if channel:
        df = df[df["channel"] == channel]

    top = df.groupby("sku")["units_sold"].sum().nlargest(n).reset_index()
    lines = [f"  {i+1}. {row['sku']}: {fmt_num(row['units_sold'])} units"
             for i, row in top.iterrows()]
    return "Top {} SKUs by units sold:\n".format(n) + "\n".join(lines)


def handle_promotion_impact(text):
    """Compare sales during promotions vs non-promotions."""
    sku = extract_sku(text)
    df = df_fmcg.copy()
    if sku:
        df = df[df["sku"] == sku]

    promo    = df[df["promotion_flag"] == 1]["units_sold"].mean()
    no_promo = df[df["promotion_flag"] == 0]["units_sold"].mean()
    lift = ((promo - no_promo) / no_promo * 100) if no_promo else 0

    label = f" for {sku}" if sku else ""
    return (f"Promotion impact{label}:\n"
            f"  Avg sales WITH promotion:    {fmt_num(promo)}\n"
            f"  Avg sales WITHOUT promotion: {fmt_num(no_promo)}\n"
            f"  Lift: {lift:+.1f}%")


def handle_channel_breakdown(text):
    """Sales breakdown by channel."""
    sku = extract_sku(text)
    year = extract_year(text)
    df = df_fmcg.copy()

    if sku:
        df = df[df["sku"] == sku]
    if year:
        df = df[df["date"].dt.year == year]

    breakdown = df.groupby("channel")["units_sold"].sum().sort_values(ascending=False)
    lines = [f"  {ch}: {fmt_num(v)}" for ch, v in breakdown.items()]
    label = (f" for {sku}" if sku else "") + (f" in {year}" if year else "")
    return f"Sales by channel{label}:\n" + "\n".join(lines)


def handle_region_breakdown(text):
    """Sales breakdown by region."""
    sku = extract_sku(text)
    year = extract_year(text)
    df = df_fmcg.copy()

    if sku:
        df = df[df["sku"] == sku]
    if year:
        df = df[df["date"].dt.year == year]

    breakdown = df.groupby("region")["units_sold"].sum().sort_values(ascending=False)
    lines = [f"  {r}: {fmt_num(v)}" for r, v in breakdown.items()]
    label = (f" for {sku}" if sku else "") + (f" in {year}" if year else "")
    return f"Sales by region{label}:\n" + "\n".join(lines)


def handle_sku_info(text):
    """Show summary info for a specific SKU."""
    sku = extract_sku(text)
    if not sku:
        return "Please specify a SKU (e.g. MI-006, JU-021). Type 'list skus' to see all."

    df = df_fmcg[df_fmcg["sku"] == sku]
    brand    = df["brand"].iloc[0]
    category = df["category"].iloc[0]
    segment  = df["segment"].iloc[0] if "segment" in df.columns else "N/A"
    total    = df["units_sold"].sum()
    avg_price= df["price_unit"].mean()
    promo_pct= df["promotion_flag"].mean() * 100
    date_range = f"{df['date'].min().date()} → {df['date'].max().date()}"

    return dedent(f"""
        SKU: {sku}
          Brand:       {brand}
          Category:    {category}
          Segment:     {segment}
          Total sold:  {fmt_num(total)} units
          Avg price:   {fmt_num(avg_price)}
          Promo weeks: {promo_pct:.1f}%
          Date range:  {date_range}
    """).strip()


def handle_stock(text):
    """Average stock availability."""
    sku = extract_sku(text)
    channel = extract_channel(text)
    df = df_fmcg.copy()

    if sku:
        df = df[df["sku"] == sku]
    if channel:
        df = df[df["channel"] == channel]

    avg_stock = df["stock_available"].mean()
    label = []
    if sku: label.append(f"SKU {sku}")
    if channel: label.append(f"channel {channel}")
    label = " | ".join(label) if label else "all"
    return f"Average stock available ({label}): {fmt_num(avg_stock)}"


def handle_category_breakdown(text):
    """Sales by category."""
    year = extract_year(text)
    df = df_fmcg.copy()
    if year:
        df = df[df["date"].dt.year == year]

    breakdown = df.groupby("category")["units_sold"].sum().sort_values(ascending=False)
    lines = [f"  {cat}: {fmt_num(v)}" for cat, v in breakdown.items()]
    return ("Category sales" + (f" in {year}" if year else "") + ":\n" + "\n".join(lines))


def handle_yearly_trend(text):
    """Year-over-year total sales."""
    sku = extract_sku(text)
    category = extract_category(text)
    df = df_fmcg.copy()

    if sku:
        df = df[df["sku"] == sku]
    if category:
        df = df[df["category"] == category]

    trend = df.groupby(df["date"].dt.year)["units_sold"].sum()
    lines = [f"  {yr}: {fmt_num(v)}" for yr, v in trend.items()]
    label = sku or category or "all"
    return f"Yearly sales trend ({label}):\n" + "\n".join(lines)


def handle_list_skus(text):
    category = extract_category(text)
    if category:
        skus = sorted(df_fmcg[df_fmcg["category"] == category]["sku"].unique())
        return f"SKUs in {category}:\n  " + ", ".join(skus)
    return "All SKUs:\n  " + ", ".join(ALL_SKUS)


def handle_list_categories(_):
    return "Categories: " + ", ".join(ALL_CATEGORIES)


def handle_list_brands(_):
    brands_by_cat = df_fmcg.groupby("category")["brand"].unique()
    lines = [f"  {cat}: {', '.join(sorted(brands))}" for cat, brands in brands_by_cat.items()]
    return "Brands by category:\n" + "\n".join(lines)


def handle_mi006_detail(text):
    """Detailed MI-006 enriched data queries."""
    df = df_mi006.copy()
    year = extract_year(text)
    channel = extract_channel(text)

    if year:
        df = df[df["week"].dt.year == year]
    if channel:
        df = df[df["channel"] == channel]

    avg_temp   = df["avg_temp"].mean() if "avg_temp" in df.columns else None
    avg_infl   = df["inflation_index"].mean() if "inflation_index" in df.columns else None
    total_sold = df["units_sold"].sum()
    avg_cattrend = df["category_trend"].mean() if "category_trend" in df.columns else None

    label = []
    if year: label.append(str(year))
    if channel: label.append(channel)
    label = " | ".join(label) if label else "full dataset"

    lines = [f"MI-006 enriched summary ({label}):"]
    lines.append(f"  Total units sold:    {fmt_num(total_sold)}")
    if avg_temp is not None:
        lines.append(f"  Avg temperature:    {avg_temp:.2f}")
    if avg_infl is not None:
        lines.append(f"  Avg inflation idx:  {avg_infl:.2f}")
    if avg_cattrend is not None:
        lines.append(f"  Avg category trend: {avg_cattrend:.3f}")
    return "\n".join(lines)


def get_ai_analysis(user_query, data_context):
    """Get AI-powered analysis using Groq API."""
    if not client:
        return None
    
    try:
        message = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert FMCG data analyst. Analyze the provided data context "
                        "and answer the user's question with actionable insights, trends, patterns, "
                        "and predictions. Be concise but insightful. Focus on business implications."
                    )
                },
                {
                    "role": "user",
                    "content": f"User Query: {user_query}\n\nData Context:\n{data_context}"
                }
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return message.choices[0].message.content
    except Exception as e:
        return f"⚠️  AI analysis error: {str(e)}"


def handle_predictive_analysis(text):
    """Use AI to predict trends and provide analysis."""
    sku = extract_sku(text)
    category = extract_category(text)
    year = extract_year(text)
    
    df = df_fmcg.copy()
    
    if sku:
        df = df[df["sku"] == sku]
        focus = f"SKU {sku}"
    elif category:
        df = df[df["category"] == category]
        focus = f"category {category}"
    else:
        focus = "overall market"
    
    if year:
        df = df[df["date"].dt.year == year]
    
    # Prepare data context for AI
    summary_stats = {
        "total_units": df["units_sold"].sum(),
        "avg_price": df["price_unit"].mean(),
        "avg_stock": df["stock_available"].mean(),
        "promo_weeks_pct": (df["promotion_flag"].mean() * 100),
        "channels": df["channel"].unique().tolist(),
        "regions": df["region"].unique().tolist(),
    }
    
    data_context = f"""
    Data for {focus}:
    - Total units sold: {fmt_num(summary_stats['total_units'])}
    - Average price: {fmt_num(summary_stats['avg_price'])}
    - Average stock: {fmt_num(summary_stats['avg_stock'])}
    - Weeks with promotion: {summary_stats['promo_weeks_pct']:.1f}%
    - Active channels: {', '.join(summary_stats['channels'])}
    - Active regions: {', '.join(summary_stats['regions'])}
    """
    
    ai_response = get_ai_analysis(text, data_context)
    if ai_response:
        return f"🤖 AI Analysis for {focus}:\n\n{ai_response}"
    else:
        return f"Data summary for {focus}:\n{data_context}"


def handle_help(_):
    return dedent("""
        What you can ask me:

        📦 SKU info & listing
          "Tell me about MI-006"
          "List all SKUs"
          "List SKUs in Milk"

        📊 Sales totals & trends
          "Total sales for JU-021"
          "Total sales in 2023"
          "Yearly sales trend for Juice"
          "Top 5 SKUs by sales"

        🏪 Breakdowns
          "Sales by channel"
          "Sales by region for MI-006"
          "Sales by category in 2022"

        📣 Promotions
          "Promotion impact for MI-006"
          "How does promotion affect sales?"

        📦 Stock
          "Average stock for MI-006"
          "Stock in Retail channel"

        🔎 MI-006 deep dive
          "MI-006 enriched data in 2023"

        🤖 AI Analysis (with Groq)
          "Analyze MI-006 performance"
          "Predict trends for Juice category"
          "What insights can you give about 2023?"

        🔤 Meta
          "list categories"
          "list brands"
          "help"
    """).strip()


# ──────────────────────────────────────────────
# 4. INTENT ROUTER
# ──────────────────────────────────────────────

INTENT_PATTERNS = [
    # (regex pattern, handler_function)
    (r'\b(analyze|predict|insight|forecast|trend|pattern|recommendation)\b', handle_predictive_analysis),
    (r'\b(what|give).*\b(insight|analysis|recommendation|forecast)\b', handle_predictive_analysis),
    
    (r'\b(list|show|what).*(sku|product)', handle_list_skus),
    (r'\b(list|show|what).*(categor)', handle_list_categories),
    (r'\b(list|show|what).*(brand)', handle_list_brands),
    (r'\b(help|commands|what can|options)\b', handle_help),

    (r'\b(top|best|highest).*(sku|product|seller)', handle_top_skus),
    (r'\b(promo|promotion|discount).*(impact|effect|lift|compare)', handle_promotion_impact),
    (r'\b(promo|promotion).*(sales|affect)', handle_promotion_impact),
    (r'\b(impact|effect|lift)\b.*(promo)', handle_promotion_impact),

    (r'\b(channel).*(breakdown|split|sales)', handle_channel_breakdown),
    (r'\bsales\b.*(channel)', handle_channel_breakdown),
    (r'\b(region).*(breakdown|split|sales)', handle_region_breakdown),
    (r'\bsales\b.*(region)', handle_region_breakdown),
    (r'\b(category|categories).*(breakdown|split|sales)', handle_category_breakdown),
    (r'\bsales\b.*(category|categories)', handle_category_breakdown),

    (r'\b(trend|year.over.year|yearly|annual)', handle_yearly_trend),

    (r'\b(stock|inventory|availability)\b', handle_stock),

    (r'\bmi.?006\b.*(enriched|detail|temp|inflation|trend)', handle_mi006_detail),
    (r'\b(enriched|detail|temp|inflation)\b.*mi.?006', handle_mi006_detail),

    (r'\b(info|about|detail|summary)\b', handle_sku_info),
    (r'\b(total|sum|how many|how much|sold|sales)\b', handle_total_sales),

    (r'\bmi.?006\b', handle_mi006_detail),  # fallback for any MI-006 query
]

def route(text):
    text_l = text.lower().strip()

    if not text_l:
        return "Please type a question. Type 'help' for options."

    for pattern, handler in INTENT_PATTERNS:
        if re.search(pattern, text_l):
            try:
                return handler(text)
            except Exception as e:
                return f"⚠️  Error processing that query: {e}"

    # Fallback: try total sales with any filter found
    sku = extract_sku(text)
    if sku:
        return handle_sku_info(text)

    return ("I didn't understand that. Try asking about sales, promotions, "
            "stock, regions, channels, or SKUs.\nType 'help' to see all options.")


# ──────────────────────────────────────────────
# 5. CHAT LOOP
# ──────────────────────────────────────────────

WELCOME = dedent("""
╔══════════════════════════════════════════════════════╗
║     FMCG Data Assistant with AI (2022–2024)          ║
║  Ask questions about your sales, promotions,         ║
║  SKUs, channels, regions, and get AI-powered         ║
║  insights, predictions, and trends analysis.         ║
║  Type 'help' for commands. 'quit' to exit.           ║
╚══════════════════════════════════════════════════════╝
""")

def main():
    print(WELCOME)
    while True:
        try:
            user_input = input("\n📊 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = route(user_input)
        print(f"\n🤖 Bot: {response}")

if __name__ == "__main__":
    main()
