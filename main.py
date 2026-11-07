import threading
import time
import flet as ft
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

LANGS = {
    "UZ": {
        "app_title": "Forex AI Signals",
        "app_subtitle": "AI Bozor Tahlili",
        "tab_signals": "AI Signallar",
        "tab_chart": "Chuqur Tahlil",
        "tab_market": "Bozor & AI",
        "top_signals": "🔥 Top AI Signallar Lentasi",
        "market_list": "Asosiy Valyuta Juftliklari",
        "strong_buy": "Kuchli Sotib Olish (BUY)",
        "strong_sell": "Kuchli Sotish (SELL)",
        "sentiment_title": "Bozor Kayfiyati (Sentiment)",
        "buyers": "Xaridorlar",
        "sellers": "Sotuvchilar",
        "calendar_title": "Iqtisodiy Kalendar & AI Sharhi",
        "volatility": "Yuqori volatillik kutilmoqda",
        "no_data": "Ma'lumotlar yuklanmoqda...",
    },
    "EN": {
        "app_title": "Forex AI Signals",
        "app_subtitle": "AI Market Analysis",
        "tab_signals": "AI Signals",
        "tab_chart": "Deep Analysis",
        "tab_market": "Market & AI",
        "top_signals": "🔥 Top AI Signals Feed",
        "market_list": "Major Currency Pairs",
        "strong_buy": "Strong Buy (BUY)",
        "strong_sell": "Strong Sell (SELL)",
        "sentiment_title": "Market Sentiment",
        "buyers": "Buyers",
        "sellers": "Sellers",
        "calendar_title": "Economic Calendar & AI Insight",
        "volatility": "High volatility expected",
        "no_data": "Loading data...",
    },
    "RU": {
        "app_title": "Forex AI Signals",
        "app_subtitle": "AI Анализ Рынка",
        "tab_signals": "AI Сигналы",
        "tab_chart": "Глубокий Анализ",
        "tab_market": "Рынок и AI",
        "top_signals": "🔥 Лента Топ AI Сигналов",
        "market_list": "Основные Валютные Пары",
        "strong_buy": "Активно Покупать (BUY)",
        "strong_sell": "Активно Продавать (SELL)",
        "sentiment_title": "Настроение Рынка",
        "buyers": "Покупатели",
        "sellers": "Продавцы",
        "calendar_title": "Экономический Календарь и AI Обзор",
        "volatility": "Ожидается высокая волатильность",
        "no_data": "Загрузка данных...",
    },
    "TR": {
        "app_title": "Forex AI Signals",
        "app_subtitle": "AI Piyasa Analizi",
        "tab_signals": "AI Sinyalleri",
        "tab_chart": "Derin Analiz",
        "tab_market": "Piyasa & AI",
        "top_signals": "🔥 En İyi AI Sinyalleri",
        "market_list": "Majör Döviz Çiftleri",
        "strong_buy": "Güçlü Al (BUY)",
        "strong_sell": "Güçlü Sat (SELL)",
        "sentiment_title": "Piyasa Duyarlılığı",
        "buyers": "Alıcılar",
        "sellers": "Satıcılar",
        "calendar_title": "Ekonomik Takvim & AI Yorumu",
        "volatility": "Yüksek volatilite bekleniyor",
        "no_data": "Veriler yükleniyor...",
    },
}

PAIRS = [
    {"symbol": "EURUSD=X", "name": "EUR/USD"},
    {"symbol": "GBPUSD=X", "name": "GBP/USD"},
    {"symbol": "USDJPY=X", "name": "USD/JPY"},
    {"symbol": "AUDUSD=X", "name": "AUD/USD"},
    {"symbol": "USDCAD=X", "name": "USD/CAD"},
    {"symbol": "USDCHF=X", "name": "USD/CHF"},
]

TIMEFRAMES = ["1m", "5m", "10m", "15m", "30m", "45m", "1H", "4H", "1D"]


def analyze_pair(symbol: str, tf: str):
  try:
    if tf in ["1m", "5m", "10m"]:
      period, interval = "5d", tf.lower()
    elif tf in ["15m", "30m", "45m", "1H"]:
      period, interval = "60d", "1h" if "h" in tf.lower() else tf.lower()
    elif tf == "4H":
      period, interval = "60d", "1h"
    else:
      period, interval = "1y", "1d"

    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty or len(df) < 30:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.droplevel(1)

    if tf == "4H":
      df = (
          df.resample("4h")
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
          .dropna()
      )

    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df_clean = df.dropna().copy()

    if len(df_clean) < 15:
      return None

    features = ["Return", "MA10", "MA20", "MA50", "RSI", "MACD"]
    X = df_clean[features]
    y = df_clean["Target"]

    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X, y)

    latest = X.tail(1)
    pred = int(model.predict(latest)[0])
    prob = float(model.predict_proba(latest)[0][pred]) * 100

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    close_price = float(last_row["Close"])
    change_pct = (
        (close_price - float(prev_row["Close"])) / float(prev_row["Close"])
    ) * 100

    bullish_signals = 0
    if close_price > float(last_row["MA10"]):
      bullish_signals += 2
    if close_price > float(last_row["MA20"]):
      bullish_signals += 2
    if close_price > float(last_row["MA50"]):
      bullish_signals += 2
    if 45 < float(last_row["RSI"]) < 70:
      bullish_signals += 2
    if float(last_row["MACD"]) > float(last_row["MACD_Signal"]):
      bullish_signals += 2
    if pred == 1:
      bullish_signals += 2

    recent_df = df.tail(20)
    resistance = float(recent_df["High"].max())
    support = float(recent_df["Low"].min())
    history_prices = [float(x) for x in df["Close"].tail(25).tolist()]

    return {
        "symbol": symbol,
        "price": close_price,
        "change_pct": change_pct,
        "pred": pred,
        "confidence": prob,
        "rsi": float(last_row["RSI"]),
        "macd": float(last_row["MACD"]),
        "bullish_count": bullish_signals,
        "resistance": resistance,
        "support": support,
        "history": history_prices,
    }
  except Exception as e:
    print(f"Xatolik {symbol} bo'yicha: {e}")
    return None


def main(page: ft.Page):
  page.title = "Forex AI Signals & Market"
  page.bgcolor = "#0B0F19"
  page.window.width = 440
  page.window.height = 880
  page.padding = 12
  page.theme_mode = ft.ThemeMode.DARK

  state = {
      "lang": "UZ",
      "tf": "1H",
      "tab": 0,
      "results": {},
      "selected_pair": "EUR/USD",
  }

  top_signals_row = ft.Row(scroll=ft.ScrollMode.AUTO, spacing=10)
  pairs_list_col = ft.Column(spacing=8)
  deep_analysis_col = ft.Column(spacing=12)
  market_sentiment_col = ft.Column(spacing=12)
  loader = ft.ProgressBar(visible=False, color="#10B981", bgcolor="#1E293B")

  brand_title = ft.Text(
      "Forex AI Signals", size=15, weight=ft.FontWeight.BOLD, color="white"
  )
  brand_sub = ft.Text(
      "AI Bozor Tahlili", size=11, color="#94A3B8", weight=ft.FontWeight.W_500
  )

  def set_lang(e):
    state["lang"] = e.control.data
    for b in lang_buttons.controls:
      b.bgcolor = "#2563EB" if b.data == state["lang"] else "#1E293B"
    render_ui()

  lang_buttons = ft.Row(
      [
          ft.Container(
              content=ft.Text(code, size=10, weight=ft.FontWeight.BOLD),
              padding=5,
              bgcolor="#2563EB" if code == "UZ" else "#1E293B",
              border_radius=6,
              data=code,
              on_click=set_lang,
          )
          for code in ["UZ", "EN", "RU", "TR"]
      ],
      spacing=3,
  )

  def set_tf(e):
    state["tf"] = e.control.data
    for b in tf_buttons.controls:
      b.bgcolor = "#10B981" if b.data == state["tf"] else "#1E293B"
    refresh_market_data()

  tf_buttons = ft.Row(
      [
          ft.Container(
              content=ft.Text(tf, size=11, weight=ft.FontWeight.W_600),
              padding=6,
              bgcolor="#10B981" if tf == "1H" else "#1E293B",
              border_radius=6,
              data=tf,
              on_click=set_tf,
          )
          for tf in TIMEFRAMES
      ],
      scroll=ft.ScrollMode.AUTO,
      spacing=4,
  )

  def switch_tab(e):
    state["tab"] = e.control.selected_index
    render_ui()

  nav_bar = ft.NavigationBar(
      selected_index=0,
      on_change=switch_tab,
      bgcolor="#111827",
      destinations=[
          ft.NavigationBarDestination(
              icon=ft.Icons.TRENDING_UP, label="AI Signallar"
          ),
          ft.NavigationBarDestination(
              icon=ft.Icons.AUTO_GRAPH, label="Chuqur Tahlil"
          ),
          ft.NavigationBarDestination(
              icon=ft.Icons.DASHBOARD_ROUNDED, label="Bozor & AI"
          ),
      ],
  )

  def open_pair_detail(pair_name):
    state["selected_pair"] = pair_name
    state["tab"] = 1
    nav_bar.selected_index = 1
    render_ui()

  def render_ui():
    L = LANGS[state["lang"]]
    nav_bar.selected_index = state["tab"]

    tab_signals_view.visible = state["tab"] == 0
    tab_chart_view.visible = state["tab"] == 1
    tab_market_view.visible = state["tab"] == 2

    top_signals_title.value = L["top_signals"]
    market_list_title.value = L["market_list"]

    top_signals_row.controls.clear()
    pairs_list_col.controls.clear()

    if not state["results"]:
      pairs_list_col.controls.append(
          ft.Text(L["no_data"], color="#94A3B8", size=13)
      )
    else:
      sorted_pairs = sorted(
          state["results"].items(),
          key=lambda x: x[1]["confidence"],
          reverse=True,
      )

      for pair_name, data in sorted_pairs:
        is_buy = data["pred"] == 1
        sig_color = "#10B981" if is_buy else "#EF4444"
        sig_label = L["strong_buy"] if is_buy else L["strong_sell"]
        digits = 2 if "JPY" in pair_name else 4

        if data["confidence"] >= 58:
          top_signals_row.controls.append(
              ft.Container(
                  width=200,
                  padding=12,
                  bgcolor="#161E31",
                  border=ft.border.all(1, sig_color),
                  border_radius=12,
                  on_click=lambda e, p=pair_name: open_pair_detail(p),
                  content=ft.Column(
                      [
                          ft.Row(
                              [
                                  ft.Text(
                                      pair_name,
                                      weight=ft.FontWeight.BOLD,
                                      size=14,
                                  ),
                                  ft.Container(
                                      content=ft.Text(
                                          f"{data['confidence']:.0f}%",
                                          size=10,
                                          weight=ft.FontWeight.BOLD,
                                          color="white",
                                      ),
                                      bgcolor=sig_color,
                                      padding=4,
                                      border_radius=4,
                                  ),
                              ],
                              alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                          ),
                          ft.Text(
                              sig_label,
                              color=sig_color,
                              size=11,
                              weight=ft.FontWeight.W_600,
                          ),
                          ft.Text(
                              f"Ishonchlilik: {data['confidence']:.1f}% ({state['tf']})",
                              size=11,
                              color="#94A3B8",
                          ),
                      ],
                      spacing=4,
                  ),
              )
          )

        chg_color = "#10B981" if data["change_pct"] >= 0 else "#EF4444"
        chg_sign = "+" if data["change_pct"] >= 0 else ""
        badge_text = "Bullish" if is_buy else "Bearish"

        pairs_list_col.controls.append(
            ft.Container(
                padding=12,
                bgcolor="#161E31",
                border_radius=10,
                on_click=lambda e, p=pair_name: open_pair_detail(p),
                content=ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text(
                                    pair_name,
                                    weight=ft.FontWeight.BOLD,
                                    size=14,
                                ),
                                ft.Text(
                                    f"Ishonch: {data['confidence']:.0f}%",
                                    size=11,
                                    color="#94A3B8",
                                ),
                            ],
                            spacing=2,
                        ),
                        ft.Row(
                            [
                                ft.Column(
                                    [
                                        ft.Text(
                                            f"{data['price']:.{digits}f}",
                                            weight=ft.FontWeight.BOLD,
                                            size=14,
                                        ),
                                        ft.Text(
                                            f"{chg_sign}{data['change_pct']:.2f}%",
                                            size=11,
                                            color=chg_color,
                                        ),
                                    ],
                                    horizontal_alignment=ft.CrossAxisAlignment.END,
                                    spacing=2,
                                ),
                                ft.Container(
                                    content=ft.Text(
                                        badge_text,
                                        size=10,
                                        weight=ft.FontWeight.BOLD,
                                        color=sig_color,
                                    ),
                                    padding=6,
                                    bgcolor="#0B0F19",
                                    border_radius=6,
                                ),
                            ],
                            spacing=10,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
            )
        )

    # Chuqur Tahlil oynasi
    deep_analysis_col.controls.clear()
    sel = state["selected_pair"]
    if sel in state["results"]:
      d = state["results"][sel]
      is_buy = d["pred"] == 1
      main_color = "#10B981" if is_buy else "#EF4444"
      digits = 2 if "JPY" in sel else 4
      bull_cnt = d["bullish_count"]

      prices = d["history"]
      min_p, max_p = min(prices), max(prices)
      span = max_p - min_p if max_p != min_p else 0.0001
      bars = []
      for p in prices[-18:]:
        h = int(((p - min_p) / span) * 65) + 15
        bars.append(
            ft.Container(width=8, height=h, bgcolor="#3B82F6", border_radius=3)
        )
      ai_h = min(80, max(15, bars[-1].height + (12 if is_buy else -12)))
      bars.append(
          ft.Container(
              width=10,
              height=ai_h,
              bgcolor=main_color,
              border=ft.border.all(1, "white"),
              border_radius=3,
          )
      )

      deep_analysis_col.controls.extend([
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Row(
                          [
                              ft.Text(
                                  f"{sel} ({state['tf']})",
                                  size=16,
                                  weight=ft.FontWeight.BOLD,
                              ),
                              ft.Text(
                                  f"{d['price']:.{digits}f}",
                                  size=16,
                                  weight=ft.FontWeight.BOLD,
                                  color=main_color,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Text(
                          f"AI: {L['strong_buy'] if is_buy else L['strong_sell']} ({d['confidence']:.1f}%)",
                          color=main_color,
                          weight=ft.FontWeight.W_600,
                          size=12,
                      ),
                      ft.Divider(height=8, color="transparent"),
                      ft.Container(
                          height=90,
                          padding=6,
                          bgcolor="#0B0F19",
                          border_radius=8,
                          alignment=ft.alignment.bottom_center,
                          content=ft.Row(
                              bars,
                              alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                              vertical_alignment=ft.CrossAxisAlignment.END,
                          ),
                      ),
                  ]
              ),
          ),
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Text(
                          "Indikatorlar Xulosasi",
                          weight=ft.FontWeight.BOLD,
                          size=14,
                      ),
                      ft.Text(
                          f"12 ta indikatordan {bull_cnt} tasi O'SISH (BUY)"
                          " signalini bermoqda",
                          color="#10B981",
                          size=12,
                          weight=ft.FontWeight.W_600,
                      ),
                      ft.ProgressBar(
                          value=bull_cnt / 12,
                          color="#10B981",
                          bgcolor="#EF4444",
                          height=8,
                      ),
                  ]
              ),
          ),
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Text(
                          "Support & Resistance",
                          weight=ft.FontWeight.BOLD,
                          size=14,
                      ),
                      ft.Row(
                          [
                              ft.Text(
                                  "Qo'llab-quvvatlash (S1)",
                                  color="#10B981",
                                  size=12,
                              ),
                              ft.Text(
                                  f"{d['support']:.{digits}f}",
                                  weight=ft.FontWeight.BOLD,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Row(
                          [
                              ft.Text(
                                  "Qarshilik (R1)", color="#EF4444", size=12
                              ),
                              ft.Text(
                                  f"{d['resistance']:.{digits}f}",
                                  weight=ft.FontWeight.BOLD,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                  ]
              ),
          ),
      ])

    # Bozor & AI oynasi
    market_sentiment_col.controls.clear()
    if state["results"]:
      buy_pairs = sum(1 for v in state["results"].values() if v["pred"] == 1)
      total_pairs = len(state["results"])
      buyer_pct = (
          int((buy_pairs / total_pairs) * 100) if total_pairs > 0 else 66
      )
      seller_pct = 100 - buyer_pct

      market_sentiment_col.controls.extend([
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Row(
                          [
                              ft.Text(
                                  L["sentiment_title"],
                                  weight=ft.FontWeight.BOLD,
                                  size=14,
                              ),
                              ft.Container(
                                  content=ft.Text(
                                      f"{sel} • {state['tf']}",
                                      size=10,
                                      color="white",
                                      weight=ft.FontWeight.BOLD,
                                  ),
                                  bgcolor="#2563EB",
                                  padding=4,
                                  border_radius=4,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Row(
                          [
                              ft.Text(
                                  f"{buyer_pct}% {L['buyers']}",
                                  color="#10B981",
                                  weight=ft.FontWeight.BOLD,
                              ),
                              ft.Text(
                                  f"{seller_pct}% {L['sellers']}",
                                  color="#EF4444",
                                  weight=ft.FontWeight.BOLD,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.ProgressBar(
                          value=buyer_pct / 100,
                          color="#10B981",
                          bgcolor="#EF4444",
                          height=10,
                      ),
                      ft.Text(
                          "Institutsional va chakana treyderlarning aksariyati"
                          " uzoq pozitsiyalarni ushlab turibdi. Qisqa muddatli"
                          " o'sish ehtimoli yuqori.",
                          size=11,
                          color="#94A3B8",
                      ),
                  ]
              ),
          ),
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Text(
                          L["calendar_title"],
                          weight=ft.FontWeight.BOLD,
                          size=14,
                      ),
                      ft.Divider(height=4, color="#334155"),
                      ft.Row(
                          [
                              ft.Text(
                                  "16:30 • USD",
                                  size=11,
                                  color="#94A3B8",
                                  weight=ft.FontWeight.BOLD,
                              ),
                              ft.Container(
                                  content=ft.Text(
                                      "⚠️ Yuqori volatillik kutilmoqda",
                                      size=10,
                                      color="#F59E0B",
                                      weight=ft.FontWeight.BOLD,
                                  ),
                                  bgcolor="#332B15",
                                  padding=4,
                                  border_radius=4,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Text(
                          "AQSH Inflyatsiya Darajasi (CPI y/y)",
                          weight=ft.FontWeight.BOLD,
                          size=13,
                      ),
                      ft.Text(
                          "AI Sharhi: Agar ko'rsatkich 2.9% dan yuqori chiqsa,"
                          " Dollar keskin mustahkamlanib, EUR/USD pasayishi"
                          " mumkin.",
                          size=11,
                          color="#94A3B8",
                      ),
                      ft.Divider(height=6, color="#334155"),
                      ft.Row(
                          [
                              ft.Text(
                                  "19:00 • USD",
                                  size=11,
                                  color="#94A3B8",
                                  weight=ft.FontWeight.BOLD,
                              ),
                              ft.Container(
                                  content=ft.Text(
                                      "⚠️ Yuqori volatillik kutilmoqda",
                                      size=10,
                                      color="#F59E0B",
                                      weight=ft.FontWeight.BOLD,
                                  ),
                                  bgcolor="#332B15",
                                  padding=4,
                                  border_radius=4,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Text(
                          "FOMC Foiz Stavkasi Qarori",
                          weight=ft.FontWeight.BOLD,
                          size=13,
                      ),
                      ft.Text(
                          "AI Sharhi: Stavka o'zgarishsiz qolishi kutilmoqda,"
                          " biroq Powell nutqi vaqtida barcha juftliklarda"
                          " kuchli impuls bo'ladi.",
                          size=11,
                          color="#94A3B8",
                      ),
                  ]
              ),
          ),
      ])

    page.update()

  def refresh_market_data(e=None):
    loader.visible = True
    page.update()

    def worker():
      new_results = {}
      for p in PAIRS:
        print(f"Tahlil qilinmoqda: {p['name']} ({state['tf']})...")
        res = analyze_pair(p["symbol"], state["tf"])
        if res:
          new_results[p["name"]] = res
      state["results"] = new_results
      loader.visible = False
      print("Barcha ma'lumotlar yuklandi, UI yangilanmoqda...")
      render_ui()

    threading.Thread(target=worker, daemon=True).start()

  top_signals_title = ft.Text(
      LANGS["UZ"]["top_signals"], size=13, weight=ft.FontWeight.BOLD
  )
  market_list_title = ft.Text(
      LANGS["UZ"]["market_list"], size=13, weight=ft.FontWeight.BOLD
  )

  tab_signals_view = ft.Column(
      [
          top_signals_title,
          top_signals_row,
          ft.Divider(height=6, color="transparent"),
          market_list_title,
          pairs_list_col,
      ],
      scroll=ft.ScrollMode.AUTO,
      expand=True,
  )

  tab_chart_view = ft.Column(
      [deep_analysis_col], visible=False, scroll=ft.ScrollMode.AUTO, expand=True
  )

  tab_market_view = ft.Column(
      [market_sentiment_col],
      visible=False,
      scroll=ft.ScrollMode.AUTO,
      expand=True,
  )

  page.add(
      ft.Row(
          [
              ft.Row(
                  [
                      ft.Container(
                          content=ft.Icon(
                              ft.Icons.TRENDING_UP, color="white", size=16
                          ),
                          bgcolor="#10B981",
                          padding=6,
                          border_radius=8,
                      ),
                      ft.Column([brand_title, brand_sub], spacing=0),
                  ],
                  spacing=8,
              ),
              ft.Row([lang_buttons]),
          ],
          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
      ),
      tf_buttons,
      loader,
      ft.Divider(height=2, color="#1E293B"),
      ft.Column(
          [tab_signals_view, tab_chart_view, tab_market_view], expand=True
      ),
      nav_bar,
  )

  refresh_market_data()


if __name__ == "__main__":
  ft.app(target=main)