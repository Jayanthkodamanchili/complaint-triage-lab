# NEW CODE START
"""The app's look: sunset colours, fonts, header and footer (paint only, no logic)."""

import gradio as gr

# Part 1: the colour palette (one place to change any colour)
NAVY = "#161B2F"  # page background (deepest navy)
TEXT = "#e7edf4"  # main text
SOFT_TEXT = "#96a6b8"  # quieter text
GOLD = "#FFA586"  # peach: the accent colour (name kept, app.py uses it)
LIGHT_GOLD = "#FFC0A8"  # button when the mouse is on it
CARD = "rgba(36, 47, 73, 0.75)"  # see-through dark blue boxes
INPUT = "rgba(22, 27, 47, 0.9)"  # text boxes
EDGE = "#384358"  # slate thin borders
CRIMSON = "#B51A2B"  # deep red
HOT_RED = "#E63946"  # brighter red for gradients
SUNSET = f"linear-gradient(90deg, {CRIMSON}, {HOT_RED}, {GOLD})"  # red to peach

# Part 2: the theme: Gradio's plain Base theme, repainted with our colours
COLOURS = {
    "body_background_fill": NAVY,
    "body_text_color": TEXT,
    "body_text_color_subdued": SOFT_TEXT,
    "background_fill_primary": NAVY,
    "background_fill_secondary": CARD,
    "block_background_fill": CARD,
    "block_border_color": EDGE,
    "block_label_background_fill": CARD,
    "block_label_text_color": GOLD,
    "block_title_text_color": GOLD,
    "border_color_primary": EDGE,
    "input_background_fill": INPUT,
    "input_border_color": EDGE,
    "input_border_color_focus": GOLD,
    "input_placeholder_color": "#8a9db3",
    "color_accent_soft": "rgba(255, 165, 134, 0.15)",
    "button_primary_background_fill": SUNSET,
    "button_primary_background_fill_hover": f"linear-gradient(90deg, {HOT_RED}, {CRIMSON}, {HOT_RED})",
    "button_primary_text_color": "#ffffff",
    "button_secondary_background_fill": "transparent",
    "button_secondary_border_color": EDGE,
    "button_secondary_text_color": TEXT,
    "table_even_background_fill": CARD,
    "table_odd_background_fill": INPUT,
}
# the same colours in light and dark mode, so every visitor sees the sunset look
SETTINGS = {**COLOURS, **{name + "_dark": value for name, value in COLOURS.items()}}

THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Space Mono"), "monospace"],
    radius_size="sm",
).set(color_accent=GOLD, **SETTINGS)

# Part 3: a little extra CSS: page width, sunset sky, shiny title, glowing cards and button
CSS = f"""
.gradio-container {{ max-width: 1120px !important; margin: 0 auto !important; }}
body, gradio-app {{
  background:
    radial-gradient(ellipse 80% 40% at 50% -5%, rgba(230, 57, 70, 0.65), transparent 70%),
    radial-gradient(ellipse 50% 22% at 50% 14%, rgba(255, 165, 134, 0.30), transparent 70%),
    radial-gradient(ellipse 40% 30% at 100% 100%, rgba(181, 26, 43, 0.25), transparent 70%),
    linear-gradient(180deg, #541A2E 0px, {NAVY} 480px) !important;
  background-attachment: fixed !important;
}}
a {{ color: {GOLD} !important; }}
.hero {{ text-align: center; padding: 8px 0 4px; }}
.hero h1 {{
  margin: 0; font-size: 2.2rem;
  background: linear-gradient(90deg, #FFC0A8, #ffffff, #FFA586);
  text-shadow: 0 0 30px rgba(255, 165, 134, 0.35);
  -webkit-background-clip: text; background-clip: text; color: transparent !important;
}}
.hero p {{ margin: 6px 0 16px; color: {SOFT_TEXT}; font-size: 1.05rem; }}
.stats {{ display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }}
.stat {{
  padding: 12px 22px; border-radius: 14px; backdrop-filter: blur(12px);
  border: 1px solid transparent;
  background: linear-gradient({NAVY}, {NAVY}) padding-box, {SUNSET} border-box;
  box-shadow: 0 0 24px rgba(230, 57, 70, 0.25);
  transition: transform 0.2s, box-shadow 0.2s;
}}
.stat:hover {{ transform: translateY(-3px); box-shadow: 0 0 32px rgba(255, 165, 134, 0.45); }}
.stat b {{
  display: block; font-size: 1.7rem; background: {SUNSET};
  -webkit-background-clip: text; background-clip: text; color: transparent;
}}
button.primary {{
  box-shadow: 0 4px 20px rgba(230, 57, 70, 0.35); transition: all 0.2s;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.45); font-weight: 700;
}}
button.primary:hover {{ box-shadow: 0 6px 28px rgba(255, 165, 134, 0.5); }}
.tab-container button.selected, button[role="tab"][aria-selected="true"] {{ color: {GOLD} !important; }}
.gallery-item:hover, button.gallery-item:hover {{ border-color: {GOLD} !important; color: {GOLD} !important; }}
.stat span {{ font-size: 0.85rem; color: {SOFT_TEXT}; }}
"""

# Part 4: smooth motion (CSS animations; switched off for "reduce motion" visitors)
ANIMATIONS = f"""
@keyframes rise {{ from {{ opacity: 0; transform: translateY(18px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes breathe {{ 0%, 100% {{ opacity: 0.55; }} 50% {{ opacity: 1; }} }}
@keyframes shimmer {{ to {{ background-position: 200% center; }} }}

/* 1. entrance: title, line, then the 3 cards one after another */
.hero h1 {{ animation: rise 0.8s ease-out both, shimmer 6s linear infinite; background-size: 200% auto; }}
.hero p {{ animation: rise 0.8s ease-out 0.15s both; }}
.stat {{ animation: rise 0.7s ease-out both; }}
.stat:nth-child(1) {{ animation-delay: 0.3s; }}
.stat:nth-child(2) {{ animation-delay: 0.45s; }}
.stat:nth-child(3) {{ animation-delay: 0.6s; }}

/* 2. breathing sunset: a glow layer behind the page that slowly pulses */
body::before {{
  content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background: radial-gradient(ellipse 70% 35% at 50% 0%, rgba(255, 90, 100, 0.35), transparent 70%);
  animation: breathe 7s ease-in-out infinite;
}}

/* 3. smooth tab switches and answers that fade in */
.tabitem {{ animation: rise 0.45s ease-out both; }}
.prose, .json-holder, .message {{ animation: fade 0.5s ease-out both; }}

/* 4. small touches: boxes glow when hovered, buttons lift, smooth scrolling */
.block {{ transition: border-color 0.3s, box-shadow 0.3s; }}
.block:hover {{ border-color: rgba(255, 165, 134, 0.45) !important; box-shadow: 0 0 18px rgba(255, 165, 134, 0.08); }}
button {{ transition: all 0.2s ease !important; }}
button.primary:hover, .gallery-item:hover {{ transform: translateY(-2px); }}
button.primary:active {{ transform: scale(0.98); }}
html {{ scroll-behavior: smooth; }}
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-thumb {{ background: {SUNSET}; border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: {NAVY}; }}

/* respect visitors who ask their computer for less motion */
@media (prefers-reduced-motion: reduce) {{
  *, *::before {{ animation: none !important; transition: none !important; }}
}}
"""
CSS += ANIMATIONS

# Part 5: the header: title, one-line pitch and 3 headline numbers (from Phase 4)
HEADER = """
<div class="hero">
  <h1>Complaint Triage Lab</h1>
  <p>Which AI should a bank use to sort customer complaints?
     6 approaches tested on 500 real US complaints.</p>
  <div class="stats">
    <div class="stat"><b>95%</b><span>accuracy (smart router)</span></div>
    <div class="stat"><b>$43</b><span>per million complaints</span></div>
    <div class="stat"><b>$0</b><span>to try this demo</span></div>
  </div>
</div>
"""

FOOTER = (
    "<center>Built by Jay · "
    '<a href="https://github.com/Jayanthkodamanchili/complaint-triage-lab">Code on GitHub</a>'
    " · Data: CFPB consumer complaints</center>"
)
# NEW CODE END
