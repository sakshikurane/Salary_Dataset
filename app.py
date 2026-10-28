import os
import pickle
import warnings
from datetime import date
 
import pandas as pd
from flask import Flask, render_template_string, request
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
 
# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liner.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)   # exact column order the model expects
COEF = dict(zip(FEATURES, model.coef_))     # linear model: price change per unit of each feature
 
# --------------------------------------------------------------------------
# Form definition
# --------------------------------------------------------------------------
SECTIONS = [
    {"title": "📐 Size & Layout", "fields": [
        {"name": "Square_Footage", "label": "Square Footage (sq ft)", "kind": "range",
         "min": 500, "max": 4000, "step": 10, "default": 1800},
        {"name": "Num_Bedrooms", "label": "Bedrooms", "kind": "select",
         "options": [(i, f"{i} bedroom{'s' if i > 1 else ''}") for i in range(1, 7)], "default": 3},
        {"name": "Num_Bathrooms", "label": "Bathrooms", "kind": "select",
         "options": [(i, f"{i} bathroom{'s' if i > 1 else ''}") for i in range(1, 6)], "default": 2},
    ]},
    {"title": "🏡 Property Details", "fields": [
        {"name": "Year_Built", "label": "Year Built", "kind": "range",
         "min": 1950, "max": 2025, "step": 1, "default": 1990},
        {"name": "Lot_Size", "label": "Lot Size (acres)", "kind": "range",
         "min": 0.5, "max": 5, "step": 0.1, "default": 2.5},
        {"name": "Garage_Size", "label": "Garage Size (cars)", "kind": "select",
         "options": [(0, "No garage"), (1, "1 car"), (2, "2 cars"), (3, "3 cars")], "default": 1},
    ]},
    {"title": "📍 Location", "fields": [
        {"name": "Neighborhood_Quality", "label": "Neighborhood Quality (1-10)", "kind": "range",
         "min": 1, "max": 10, "step": 1, "default": 5},
    ]},
]
 
# Field -> (nice label, slider/select minimum used as the reference point for the "drivers" chart)
DRIVER_LABELS = {
    "Square_Footage": ("Square footage", 500),
    "Num_Bedrooms": ("Bedrooms", 1),
    "Num_Bathrooms": ("Bathrooms", 1),
    "Year_Built": ("Year built", 1950),
    "Lot_Size": ("Lot size", 0.5),
    "Garage_Size": ("Garage", 0),
    "Neighborhood_Quality": ("Neighbourhood", 1),
}
DRIVER_COLORS = ["#ffc8dd", "#bde0fe", "#c9f2dd", "#ffe3a8", "#d9ccff", "#ffd6a5", "#b8f0ef"]
 
 
# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def money(x: float) -> str:
    return f"${x:,.0f}"
 
 
def build_drivers(v: dict) -> list:
    """Value each feature adds above the smallest option of its input range (linear model)."""
    rows = []
    for i, (name, (label, ref)) in enumerate(DRIVER_LABELS.items()):
        rows.append({"label": label, "amount": (v[name] - ref) * COEF[name], "color": DRIVER_COLORS[i]})
    top = max(abs(r["amount"]) for r in rows) or 1
    for r in rows:
        r["pct"] = round(abs(r["amount"]) / top * 100, 1)
        r["text"] = ("+" if r["amount"] >= 0 else "-") + money(abs(r["amount"]))
    return sorted(rows, key=lambda r: -abs(r["amount"]))
 
 
def build_snapshot(v: dict, price: float) -> list:
    age = max(date.today().year - int(v["Year_Built"]), 0)
    return [
        {"icon": "📏", "name": "Price per sq ft", "val": f"${price / v['Square_Footage']:,.0f}", "bg": "#ffd1dc"},
        {"icon": "🕰️", "name": "Home age", "val": f"{age} yrs", "bg": "#c6e4ff"},
        {"icon": "🌳", "name": "Lot size", "val": f"{v['Lot_Size']:g} acres", "bg": "#c9f2dd"},
        {"icon": "🚗", "name": "Garage", "val": f"{int(v['Garage_Size'])} car", "bg": "#ffe8b8"},
    ]
 
 
# --------------------------------------------------------------------------
# HTML template (pastel theme)
# --------------------------------------------------------------------------
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{
    --lav:#d9ccff; --pink:#ffd1dc; --mint:#c9f2dd; --sky:#c6e4ff; --peach:#ffe3c6; --butter:#fff1b8;
    --ink:#3b3355; --muted:#7a7396; --card:rgba(255,255,255,.74); --line:rgba(120,100,170,.18);
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Nunito',system-ui,sans-serif;color:var(--ink);min-height:100vh;padding:34px 16px 60px;
    background:
      radial-gradient(700px 420px at 100% 0%, var(--peach), transparent 65%),
      radial-gradient(700px 460px at 0% 10%, var(--sky), transparent 65%),
      radial-gradient(800px 520px at 50% 110%, var(--pink), transparent 65%),
      linear-gradient(160deg,#fffaf5,#f4f8ff);
    background-attachment:fixed;
  }
  .wrap{max-width:1080px;margin:0 auto}
  header{text-align:center;margin-bottom:30px}
  .badge{display:inline-block;padding:6px 16px;border-radius:999px;font-size:12px;font-weight:800;
    letter-spacing:.08em;text-transform:uppercase;color:#2f7d62;background:var(--mint);
    border:1px solid rgba(47,125,98,.2)}
  h1{font-family:'Quicksand',sans-serif;font-size:clamp(28px,5vw,46px);font-weight:700;margin:14px 0 8px}
  h1 span{background:linear-gradient(90deg,#f49ac1,#a98bf0 55%,#7ab8f5);-webkit-background-clip:text;
    background-clip:text;color:transparent}
  header p{color:var(--muted);max-width:560px;margin:0 auto;font-size:15px}
 
  /* result */
  .result{margin-bottom:28px;padding:30px;border-radius:26px;
    background:linear-gradient(135deg,#ffe8d6,#ffd9e6 50%,#e3d9ff);
    border:2px solid #f6c3d4;box-shadow:0 18px 50px rgba(160,120,190,.2);animation:pop .5s ease}
  @keyframes pop{from{opacity:0;transform:translateY(14px) scale(.98)}to{opacity:1;transform:none}}
  .price-box{text-align:center}
  .price-box .small{font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
  .price{font-family:'Quicksand',sans-serif;font-size:clamp(40px,8vw,66px);font-weight:700;margin:6px 0 2px;
    color:var(--ink)}
  .price-box p{font-size:13px;color:#5d5578}
  .chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:22px}
  .chip{padding:12px 14px;border-radius:16px;text-align:center;border:1px solid var(--line)}
  .chip .ci{font-size:22px}
  .chip .cn{font-size:12px;color:#5d5578;font-weight:600;margin-top:2px}
  .chip .cv{font-size:18px;font-weight:800}
  .sub{font-size:13px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
    margin:26px 0 12px}
  .drv{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px;font-weight:600}
  .drv .dn{width:110px;color:#5d5578}
  .drv .track{flex:1;height:14px;border-radius:99px;background:rgba(255,255,255,.65);overflow:hidden}
  .drv .fill{height:100%;border-radius:99px;border:1px solid rgba(59,51,85,.12)}
  .drv .dv{width:92px;text-align:right;font-weight:800}
  .note{font-size:12px;color:var(--muted);margin-top:8px}
 
  .error{background:#ffd9d9;border:1px solid #f3a6a6;padding:14px 18px;border-radius:14px;margin-bottom:20px}
 
  /* form */
  form{display:grid;gap:20px}
  .grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:22px;
    box-shadow:0 10px 30px rgba(120,100,170,.10);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}
  .card:nth-child(1){border-top:6px solid var(--pink)}
  .card:nth-child(2){border-top:6px solid var(--sky)}
  .card:nth-child(3){border-top:6px solid var(--mint)}
  .card h2{font-size:17px;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--line)}
  .field{margin-bottom:18px}.field:last-child{margin-bottom:0}
  label{display:flex;justify-content:space-between;font-size:13px;font-weight:700;color:var(--muted);margin-bottom:8px}
  label output{color:#7a5fc4;font-weight:800;background:var(--lav);padding:0 9px;border-radius:99px}
  select{width:100%;padding:11px 12px;border-radius:12px;font:inherit;color:var(--ink);font-weight:600;
    background:#fff;border:1px solid #d9d2ee;outline:none;transition:.2s}
  select:focus{border-color:#a98bf0;box-shadow:0 0 0 4px rgba(169,139,240,.28)}
  input[type=range]{width:100%;-webkit-appearance:none;height:8px;border-radius:99px;outline:none;
    background:linear-gradient(90deg,var(--pink),var(--sky))}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;
    background:#fff;border:4px solid #f49ac1;cursor:pointer;box-shadow:0 3px 8px rgba(120,100,170,.35)}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#fff;
    border:4px solid #f49ac1;cursor:pointer}
 
  .actions{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:6px}
  button,.reset{padding:15px 38px;border-radius:16px;border:0;font:inherit;font-weight:800;font-size:16px;
    cursor:pointer;transition:.2s;text-decoration:none;display:inline-block}
  button{color:var(--ink);background:linear-gradient(90deg,#ffc1d6,#d9ccff 55%,#bde0fe);
    box-shadow:0 10px 26px rgba(244,154,193,.4)}
  button:hover{transform:translateY(-2px);box-shadow:0 14px 32px rgba(169,139,240,.45)}
  .reset{color:var(--ink);background:#fff;border:1px solid #d9d2ee}
  .reset:hover{background:var(--butter)}
  footer{text-align:center;color:var(--muted);font-size:12px;margin:34px auto 0;max-width:620px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">Linear Regression · Real Estate AI</span>
    <h1>House <span>Price Predictor</span></h1>
    <p>Describe the property and get an instant estimate of its market price.</p>
  </header>
 
  {% if error %}<div class="error">⚠️ {{ error }}</div>{% endif %}
 
  {% if result %}
  <div class="result">
    <div class="price-box">
      <div class="small">Estimated market price</div>
      <div class="price">{{ result.price }}</div>
      <p>Estimate in US dollars, based on the property details entered below.</p>
    </div>
 
    <div class="chips">
      {% for c in result.snapshot %}
      <div class="chip" style="background:{{ c.bg }}">
        <div class="ci">{{ c.icon }}</div><div class="cn">{{ c.name }}</div><div class="cv">{{ c.val }}</div>
      </div>
      {% endfor %}
    </div>
 
    <div class="sub">What adds value</div>
    {% for d in result.drivers %}
    <div class="drv">
      <span class="dn">{{ d.label }}</span>
      <div class="track"><div class="fill" style="width:{{ d.pct }}%;background:{{ d.color }}"></div></div>
      <span class="dv">{{ d.text }}</span>
    </div>
    {% endfor %}
    <p class="note">Each bar shows how much that feature adds compared with the smallest option on its input.</p>
  </div>
  {% endif %}
 
  <form method="POST" action="/">
    <div class="grid3">
    {% for s in sections %}
      <div class="card">
        <h2>{{ s.title }}</h2>
        {% for f in s.fields %}
        <div class="field">
          {% set val = values.get(f.name, f.default) %}
          {% if f.kind == 'select' %}
            <label for="{{ f.name }}">{{ f.label }}</label>
            <select id="{{ f.name }}" name="{{ f.name }}">
              {% for code, text in f.options %}
              <option value="{{ code }}" {{ 'selected' if code|string == val|string }}>{{ text }}</option>
              {% endfor %}
            </select>
          {% else %}
            <label for="{{ f.name }}">{{ f.label }}<output id="o_{{ f.name }}">{{ val }}</output></label>
            <input type="range" id="{{ f.name }}" name="{{ f.name }}" min="{{ f.min }}" max="{{ f.max }}"
                   step="{{ f.step }}" value="{{ val }}"
                   oninput="document.getElementById('o_{{ f.name }}').textContent=this.value">
          {% endif %}
        </div>
        {% endfor %}
      </div>
    {% endfor %}
    </div>
    <div class="actions">
      <button type="submit">🏠 Estimate Price</button>
      <a class="reset" href="/">↺ Reset</a>
    </div>
  </form>
  <footer>Built with Flask &amp; scikit-learn. This is a statistical estimate and not a formal property valuation.</footer>
</div>
</body>
</html>
"""
 
 
# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result, error, values = None, None, {}
 
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            v = {name: float(values[name]) for name in FEATURES}
            X = pd.DataFrame([v])[FEATURES]
 
            price = max(float(model.predict(X)[0]), 0.0)
            result = {
                "price": money(price),
                "snapshot": build_snapshot(v, price),
                "drivers": build_drivers(v),
            }
        except (KeyError, ValueError) as e:
            error = f"Please fill every field with valid values ({e})."
        except Exception as e:  # noqa: BLE001
            error = f"Prediction failed: {e}"
 
    return render_template_string(
        TEMPLATE, sections=SECTIONS, values=values, result=result, error=error
    )
 
 
if __name__ == "__main__":
    app.run(debug=True)
 
