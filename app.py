import os
import re
import io
import numpy as np
import pickle
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, redirect
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

tokenizer = None
lstm_model = None
tfidf_vectorizer = None
logistic_model = None
rf_model = None

MAX_SEQUENCE_LENGTH = 200
MODEL_ACCURACY = 98

# URL fetch
try:
    import requests
    from bs4 import BeautifulSoup
    URL_FETCH_AVAILABLE = True
except ImportError:
    URL_FETCH_AVAILABLE = False

# OCR - try pytesseract first, then easyocr
OCR_AVAILABLE = False
OCR_ENGINE = None

def _init_ocr():
    global OCR_AVAILABLE, OCR_ENGINE
    # Try pytesseract (needs Tesseract installed)
    try:
        import pytesseract
        from PIL import Image
        # Windows common path
        for path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        pytesseract.get_tesseract_version()
        OCR_ENGINE = "pytesseract"
        OCR_AVAILABLE = True
        return
    except Exception:
        pass
    # Try easyocr (no external deps, but heavy)
    try:
        import easyocr
        from PIL import Image
        OCR_ENGINE = "easyocr"
        OCR_AVAILABLE = True
    except ImportError:
        pass

_init_ocr()

# Fraud indicators
FRAUD_INDICATORS = [
    (r'\bdeposit\b', "Requests payment/deposit"),
    (r'\brefundable\b', "Mentions refundable fees"),
    (r'\bwire.?transfer\b', "Asks for wire transfer"),
    (r'\bwestern union\b', "Western Union payment"),
    (r'\bmoney.?gram\b', "MoneyGram payment"),
    (r'\bimmediate.?hire\b', "Immediate hire - urgent pressure"),
    (r'\bno.?experience.?needed\b', "No experience required"),
    (r'\btraining.?fee\b', "Training/registration fee"),
    (r'\bregistration.?fee\b', "Registration fee"),
    (r'\badmin.?fee\b', "Administrative fee"),
    (r'\bapply.?via.?whatsapp\b', "Apply via WhatsApp only"),
    (r'\bapply.?via.?telegram\b', "Apply via Telegram only"),
    (r'\brefundable\s+(rs\.?|₹|\$)\s*\d+', "Refundable deposit amount"),
    (r'\b(rs\.?|₹)\s*\d{4,}', "Large amount in rupees"),
    (r'\bno.?interview\b', "No interview process"),
]

LEGIT_SIGNALS = [
    (r'\b(bachelor|master|b\.?s\.?|m\.?s\.?|phd)\b', "Educational requirements specified"),
    (r'\b(years?\s+of\s+experience|experience\s+required)\b', "Experience requirements clear"),
    (r'\b(health\s+insurance|benefits|401k|retirement)\b', "Benefits mentioned"),
    (r'\b(careers\s+page|apply\s+online|linkedin)\b', "Professional application process"),
    (r'\b(salary|compensation|pay\s+range)\b', "Salary/compensation discussed"),
]

def load_models():
    global tokenizer, lstm_model, tfidf_vectorizer, logistic_model, rf_model
    print("Loading models...")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    lstm_model = tf.keras.models.load_model("fake_job_lstm_model.h5")
    if os.path.exists("tfidf_vectorizer.pkl"):
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open("logistic_model.pkl", "rb") as f:
            logistic_model = pickle.load(f)
        with open("random_forest_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        print("  [OK] All models loaded")
    else:
        print("  [WARN] TF-IDF models not found")

load_models()

def clean_text(text):
    text = str(text) if text else ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower().strip()

def preprocess_lstm(text):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, dtype="float32")

def get_rule_based_analysis(text):
    text_lower = text.lower()
    matched_fraud = [(label, pat) for pat, label in FRAUD_INDICATORS if re.search(pat, text_lower, re.I)]
    matched_legit = [(label, pat) for pat, label in LEGIT_SIGNALS if re.search(pat, text_lower, re.I)]
    fraud_score = min(len(matched_fraud) * 0.12, 0.95)
    return fraud_score, list(set(m[0] for m in matched_fraud)), list(set(m[0] for m in matched_legit))

def predict_lstm(text):
    input_data = preprocess_lstm(text)
    raw = lstm_model.predict(input_data, verbose=0)[0][0]
    prob = 1 / (1 + np.exp(-raw))  # sigmoid
    return float(prob)



def predict_tfidf(text):
    if tfidf_vectorizer is None or logistic_model is None or rf_model is None:
        return None, None

    cleaned = clean_text(text)
    X = tfidf_vectorizer.transform([cleaned])

    lr_proba = logistic_model.predict_proba(X)[0][1]

    rf_raw = rf_model.predict(X)[0]
    rf_proba = rf_raw / 100.0

    return lr_proba, rf_proba


def build_reasons(is_fraud, ensemble_prob, fraud_indicators, legit_signals, results):
    reasons = []
    if is_fraud:
        if fraud_indicators:
            reasons.append({"type": "warning", "text": f"Detected {len(fraud_indicators)} red flag(s):", "subitems": fraud_indicators})
        if ensemble_prob > 0.7:
            reasons.append({"type": "model", "text": "AI models show strong fraud signal."})
    else:
        if legit_signals:
            reasons.append({"type": "positive", "text": "Positive signals found:", "subitems": legit_signals[:5]})
        if not fraud_indicators:
            reasons.append({"type": "positive", "text": "No payment requests or urgent pressure language detected."})
    return reasons

def fetch_text_from_url(url):
    if not URL_FETCH_AVAILABLE:
        return None, "Install: pip install requests beautifulsoup4"
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
            tag.decompose()
        # Try job-specific selectors first
        job_selectors = [
            "[class*='job-description']", "[data-job-description]",
            "article", "main", "[role='main']", ".content", ".post", ".entry", "body"
        ]
        text = ""
        for sel in job_selectors:
            try:
                el = soup.select_one(sel)
                if el:
                    t = el.get_text(separator=" ", strip=True)
                    if len(t) > 200:
                        text = t
                        break
            except Exception:
                pass
        if not text:
            text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text).strip()
        if len(text) < 80:
            return None, "Could not extract enough text. This page may require JavaScript or block scraping."
        return text[:20000], None
    except requests.RequestException as e:
        return None, f"Could not fetch URL: {str(e)[:100]}"
    except Exception as e:
        return None, str(e)[:150]

def extract_text_from_image(file_storage):
    if not OCR_AVAILABLE:
        return None, "OCR not available. Install: pip install pytesseract Pillow. Then install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki OR use: pip install easyocr"
    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        img = img.convert("RGB")
        if OCR_ENGINE == "pytesseract":
            import pytesseract
            text = pytesseract.image_to_string(img)
        else:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            result = reader.readtext(np.array(img))
            text = " ".join([r[1] for r in result])
        text = text.strip()
        if len(text) < 15:
            return None, "Could not extract enough text. Try a clearer image with readable text."
        return text, None
    except Exception as e:
        return None, str(e)[:200]

def run_prediction(combined_text):
    cleaned = clean_text(combined_text)
    if len(cleaned.split()) < 5:
        return None, "Please provide more detail for accurate analysis."
    results = {}
    lstm_prob = predict_lstm(cleaned)
    results["lstm"] = {"prob": lstm_prob, "label": "Fraudulent" if lstm_prob > 0.5 else "Legitimate", "pct": round(lstm_prob * 100, 0)}
    lr_proba, rf_proba = predict_tfidf(combined_text)
    if lr_proba is not None:
        results["logistic"] = {"prob": lr_proba, "label": "Fraudulent" if lr_proba > 0.5 else "Legitimate", "pct": round(lr_proba * 100, 0)}
    if rf_proba is not None:
        results["random_forest"] = {"prob": rf_proba, "label": "Fraudulent" if rf_proba > 0.5 else "Legitimate", "pct": round(rf_proba * 100, 0)}
    rule_score, fraud_indicators, legit_signals = get_rule_based_analysis(combined_text)
    results["rule_based"] = {"prob": rule_score, "label": "Fraudulent" if rule_score > 0.5 else "Legitimate", "pct": round(rule_score * 100, 0)}
    if lr_proba is not None:
        ensemble_prob = 0.10 * lstm_prob + 0.93 * lr_proba + 0.00 * rf_proba + 0.10 * rule_score
    else:
        ensemble_prob = 0.85 * lstm_prob + 0.15 * rule_score
    results["ensemble"] = {"prob": ensemble_prob, "label": "Fraudulent" if ensemble_prob > 0.5 else "Legitimate", "pct": round(ensemble_prob * 100, 1)}
    final_label = results["ensemble"]["label"]
    is_fraud = final_label == "Fraudulent"
    confidence = ensemble_prob if is_fraud else (1 - ensemble_prob)
    reasons = build_reasons(is_fraud, ensemble_prob, fraud_indicators, legit_signals, results)
    return {
        "prediction": f"This job posting appears to be {final_label.lower()}.",
        "confidence_pct": round(confidence * 100, 1),
        "is_fraud": is_fraud,
        "results": results,
        "reasons": reasons,
    }, None

# Shared context
def _ctx(**kw):
    return {
        "model_accuracy": MODEL_ACCURACY,
        "url_fetch_available": URL_FETCH_AVAILABLE,
        "ocr_available": OCR_AVAILABLE,
        **kw
    }

@app.route("/")
def index():
    return redirect("/home")

@app.route("/home")
def home():
    return render_template("home.html", **_ctx())

@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    if request.method == "POST":
        combined_text = request.form.get("combined_text", "").strip()
        if not combined_text:
            return render_template("analysis.html", **_ctx(
                show_result=False,
                error="Please enter or paste job posting text, then click Analyze."
            ))
        result, err = run_prediction(combined_text)
        if err:
            return render_template("analysis.html", **_ctx(
                show_result=False,
                combined_text=combined_text,
                error=err
            ))
        return render_template("analysis.html", **_ctx(
            show_result=True,
            combined_text=combined_text,
            prediction=result["prediction"],
            confidence_pct=result["confidence_pct"],
            is_fraud=result["is_fraud"],
            results=result["results"],
            reasons=result["reasons"],
        ))
    prefilled = request.args.get("text", "")
    return render_template("analysis.html", **_ctx(show_result=False, combined_text=prefilled))

@app.route("/examples")
def examples():
    return render_template("examples.html", **_ctx())

@app.route("/tips")
def tips():
    return render_template("tips.html", **_ctx())

@app.route("/visualization")
def viz():
    return render_template("visualization.html", **_ctx())

# API: fetch URL text
@app.route("/api/fetch-url", methods=["POST"])
def api_fetch_url():
    url = (request.get_json() or {}).get("url", "").strip()
    if not url:
        return jsonify({"ok": False, "error": "No URL provided"})
    text, err = fetch_text_from_url(url)
    if err:
        return jsonify({"ok": False, "error": err})
    return jsonify({"ok": True, "text": text})

# API: extract text from image
@app.route("/api/extract-image", methods=["POST"])
def api_extract_image():
    if "image" not in request.files and "image_file" not in request.files:
        return jsonify({"ok": False, "error": "No image file provided"})
    f = request.files.get("image") or request.files.get("image_file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "No image file selected"})
    text, err = extract_text_from_image(f)
    if err:
        return jsonify({"ok": False, "error": err})
    return jsonify({"ok": True, "text": text})

@app.route("/predict", methods=["POST"])
def predict():
    combined_text = request.form.get("combined_text", "").strip()
    if not combined_text:
        return render_template("analysis.html", **_ctx(
            show_result=False,
            error="Please enter or paste job posting text, then click Analyze."
        ))
    result, err = run_prediction(combined_text)
    if err:
        return render_template("analysis.html", **_ctx(
            show_result=False,
            combined_text=combined_text,
            error=err
        ))
    return render_template("analysis.html", **_ctx(
        show_result=True,
        combined_text=combined_text,
        prediction=result["prediction"],
        confidence_pct=result["confidence_pct"],
        is_fraud=result["is_fraud"],
        results=result["results"],
        reasons=result["reasons"],
    ))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
