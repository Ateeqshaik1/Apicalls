#!/usr/bin/env python3
"""
health_risk_with_keywords.py

- Adds FEATURE_KEYWORDS for Liver, Kidney, Diabetes, and Heart using the sample reports you provided.
- OCR (EasyOCR preferred, pytesseract fallback) -> parse -> map to canonical features.
- Training template (RandomForest) per disease; inference helper that returns risk percentages.

Usage:
    1. Install dependencies (suggested):
       pip install pandas numpy scikit-learn joblib easyocr pytesseract opencv-python pillow
       - If using pytesseract, install tesseract binary separately.
    2. Place your training CSVs in /mnt/data or adjust paths in CSV_PATHS.
    3. Run: python health_risk_with_keywords.py

Note:
- The OCR parsing is heuristic — adjust FEATURE_KEYWORDS / regexes for your lab report variations for best extraction.
- This script doesn't automatically retrain models on arbitrary label formats. Make sure your datasets' label columns are correct.
"""

import re
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Optional OCR libs
try:
    import easyocr
except Exception:
    easyocr = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import cv2
    from PIL import Image
except Exception:
    cv2 = None

# ---------- Config ----------
DATA_DIR = Path('/mnt/data')
SAMPLE_LIVER_IMAGE = DATA_DIR / 'livertestsample.png'
SAMPLE_DIABETES_IMAGE = DATA_DIR / 'diabetes_report.png'
SAMPLE_HEART_IMAGE = DATA_DIR / 'heart_report.png'
SAMPLE_KIDNEY_IMAGE = DATA_DIR / 'kidneyreport.png'

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

# ---------- FEATURE KEYWORDS (Liver, Kidney, Diabetes, Heart) ----------
# Built from the sample reports you provided. Expand synonyms as needed.
FEATURE_KEYWORDS = {
    # LIVER LFT
    'bilirubin_total': ['bilirubin total', 'total bilirubin', 'bilirubin_total', 'bilirubintotal'],
    'bilirubin_direct': ['bilirubin direct', 'direct bilirubin', 'bilirubin_direct'],
    'bilirubin_indirect': ['bilirubin indirect', 'indirect bilirubin', 'bilirubin_indirect'],
    'sgpt_alt': ['sgpt', 'alt', 'alanine aminotransferase'],
    'sgot_ast': ['sgot', 'ast', 'aspartate aminotransferase'],
    'sgot_sgpt_ratio': ['sgot/sgpt ratio', 'sgot/sgpt', 'sgot sgpt ratio'],
    'alkaline_phosphatase': ['alkaline phosphatase', 'alp', 'alk phos'],
    'ggt': ['gamma glutamyl transferase', 'ggt'],
    'total_proteins': ['total proteins', 'total protein'],
    'albumin': ['albumin'],
    'globulin': ['globulin'],
    'a_g_ratio': ['a : g ratio', 'a:g ratio', 'a/g ratio', 'a : g'],

    # KIDNEY / KFT
    'bun': ['bun', 'blood urea nitrogen', 'blood urea'],
    'serum_urea': ['serum urea', 'urea'],
    'serum_creatinine': ['serum creatinine', 'creatinine', 'cr'],
    'egfr': ['egfr', 'e g f r', 'estimated glomerular filtration rate'],
    'egfr_category': ['egfr category', 'egfr_category'],
    'serum_calcium': ['serum calcium', 'calcium'],
    'serum_potassium': ['serum potassium', 'potassium', 'k+'],
    'serum_sodium': ['serum sodium', 'sodium', 'na'],
    'serum_uric_acid': ['serum uric acid', 'uric acid'],
    'urea_creatinine_ratio': ['urea / creatinine ratio', 'urea/creatinine ratio'],
    'bun_creatinine_ratio': ['bun / creatinine ratio', 'bun/creatinine ratio'],

    # DIABETES / GLUCOSE
    'fbs': ['fasting blood sugar', 'fasting blood glucose', 'fbs', 'fasting sugar'],
    'ppbs': ['postprandial', 'post prandial', 'postprandial blood sugar', 'ppbs'],
    'random_blood_sugar': ['random blood sugar', 'rbs', 'random glucose'],
    'hba1c': ['hba1c', 'hb a1c', 'a1c'],
    'glucose': ['glucose'],  # general
    'bmi': ['bmi'],
    'weight': ['weight', 'wt'],
    'height': ['height', 'ht'],

    # HEART / LIPID / VITALS / ECHO
    'cholesterol_total': ['cholesterol', 'total cholesterol', 'cholesterol total'],
    'triglycerides': ['triglyceride', 'triglycerides', 'tg'],
    'hdl': ['hdl', 'hdl cholesterol'],
    'ldl': ['ldl', 'ldl cholesterol'],
    'vldl': ['vldl'],
    'chol_hdl_ratio': ['cholesterol/hdl ratio', 'chol/hdl', 'cholesterol to hdl ratio'],
    # Vitals (from diabetes report screenshot)
    'blood_pressure_sys': ['blood pressure', 'bp', 'bloodpressure.systolic'],  # special parse for x/y
    'blood_pressure_dia': ['blood pressure', 'bp', 'bloodpressure.diastolic'],
    'heart_rate': ['heart rate', 'hr'],
    'spo2': ['spo2', 'sp02', 'o2 sat', 'oxygen saturation'],
    # Echo measures (if present in echo report)
    'ef_percent': ['ef', 'ejection fraction', 'ef (a4c)'],
}

# ---------- OCR helpers ----------
def init_easyocr_reader(langs=['en']):
    if easyocr is None:
        raise ImportError("easyocr is not installed. Install via: pip install easyocr")
    return easyocr.Reader(langs, gpu=False)

def preprocess_image_for_ocr(image_path: Path, out_tmp: bool = True) -> str:
    """Simple preprocessing: grayscale, resize, threshold. Returns path to processed image."""
    if cv2 is None:
        return str(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        return str(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize (scale up small images)
    h, w = gray.shape[:2]
    scale = max(1, 1200 / max(h, w))
    if scale != 1:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tmp_path = str(image_path.with_suffix('')) + '_ocr_tmp.png' if out_tmp else str(image_path)
    cv2.imwrite(tmp_path, th)
    return tmp_path

def ocr_image_to_text(image_path: Path, reader=None) -> str:
    """Prefer easyocr reader if provided; fallback to pytesseract."""
    img_path = preprocess_image_for_ocr(image_path)
    if reader is not None:
        try:
            res = reader.readtext(img_path, detail=0)
            return "\n".join(res)
        except Exception:
            pass
    if pytesseract is not None:
        try:
            return pytesseract.image_to_string(Image.open(img_path))
        except Exception:
            # last fallback: try cv2 -> pytesseract
            try:
                import cv2
                img = cv2.imread(img_path)
                return pytesseract.image_to_string(img)
            except Exception:
                return ""
    raise RuntimeError("No OCR backend available. Install easyocr or pytesseract + tesseract binary.")

# ---------- Parsing utilities ----------
def parse_bp_from_text(text: str) -> Tuple[Any, Any]:
    """
    Find first blood pressure pattern like '150/90' and return systolic, diastolic as floats.
    """
    m = re.search(r'(\b[1-2]?\d{2})\s*/\s*([4-9]?\d)\b', text)  # approximate systolic/diastolic ranges
    if m:
        try:
            s = float(m.group(1))
            d = float(m.group(2))
            return s, d
        except:
            return None, None
    # alternative forms like 'Blood Pressure : 150/90 mmHg'
    m2 = re.search(r'bp[:\s]([0-9]{2,3})\s/\s*([0-9]{2,3})', text, re.IGNORECASE)
    if m2:
        return float(m2.group(1)), float(m2.group(2))
    return None, None

def parse_hr_from_text(text: str) -> Any:
    m = re.search(r'heart\s*rate\s*[:\s]*([0-9]{2,3})', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # look for pattern '80 bpm'
    m2 = re.search(r'([6-1][0-9]|[7-9][0-9]|1[0-9]{2})\s*bpm', text, re.IGNORECASE)
    if m2:
        return float(m2.group(1))
    return None

def parse_spo2_from_text(text: str) -> Any:
    m = re.search(r'spo2\s*[:\s]([0-9]{2})\s%?', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def find_generic_numeric_after_keyword(line: str) -> Tuple[Any, Any]:
    """Return first numeric token in a line as float, and its string representation."""
    m = re.search(r'([<>]?)\s*([-+]?[0-9]*\.?[0-9]+)', line)
    if not m:
        return None, None
    sym, num = m.group(1), m.group(2)
    try:
        val = float(num)
        if sym == '<':
            val = val * 0.9
        elif sym == '>':
            val = val * 1.1
        return num, val
    except:
        return None, None

def parse_report_text_to_features(text: str, debug: bool=False) -> Dict[str, float]:
    """
    Parse OCR text into canonical features using FEATURE_KEYWORDS plus patterns for BP, HR, SPO2, eGFR.
    Returns a dict: {feature_name: numeric_value}
    """
    out = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lowtext = text.lower()

    # 1) Special multi-value parsers: BP, HR, SPO2, eGFR, EF
    bp_s, bp_d = parse_bp_from_text(text)
    if bp_s is not None:
        out['blood_pressure_sys'] = bp_s
    if bp_d is not None:
        out['blood_pressure_dia'] = bp_d

    hr = parse_hr_from_text(text)
    if hr is not None:
        out['heart_rate'] = hr

    sp = parse_spo2_from_text(text)
    if sp is not None:
        out['spo2'] = sp

    # eGFR patterns (may have 'EGFR 50.32' or 'eGFR: 50.32 ml/min/1.73m^2' or a leading letter 'L 50.32')
    m_egfr = re.search(r'e-?gfr[:\s]([0-9]+\.?[0-9])', lowtext, re.IGNORECASE)
    if not m_egfr:
        # look for patterns like 'L 50.32' preceded by EGFR label line
        m2 = re.search(r'egfr.\n.?([0-9]+\.?[0-9]*)', lowtext, re.IGNORECASE)
        if m2:
            m_egfr = m2
    if m_egfr:
        try:
            out['egfr'] = float(m_egfr.group(1))
        except:
            pass

    # EF (ejection fraction) patterns, common in echo (e.g., 'EF (A4C) 68.8 %')
    m_ef = re.search(r'(ef|ejection fraction)[\s:\(]([0-9]{1,3}\.?[0-9])', lowtext, re.IGNORECASE)
    if m_ef:
        try:
            out['ef_percent'] = float(m_ef.group(2))
        except:
            pass

    # 2) Keyword-based extraction: scan every line for keywords and extract numeric
    for ln in lines:
        ln_low = ln.lower()
        for canon, synonyms in FEATURE_KEYWORDS.items():
            # Skip BP/HR/SPO2 keys handled above
            if canon in ['blood_pressure_sys', 'blood_pressure_dia', 'heart_rate', 'spo2', 'egfr', 'ef_percent']:
                continue
            for kw in synonyms:
                kw_low = kw.lower()
                if kw_low in ln_low:
                    # attempt to extract numeric value from this line
                    numstr, val = find_generic_numeric_after_keyword(ln)
                    if val is not None:
                        # some fields like 'a : g ratio 1.43' parse fine; but sometimes 'Total Proteins 8.5 gm/dL'
                        out[canon] = float(val)
                        if debug:
                            print(f"Parsed {canon} = {val} from line: {ln}")
                        break
            if canon in out:
                break

    # 3) Try global numeric search for some fields if not found (e.g., if values on separate columns)
    # Example: many lab tables have "TEST   RESULT   REF RANGE"; OCR may split words. We try to find known labels and nearby tokens.
    # (This is a best-effort heuristic.)
    if debug:
        print("Final parsed features:", out)
    return out

# ---------- Example: quick test using sample images (won't train models) ----------
def example_parse_samples(debug=False):
    # initialize reader if available
    reader = None
    if easyocr is not None:
        try:
            reader = init_easyocr_reader(['en'])
        except Exception:
            reader = None

    samples = {
        'liver': SAMPLE_LIVER_IMAGE,
        'diabetes': SAMPLE_DIABETES_IMAGE,
        'heart': SAMPLE_HEART_IMAGE,
        'kidney': SAMPLE_KIDNEY_IMAGE
    }
    results = {}
    for name, img_path in samples.items():
        if not img_path.exists():
            print(f"[WARN] sample image not found: {img_path}")
            results[name] = {}
            continue
        try:
            text = ocr_image_to_text(img_path, reader=reader)
        except Exception as e:
            print(f"[ERROR] OCR failed for {img_path}: {e}")
            results[name] = {}
            continue
        feats = parse_report_text_to_features(text, debug=debug)
        results[name] = feats
        print(f"---- Parsed ({name}) ----")
        for k, v in feats.items():
            print(f"  {k}: {v}")
    return results

# ---------- Template training & inference (user dataset required) ----------
def train_per_disease_models(csv_map: Dict[str, str], label_col_map: Dict[str, str]):
    """
    csv_map: {'liver': '/path/to/liver.csv', ...}
    label_col_map: {'liver': 'Dataset', 'diabetes': 'Outcome', ...}
    This function trains a RandomForest per disease and saves pipelines to MODEL_DIR.
    """
    pipelines = {}
    for disease, csv_path in csv_map.items():
        p = Path(csv_path)
        if not p.exists():
            print(f"[WARN] missing CSV for {disease}: {csv_path}")
            continue
        df = pd.read_csv(p)
        label_col = label_col_map.get(disease)
        if label_col not in df.columns:
            # try case-insensitive search
            found = None
            for c in df.columns:
                if c.lower() == label_col.lower():
                    found = c; break
            if found:
                label_col = found
            else:
                print(f"[WARN] Label column {label_col} not found in {csv_path}; skipping")
                continue
        # features & label
        X = df.drop(columns=[label_col])
        X = X.apply(pd.to_numeric, errors='coerce')
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        y_raw = df[label_col]
        # Binarize label if necessary: common datasets use 1/0 or 'yes'/'no'
        y = pd.to_numeric(y_raw, errors='coerce')
        if set(y.dropna().unique()) - {0, 1}:
            threshold = y.median()
            y = (y >= threshold).astype(int)
        else:
            y = y.fillna(0).astype(int)

        # Build pipeline and train
        numeric_cols = list(X_imp.columns)
        preproc = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols)
        ])
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        pipe = Pipeline([('preproc', preproc), ('clf', rf)])

        X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)
        pipe.fit(X_tr, y_tr)
        # save pipeline
        outp = MODEL_DIR / f"{disease}_rf.joblib"
        joblib.dump(pipe, outp)
        pipelines[disease] = pipe
        print(f"[INFO] Trained and saved {disease} model to {outp}")
    return pipelines

def predict_risk_from_features(pipelines: Dict[str, Pipeline], features: Dict[str, float]) -> Dict[str, float]:
    """
    Given pipelines (per-disease) and a parsed features dict, returns disease risk percentages (0-100).
    It aligns features to pipeline expected columns and fills missing values with medians.
    """
    results = {}
    for disease, pipe in pipelines.items():
        try:
            expected = pipe.named_steps['preproc'].transformers_[0][2]
            row = {}
            for col in expected:
                row[col] = features.get(col, np.nan)
            # impute median for missing
            row_vals = np.array([row[c] for c in expected], dtype=float).reshape(1, -1)
            # Many pipelines already include imputer+scaler; just call predict_proba directly
            proba = pipe.predict_proba(row_vals)[0][1]
            results[disease] = float(proba * 100.0)
        except Exception as e:
            results[disease] = None
    return results

# ---------- If module run directly: run example parsing ----------
if _name_ == "_main_":
    print("=== Running sample parse on images you uploaded (if present) ===")
    parsed = example_parse_samples(debug=True)
    print("\nParsed outputs (JSON):")
    print(json.dumps(parsed, indent=2))
    print("\n=== End ===\n")
    # Note: training functions are available above; call train_per_disease_models(...) with paths if you want to train.