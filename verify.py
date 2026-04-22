"""
verify_model.py  —  Comprehensive model verification tool
Tests your gender + age models in 4 ways:

  1. ARCHITECTURE CHECK   — prints layer structure, param count, input/output shapes
  2. SANITY CHECK         — feeds random noise, checks output is valid probability
  3. FILE TEST            — predict on any .mp3 / .wav file you provide
  4. BATCH TEST           — test a whole folder of audio files, print summary report

Usage:
    python verify_model.py                          # architecture + sanity only
    python verify_model.py mrunali.mp3              # single file
    python verify_model.py path/to/audio/folder/   # entire folder
    python verify_model.py --report                 # full report with class stats
"""

import os
import sys
import argparse
import numpy as np
import librosa
import joblib
import warnings
warnings.filterwarnings('ignore')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# ═══════════════════════════════════════
# CONFIG — edit if your files are elsewhere
# ═══════════════════════════════════════
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
GENDER_MODEL  = os.path.join(SCRIPT_DIR, "gender_model.keras")
GENDER_SCALER = os.path.join(SCRIPT_DIR, "gender_scaler.pkl")
AGE_MODEL     = os.path.join(SCRIPT_DIR, "age_model.keras")
AGE_SCALER    = os.path.join(SCRIPT_DIR, "age_scaler.pkl")

N_MFCC   = 40
SR       = 16000
DURATION = 3
HOP_LEN  = 512

SUPPORTED = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.opus')

# ═══════════════════════════════════════
# COLORS for terminal output
# ═══════════════════════════════════════
class C:
    PASS  = '\033[92m'   # green
    FAIL  = '\033[91m'   # red
    WARN  = '\033[93m'   # yellow
    BLUE  = '\033[94m'   # blue
    BOLD  = '\033[1m'
    DIM   = '\033[2m'
    RESET = '\033[0m'

def ok(msg):   print(f"  {C.PASS}✓{C.RESET} {msg}")
def fail(msg): print(f"  {C.FAIL}✗{C.RESET} {msg}")
def warn(msg): print(f"  {C.WARN}!{C.RESET} {msg}")
def info(msg): print(f"  {C.BLUE}→{C.RESET} {msg}")
def head(msg): print(f"\n{C.BOLD}{msg}{C.RESET}")
def sep():     print("─" * 60)

def bar(pct, w=24):
    filled = int(pct / 100 * w)
    return f"[{'█' * filled}{'░' * (w - filled)}] {pct:5.1f}%"


# ═══════════════════════════════════════
# STEP 1 — FILE CHECK
# ═══════════════════════════════════════
def check_files():
    head("1 / 4  File check")
    sep()
    all_ok = True
    for label, mpath, spath in [
        ("Gender", GENDER_MODEL, GENDER_SCALER),
        ("Age",    AGE_MODEL,    AGE_SCALER),
    ]:
        if os.path.exists(mpath):
            size = os.path.getsize(mpath) / 1e6
            ok(f"{label} model  : {os.path.basename(mpath)}  ({size:.1f} MB)")
        else:
            fail(f"{label} model  : NOT FOUND → {mpath}")
            all_ok = False

        if os.path.exists(spath):
            ok(f"{label} scaler : {os.path.basename(spath)}")
        else:
            fail(f"{label} scaler : NOT FOUND → {spath}")
            all_ok = False

    return all_ok


# ═══════════════════════════════════════
# STEP 2 — LOAD MODELS
# ═══════════════════════════════════════
def load_models():
    head("2 / 4  Loading models")
    sep()
    models = {}

    for label, mpath, spath, key in [
        ("Gender", GENDER_MODEL, GENDER_SCALER, 'gender'),
        ("Age",    AGE_MODEL,    AGE_SCALER,    'age'),
    ]:
        try:
            m    = tf.keras.models.load_model(mpath)
            data = joblib.load(spath)
            models[key] = {'model': m, 'data': data}
            classes = data['le'].classes_
            params  = m.count_params()
            ok(f"{label:6s} loaded  |  classes: {list(classes)}  |  params: {params:,}")
        except Exception as e:
            fail(f"{label:6s} failed to load: {e}")

    return models


# ═══════════════════════════════════════
# STEP 3 — ARCHITECTURE SUMMARY
# ═══════════════════════════════════════
def architecture_check(models):
    head("3 / 4  Architecture check")
    sep()

    for key in ['gender', 'age']:
        if key not in models:
            warn(f"{key} model not loaded, skipping")
            continue

        m = models[key]['model']
        print(f"\n  {C.BOLD}{key.upper()} MODEL{C.RESET}")
        print(f"  {'Layer':<30} {'Output shape':<25} {'Params':>10}")
        print(f"  {'─'*30} {'─'*25} {'─'*10}")

        total = 0
        for layer in m.layers:
            shape  = str(layer.output_shape)
            params = layer.count_params()
            total += params
            marker = f"{C.PASS}*{C.RESET}" if params > 0 else " "
            print(f"  {marker} {layer.name:<29} {shape:<25} {params:>10,}")

        print(f"  {'─'*65}")
        print(f"  {'Total trainable params':<55} {total:>10,}")

        inp  = m.input_shape
        outp = m.output_shape
        info(f"Input  shape : {inp}")
        info(f"Output shape : {outp}  → {len(models[key]['data']['le'].classes_)} classes")


# ═══════════════════════════════════════
# STEP 4 — SANITY CHECK (random noise)
# ═══════════════════════════════════════
def sanity_check(models):
    head("4 / 4  Sanity check  (random noise input)")
    sep()

    for key in ['gender', 'age']:
        if key not in models:
            warn(f"{key} not loaded, skipping")
            continue

        m      = models[key]['model']
        data   = models[key]['data']
        le     = data['le']
        scaler = data['scaler']

        inp_shape = m.input_shape[1:]   # e.g. (40, 94, 3)
        noise     = np.random.randn(1, *inp_shape).astype(np.float32)

        probs = m.predict(noise, verbose=0)[0]

        checks = {
            "Output length matches n_classes" : len(probs) == len(le.classes_),
            "All probs ≥ 0"                   : bool(np.all(probs >= 0)),
            "All probs ≤ 1"                   : bool(np.all(probs <= 1)),
            "Probs sum ≈ 1.0"                 : bool(abs(probs.sum() - 1.0) < 1e-4),
            "No NaN in output"                : bool(not np.any(np.isnan(probs))),
            "No Inf in output"                : bool(not np.any(np.isinf(probs))),
        }

        print(f"\n  {C.BOLD}{key.upper()}{C.RESET}")
        for desc, passed in checks.items():
            (ok if passed else fail)(f"{desc}")

        print(f"\n  Raw output on noise:")
        for cls, p in zip(le.classes_, probs):
            print(f"    {cls:<15} {bar(p*100, 20)}  ({p*100:.2f}%)")


# ═══════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════
def extract(file_path):
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION + 0.5)
    y, _ = librosa.effects.trim(y, top_db=20)
    tlen  = SR * DURATION
    y     = np.pad(y, (0, max(0, tlen - len(y))))[:tlen]
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, delta, delta2], axis=-1).astype(np.float32)


# ═══════════════════════════════════════
# PREDICT ONE FILE
# ═══════════════════════════════════════
def predict_file(file_path, models, verbose=True):
    results = {}
    try:
        feats = extract(file_path)
    except Exception as e:
        if verbose:
            fail(f"Feature extraction failed: {e}")
        return None

    for key in ['gender', 'age']:
        if key not in models:
            continue
        m      = models[key]['model']
        data   = models[key]['data']
        le     = data['le']
        scaler = data['scaler']

        f = feats.copy()
        for ch in range(f.shape[-1]):
            cd = f[:, :, ch].reshape(-1, 1)
            f[:, :, ch] = scaler[ch].transform(cd).reshape(f.shape[0], f.shape[1])

        probs     = m.predict(f[np.newaxis], verbose=0)[0]
        class_idx = np.argmax(probs)
        label     = le.classes_[class_idx]
        conf      = probs[class_idx] * 100
        all_probs = {cls: float(p * 100) for cls, p in zip(le.classes_, probs)}

        results[key] = {'label': label, 'conf': conf, 'probs': all_probs}

    if verbose and results:
        fname = os.path.basename(file_path)
        print(f"\n  {C.BOLD}File:{C.RESET} {fname}")
        print(f"  {'─'*55}")
        for key in ['gender', 'age']:
            if key not in results:
                continue
            r = results[key]
            print(f"\n  {key.upper():<8}  →  {C.BOLD}{r['label'].upper()}{C.RESET}  ({r['conf']:.1f}%)")
            for cls, pct in sorted(r['probs'].items(), key=lambda x: -x[1]):
                marker = f" {C.PASS}◀{C.RESET}" if cls == r['label'] else ""
                print(f"    {cls:<15} {bar(pct)}{marker}")

        if 'gender' in results and 'age' in results:
            g = results['gender']['label']
            a = results['age']['label']
            gc = results['gender']['conf']
            ac = results['age']['conf']
            print(f"\n  {C.BOLD}PROFILE :{C.RESET} {g}, {a}")
            print(f"  Avg confidence: {(gc + ac) / 2:.1f}%")

    return results


# ═══════════════════════════════════════
# BATCH TEST — whole folder
# ═══════════════════════════════════════
def batch_test(folder, models):
    head(f"Batch test  —  {folder}")
    sep()

    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(SUPPORTED)
    ]

    if not files:
        warn(f"No audio files found in {folder}")
        warn(f"Supported: {SUPPORTED}")
        return

    info(f"Found {len(files)} audio files")

    gender_counts = {}
    age_counts    = {}
    failed        = 0

    for fpath in files:
        r = predict_file(fpath, models, verbose=False)
        if r is None:
            failed += 1
            warn(f"Failed: {os.path.basename(fpath)}")
            continue

        if 'gender' in r:
            g = r['gender']['label']
            gender_counts[g] = gender_counts.get(g, 0) + 1
        if 'age' in r:
            a = r['age']['label']
            age_counts[a] = age_counts.get(a, 0) + 1

        g_str = f"{r['gender']['label']} ({r['gender']['conf']:.0f}%)" if 'gender' in r else "—"
        a_str = f"{r['age']['label']} ({r['age']['conf']:.0f}%)"       if 'age'    in r else "—"
        fname = os.path.basename(fpath)[:35]
        print(f"  {fname:<36}  gender: {g_str:<20}  age: {a_str}")

    total = len(files) - failed
    print(f"\n  {'─'*55}")
    print(f"  {C.BOLD}SUMMARY{C.RESET}  ({total} processed, {failed} failed)")

    if gender_counts:
        print(f"\n  Gender distribution:")
        for lbl, cnt in sorted(gender_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            print(f"    {lbl:<12} {bar(pct, 20)}  {cnt}/{total}")

    if age_counts:
        print(f"\n  Age distribution:")
        for lbl, cnt in sorted(age_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            print(f"    {lbl:<12} {bar(pct, 20)}  {cnt}/{total}")


# ═══════════════════════════════════════
# CONFIDENCE CALIBRATION CHECK
# ═══════════════════════════════════════
def calibration_check(models):
    """
    Feeds 50 random noise samples and checks if average confidence
    is close to chance level (1/n_classes). A well-calibrated model
    on random noise should output near-uniform probabilities.
    Overconfidence on noise = model is not well calibrated.
    """
    head("Bonus: Calibration check  (50 noise samples)")
    sep()

    for key in ['gender', 'age']:
        if key not in models:
            continue
        m  = models[key]['model']
        le = models[key]['data']['le']
        n  = len(le.classes_)
        chance = 100.0 / n

        inp_shape = m.input_shape[1:]
        all_confs = []

        for _ in range(50):
            noise = np.random.randn(1, *inp_shape).astype(np.float32)
            probs = m.predict(noise, verbose=0)[0]
            all_confs.append(float(probs.max() * 100))

        avg_conf = np.mean(all_confs)
        print(f"\n  {C.BOLD}{key.upper()}{C.RESET}")
        info(f"Chance level      : {chance:.1f}%  (1/{n} classes)")
        info(f"Avg max confidence: {avg_conf:.1f}%")

        gap = avg_conf - chance
        if gap < 10:
            ok(f"Calibration looks reasonable  (gap = +{gap:.1f}%)")
        elif gap < 25:
            warn(f"Slightly overconfident on noise  (gap = +{gap:.1f}%)")
        else:
            fail(f"Model is overconfident on noise  (gap = +{gap:.1f}%) — likely overfit")


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Verify voice gender + age models",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("target", nargs="?",
                        help="Audio file or folder to test (optional)")
    parser.add_argument("--report",     action="store_true",
                        help="Run all checks including calibration")
    parser.add_argument("--no-arch",    action="store_true",
                        help="Skip architecture printout")
    parser.add_argument("--gender-model", default="gender_model.keras")
    parser.add_argument("--age-model",    default="age_model.keras")
    parser.add_argument("--gender-scaler", default="gender_scaler.pkl")
    parser.add_argument("--age-scaler",    default="age_scaler.pkl")
    args = parser.parse_args()

    global GENDER_MODEL, AGE_MODEL, GENDER_SCALER, AGE_SCALER
    GENDER_MODEL  = args.gender_model
    AGE_MODEL     = args.age_model
    GENDER_SCALER = args.gender_scaler
    AGE_SCALER    = args.age_scaler

    print(f"\n{'═'*60}")
    print(f"  Voice Model Verification Tool")
    print(f"{'═'*60}")

    files_ok = check_files()
    if not files_ok:
        print(f"\n{C.FAIL}Some model files are missing. Check paths above.{C.RESET}")
        sys.exit(1)

    models = load_models()
    if not models:
        print(f"\n{C.FAIL}No models could be loaded.{C.RESET}")
        sys.exit(1)

    if not args.no_arch:
        architecture_check(models)

    sanity_check(models)

    if args.target:
        if os.path.isdir(args.target):
            batch_test(args.target, models)
        elif os.path.isfile(args.target):
            head("File prediction")
            sep()
            predict_file(args.target, models, verbose=True)
        else:
            fail(f"Target not found: {args.target}")

    if args.report:
        calibration_check(models)

    print(f"\n{'═'*60}")
    print(f"  Verification complete")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()