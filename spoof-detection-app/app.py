import os
import json
import warnings
from itertools import count
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import model.inference as deepFakeModel

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_SUBDIR = 'uploads'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, UPLOAD_SUBDIR)
COUNTER_FILE  = os.path.join(UPLOAD_FOLDER, 'counter.txt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_last_count():
    try:
        return int(open(COUNTER_FILE).read().strip())
    except:
        return 0

def save_last_count(n):
    with open(COUNTER_FILE, 'w') as f:
        f.write(str(n))

# Paths for feedback, history logs
HISTORY_FILE = os.path.join(UPLOAD_FOLDER, 'history.json')
FN_FILE = os.path.join(UPLOAD_FOLDER, 'false_negatives.json')
FP_FILE = os.path.join(UPLOAD_FOLDER, 'false_positives.json')

def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except:
        return []

def save_history(hist):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(hist, f, indent=2)

def load_json(path):
    try:
        return json.load(open(path))
    except:
        return []

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

history = load_history()
start_idx = load_last_count()
upload_counter = count(start_idx + 1)
false_neg    = load_json(FN_FILE)
false_pos    = load_json(FP_FILE)

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

bundle     = deepFakeModel.init_models()
midas      = bundle['midas']
classifier = bundle['classifier']
scaler     = bundle['scaler']

@app.route('/', methods=['GET','POST'])
def index():
    filename = label = confidence = error = None
    feedback_flag = False

    if request.method == 'POST':
        f = request.files.get('image')
        if not f:
            error = "No file part."
        elif f.filename == '':
            error = "No file selected."
        elif not allowed_file(f.filename):
            error = "Allowed: png, jpg, jpeg."
        else:
            ext = os.path.splitext(secure_filename(f.filename))[1].lower()
            idx = next(upload_counter)
            save_last_count(idx)
            fn = f"{idx:06d}{ext}"
            path = os.path.join(UPLOAD_FOLDER, fn)
            f.save(path)

            pred, rP, fP = deepFakeModel.test_image(path, classifier, scaler, midas)
            label      = pred
            confidence = f"{(fP if pred=='FAKE' else rP):.2f}%"
            filename   = fn

            entry = {"filename":fn, "label":label, "confidence":confidence}
            history.append(entry)
            save_history(history)
    else:
        filename = request.args.get('filename')
        label = request.args.get('label')
        confidence = request.args.get('confidence')
        feedback_flag = request.args.get('feedback') == "1"
    return render_template('index.html',
                           filename=filename,
                           label=label,
                           confidence=confidence,
                           error=error,
                           feedback=feedback_flag)

@app.route('/feedback', methods=['POST'])
def feedback():
    data         = request.form
    fn           = data.get('filename')
    true_or_not  = data.get('correct')     # "yes" or "no"
    pred_label   = data.get('prediction')  # "REAL" or "FAKE"
    confidence = data.get('confidence')

    # Only record when user says "No" (i.e. prediction was incorrect)
    if true_or_not == 'no' and fn and pred_label:
        error_type = 'fp' if pred_label.upper() == 'REAL' else 'fn'
        record = {
            "filename":  fn,
            "confidence": confidence
        }
        if error_type == 'fn':
            false_neg.append(record)
            save_json(FN_FILE, false_neg)
        else:
            false_pos.append(record)
            save_json(FP_FILE, false_pos)

    return redirect(
        url_for('index',
                filename=fn,
                label=pred_label,
                confidence=confidence,
                feedback="1")
    )


@app.route('/history')
def show_history():
    entries = history[::-1]
    return render_template('history.html', entries=entries)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
