import io, os, re, csv, sys, json, uuid, random, sqlite3
from flask import Flask, request, jsonify, Response, send_file, session
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("TS_DB", os.path.join(BASE, "talentscreen.db"))
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

SKILLS = ["python", "java", "c++", "javascript", "react", "sql", "machine learning", "data science", "nlp", "aws",
          "excel", "power bi", "flask", "django", "git", "docker", "kubernetes", "tensorflow", "pandas", "html", "css",
          "node.js", "tableau", "linux", "azure", "jenkins", "mysql", "mongodb", "figma", "networking", "firewall",
          "siem", "statistics", "recruitment", "payroll", "onboarding"]
CATS = {
    "Data Science": ["python", "machine learning", "statistics", "pandas", "tensorflow", "nlp", "data science", "sql"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "statistics", "python", "pandas"],
    "Machine Learning": ["machine learning", "tensorflow", "python", "nlp", "pandas", "statistics"],
    "Python Developer": ["python", "flask", "django", "git", "sql", "docker"],
    "Java Developer": ["java", "sql", "git", "mysql", "jenkins"],
    "Web Developer": ["javascript", "react", "html", "css", "node.js", "git", "mongodb"],
    "Software Developer": ["java", "c++", "python", "git", "sql", "linux"],
    "DevOps": ["docker", "kubernetes", "jenkins", "aws", "linux", "git"],
    "Cloud Engineer": ["aws", "azure", "docker", "kubernetes", "linux"],
    "Database Administrator": ["sql", "mysql", "mongodb", "linux", "azure"],
    "HR": ["recruitment", "payroll", "onboarding", "excel"],
    "Business Analyst": ["excel", "sql", "power bi", "tableau", "statistics"],
    "UI/UX Designer": ["figma", "html", "css"],
    "Network Engineer": ["networking", "firewall", "linux"],
    "Cyber Security": ["firewall", "siem", "networking", "linux"]}

S = {"resumes": {}, "jds": {}, "jd_id": None, "log": [],
     "settings": {"w_sim": 60, "shortlist": 55, "skills": list(SKILLS)}}


# ---------- SQLite persistence (stdlib, no new dependency) ----------
def q(sql, a=()):
    c = sqlite3.connect(DB)
    try:
        rows = c.execute(sql, a).fetchall(); c.commit(); return rows
    finally:
        c.close()


def init_db():
    for t in ("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, pw TEXT, role TEXT)",
              "CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT)",
              "CREATE TABLE IF NOT EXISTS resumes(id TEXT PRIMARY KEY, data TEXT)",
              "CREATE TABLE IF NOT EXISTS jds(id TEXT PRIMARY KEY, data TEXT)"):
        q(t)
    if not q("SELECT 1 FROM users"):
        q("INSERT INTO users VALUES(?,?,?)", ("admin", generate_password_hash(os.environ.get("ADMIN_PASSWORD", "admin123")), "Recruiter"))


def kv_set(k, v): q("REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v)))
def save(tbl, o): q(f"REPLACE INTO {tbl} VALUES(?,?)", (o["id"], json.dumps(o)))
def drop(tbl, i): q(f"DELETE FROM {tbl} WHERE id=?", (i,))


def load_db():
    S["resumes"] = {r[0]: json.loads(r[1]) for r in q("SELECT id,data FROM resumes ORDER BY rowid")}
    S["jds"] = {r[0]: json.loads(r[1]) for r in q("SELECT id,data FROM jds ORDER BY rowid")}
    kv = {k: json.loads(v) for k, v in q("SELECT k,v FROM kv")}
    S["settings"].update(kv.get("settings", {})); S["jd_id"] = kv.get("jd_id"); S["log"] = kv.get("log", [])
    if "secret" not in kv:
        kv["secret"] = uuid.uuid4().hex; kv_set("secret", kv["secret"])
    app.secret_key = os.environ.get("SECRET_KEY", kv["secret"])


# ---------- NLP preprocessing ----------
try:
    from nltk.stem import WordNetLemmatizer
    _wl = WordNetLemmatizer(); _wl.lemmatize("tests")
    lem = _wl.lemmatize
except Exception:  # WordNet data not installed: minimal plural stripping
    lem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "is", "us")) else w


def pre(text):
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+#.\s]", " ", text.lower())).strip()
    tokens = [t.strip(".") for t in cleaned.split()]
    tokens = [t for t in tokens if t]
    nostop = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    lemmas = [lem(t) for t in nostop]
    return {"original": text[:500], "cleaned": cleaned[:500], "tokens": tokens[:50], "nostop": nostop[:50],
            "lemmas": lemmas[:50], "processed": " ".join(lemmas)}


def find_skills(text):
    t = text.lower()
    return [s for s in S["settings"]["skills"] if re.search(r"(?<![\w+#])" + re.escape(s) + r"(?![\w+#])", t)]


# ---------- text extraction ----------
def read(name, data):
    ext = name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        import fitz
        t = "\n".join(p.get_text() for p in fitz.open(stream=data, filetype="pdf"))
        if len(t.strip()) < 30:
            return "", "Scanned/image-based PDF detected. OCR is required."
        return t, "Extracted"
    if ext == "docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs), "Extracted"
    if ext == "txt":
        return data.decode("utf-8", "ignore"), "Extracted"
    return "", "Unsupported format"


# ---------- information extraction ----------
def info(t, fn):
    L = [l.strip() for l in t.splitlines() if l.strip()]
    g = lambda p, f=re.I: (re.search(p, t, f) or [None])[0]
    name = next((l for l in L[:6] if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{2,40}", l) and len(l.split()) <= 4),
                fn.rsplit(".", 1)[0])
    college = next((l for l in L if re.search(r"universit|college|institute", l, re.I)), None)
    gpa = re.search(r"(?:cgpa|gpa)\s*[:\-]?\s*(\d+(?:\.\d+)?)", t, re.I) or re.search(r"(\d{2}(?:\.\d+)?)\s*%", t)
    ys = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", t, re.I)]
    loc = re.search(r"(?:location|address|city)\s*[:\-]\s*(.+)", t, re.I)
    pi = next((i for i, l in enumerate(L) if re.match(r"projects?\b", l, re.I)), None)
    titles = re.findall(r"(?:senior |junior )?(?:software|data|python|java|web|devops|cloud|machine learning|full stack|"
                        r"frontend|backend)\s+(?:engineer|developer|analyst|scientist|intern|administrator)", t, re.I)
    return {"name": name, "email": g(r"[\w.+-]+@[\w-]+\.[\w.]+"),
            "phone": g(r"\+\d{1,3}[\s-]?\d{5}[\s-]?\d{5}|\b\d{10}\b"),
            "location": loc.group(1).strip() if loc else None,
            "degree": g(r"\b(?:B\.?Tech|M\.?Tech|B\.?E\.?|M\.?E\.?|B\.?Sc|M\.?Sc|BCA|MCA|MBA|Ph\.?D|"
                        r"Bachelor[^\n,]{0,30}|Master[^\n,]{0,30})", 0),
            "college": college, "grad_year": g(r"\b(?:19|20)\d{2}\b") if not college else
            (re.findall(r"\b(?:19|20)\d{2}\b", college) or [None])[-1],
            "gpa": gpa.group(1) if gpa else None, "exp": min(max(ys), 40) if ys else 0,
            "companies": [l for l in L if re.search(r"\b(?:pvt|ltd|inc|llc|technologies|solutions|corp|systems|infotech)\b", l, re.I)][:4],
            "titles": sorted({x.title() for x in titles})[:5], "interns": len(re.findall(r"intern", t, re.I)),
            "certs": [l for l in L if re.search(r"certif", l, re.I)][:5],
            "projects": L[pi + 1:pi + 6] if pi is not None else []}


# ---------- classification ----------
def train():
    p = os.path.join(BASE, "data", "resume_dataset.csv")
    if os.path.exists(p):
        import pandas as pd
        df = pd.read_csv(p)
        X = [pre(t)["processed"] for t in df["Resume"]]; y = list(df["Category"])
        src = "Dataset: " + p
    else:
        rnd = random.Random(1); X = []; y = []
        for c, ks in CATS.items():
            for _ in range(80):
                w = rnd.sample(ks, rnd.randint(max(2, len(ks) // 2), len(ks))) + rnd.sample(SKILLS, 3)
                X.append(" ".join(w)); y.append(c)
        src = "Synthetic demo data (self-test only, NOT real-world accuracy). Add data/resume_dataset.csv to train on real resumes."
    mk = lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                               LogisticRegression(max_iter=1000))
    a, b, ya, yb = train_test_split(X, y, test_size=.25, random_state=1, stratify=y)
    m = mk().fit(a, ya); pr = m.predict(b)
    met = {"accuracy": round(accuracy_score(yb, pr), 3), "f1_macro": round(f1_score(yb, pr, average="macro"), 3),
           "samples": len(X), "source": src}
    return mk().fit(X, y), met


MODEL, METRICS = train()


# ---------- matching ----------
def score_for(r, jd):
    """Score one resume against one JD (pure function: used for the active JD and for best-fit ranking)."""
    res = r["res"]; st = S["settings"]; have = set(res["skills"])
    req = [x.lower() for x in jd["required"]] or find_skills(jd["text"])
    pref = [x.lower() for x in jd["preferred"]]
    m = [x for x in req if x in have]
    try:
        v = TfidfVectorizer().fit([jd["proc"], r["proc"]])
        sim = float(cosine_similarity(v.transform([jd["proc"]]), v.transform([r["proc"]]))[0][0])
    except ValueError:
        sim = 0.0
    cov = len(m) / len(req) if req else 0
    w1, w2 = st["w_sim"], 100 - st["w_sim"]
    score = round((w1 * min(1, sim * 2) + w2 * cov), 1)  # text similarity is scaled x2 (cap 1): raw TF-IDF cosine is low
    ok = res["exp"] >= jd["min_exp"]
    return dict(score=score, sim=round(sim, 3), cov=round(cov, 3), matched=m, missing=[x for x in req if x not in have],
                extra=sorted(x for x in have if x not in req), pref_hit=[x for x in pref if x in have], req_n=len(req),
                exp_ok=ok, status="Shortlisted" if score >= st["shortlist"] and ok else "Rejected")


def match(r):
    jd = S["jds"].get(S["jd_id"]); res = r["res"]
    if not jd:
        return
    res.update(score_for(r, jd), jd=jd["title"])
    res["auto"] = res["status"]
    if res.get("decision"):  # recruiter's manual decision overrides the automatic status
        res["status"] = res["decision"]


def process(r):
    old = r.get("res") or {}
    p = pre(r["text"]); r["proc"] = p["processed"]
    pr = MODEL.predict_proba([p["processed"]])[0]; cl = MODEL[-1].classes_; i = int(pr.argmax())
    res = info(r["text"], r["name"])
    res.update(skills=find_skills(r["text"]), pre=p, category=cl[i] if pr[i] >= .3 else "Other", conf=round(float(pr[i]), 3),
               top=[[c, round(float(q_), 3)] for c, q_ in sorted(zip(cl, pr), key=lambda x: -x[1])[:3]],
               decision=old.get("decision", ""), notes=old.get("notes", ""))
    r["res"] = res; r["proc_status"] = "Processed"; match(r)
    S["log"].insert(0, f"Processed {r['name']} → {res['category']}"); del S["log"][12:]
    save("resumes", r); kv_set("log", S["log"])


def add_jd(title, text, req, pref, mn):
    i = uuid.uuid4().hex[:8]
    sp = lambda s: [x.strip().lower() for x in (s or "").split(",") if x.strip()]
    try: mn = float(mn or 0)
    except ValueError: mn = 0.0
    S["jds"][i] = dict(id=i, title=title, text=text, required=sp(req), preferred=sp(pref), min_exp=mn,
                       proc=pre(title + " " + text)["processed"])
    S["jd_id"] = i; save("jds", S["jds"][i]); kv_set("jd_id", i)
    return i


def seed():
    for t, txt, rq, pf, mn in [
        ("Python Developer", "Backend Python developer building Flask REST APIs with SQL databases and Git workflows.", "python, sql, flask, git", "aws, docker", 1),
        ("Data Scientist", "Build machine learning models with Python, pandas and statistics; NLP experience preferred.", "python, machine learning, statistics, pandas", "nlp, tensorflow, sql", 1)]:
        add_jd(t, txt, rq, pf, mn)


init_db(); load_db()
if not S["jds"]: seed()
if S["jd_id"] not in S["jds"]:
    S["jd_id"] = next(iter(S["jds"])); kv_set("jd_id", S["jd_id"])


# ---------- auth (users live in the database; no registration page) ----------
@app.before_request
def guard():
    if request.path.startswith("/api/") and request.endpoint != "login" and "user" not in session:
        return jsonify(error="auth"), 401


@app.route("/api/login", methods=["POST"])
def login():
    b = request.json or {}
    row = q("SELECT pw FROM users WHERE username=?", (str(b.get("username", "")).strip(),))
    if row and check_password_hash(row[0][0], str(b.get("password", ""))):
        session["user"] = b["username"].strip(); return jsonify(ok=True)
    return jsonify(ok=False, error="Invalid username or password"), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear(); return jsonify(ok=True)


# ---------- API ----------
@app.route("/")
def home():
    for d in ("static", "templates", "."):
        p = os.path.join(BASE, d, "index.html")
        if os.path.exists(p):
            return send_file(p)
    return "index.html not found. Put it in a folder named 'static' next to app.py.", 404


@app.route("/api/state")
def state():
    emails = [r["res"]["email"].lower() for r in S["resumes"].values() if r["res"] and r["res"].get("email")]
    rs = []
    for r in S["resumes"].values():
        d = {k: r[k] for k in ("id", "name", "fmt", "size", "ext", "proc_status", "res")}
        if r["res"]:
            e = (r["res"].get("email") or "").lower()
            d["res"] = dict(r["res"], dup=bool(e) and emails.count(e) > 1,
                            fits=sorted(([j["title"], score_for(r, j)["score"]] for j in S["jds"].values()), key=lambda x: -x[1]))
        rs.append(d)
    return jsonify(resumes=rs, jds=list(S["jds"].values()), jd_id=S["jd_id"], settings=S["settings"], metrics=METRICS,
                   log=S["log"], user=session["user"])


@app.route("/api/upload", methods=["POST"])
def upload():
    for f in request.files.getlist("files"):
        if not f.filename: continue
        data = f.read(); i = uuid.uuid4().hex[:8]
        try:
            text, ext = read(f.filename, data)
        except Exception as e:
            text, ext = "", "Extraction failed: " + str(e)[:80]
        r = S["resumes"][i] = dict(id=i, name=f.filename, fmt=f.filename.rsplit(".", 1)[-1].upper(), size=len(data), text=text,
                                   ext=ext, proc_status="Pending", res=None)
        save("resumes", r)
    return jsonify(ok=True)


@app.route("/api/resume/<i>", methods=["GET", "DELETE"])
def resume(i):
    if request.method == "DELETE":
        S["resumes"].pop(i, None); drop("resumes", i); return jsonify(ok=True)
    if i not in S["resumes"]: return jsonify(ok=False), 404
    return jsonify(text=S["resumes"][i]["text"], skills=(S["resumes"][i]["res"] or {}).get("skills", []))


@app.route("/api/process/<i>", methods=["POST"])
def proc(i):
    r = S["resumes"].get(i)
    if not r: return jsonify(ok=False), 404
    if r["ext"] != "Extracted" or not r["text"].strip():
        r["proc_status"] = "Skipped"; save("resumes", r); return jsonify(ok=False)
    process(r); return jsonify(ok=True)


@app.route("/api/rescore", methods=["POST"])
def rescore():
    for r in S["resumes"].values():
        if r["res"]: match(r); save("resumes", r)
    return jsonify(ok=True)


@app.route("/api/decision/<i>", methods=["POST"])
def decision(i):
    r = S["resumes"].get(i); b = request.json or {}
    if not r or not r["res"]: return jsonify(ok=False), 404
    if "decision" in b:
        if b["decision"] not in ("", "Shortlisted", "Rejected", "On Hold"): return jsonify(ok=False), 400
        r["res"]["decision"] = b["decision"]
    if "notes" in b: r["res"]["notes"] = str(b["notes"])[:1000]
    match(r); save("resumes", r); return jsonify(ok=True)


@app.route("/api/jd", methods=["POST"])
def jd_save():
    b = request.json
    add_jd(b["title"], b["text"], b["required"], b["preferred"], b["min_exp"]); return rescore()


@app.route("/api/jd/<i>", methods=["DELETE"])
def jd_del(i):
    if len(S["jds"]) < 2 or i not in S["jds"]: return jsonify(ok=False, error="Keep at least one job description"), 400
    S["jds"].pop(i); drop("jds", i)
    if S["jd_id"] == i: S["jd_id"] = next(iter(S["jds"])); kv_set("jd_id", S["jd_id"])
    return rescore()


@app.route("/api/jd/select/<i>", methods=["POST"])
def jd_sel(i):
    if i in S["jds"]: S["jd_id"] = i; kv_set("jd_id", i)
    return rescore()


@app.route("/api/jd/text", methods=["POST"])
def jd_text():
    f = request.files["file"]; return jsonify(text=read(f.filename, f.read())[0])


@app.route("/api/settings", methods=["POST"])
def settings():
    b = request.json
    S["settings"].update(w_sim=int(b["w_sim"]), shortlist=int(b["shortlist"]),
                         skills=[x.strip().lower() for x in b["skills"].split(",") if x.strip()])
    kv_set("settings", S["settings"]); return rescore()


def safe(v):  # stop spreadsheet formula injection from resume-supplied text
    v = "" if v is None else v
    return "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v


@app.route("/api/report.csv")
def report():
    st = request.args.get("status", "")
    o = io.StringIO(); o.write("\ufeff"); w = csv.writer(o)
    w.writerow(["Rank", "Candidate", "Email", "Phone", "Category", "Match %", "Experience (yrs)", "Matched skills", "Missing skills", "Status", "Notes"])
    rs = sorted([r for r in S["resumes"].values() if r["res"] and (not st or r["res"].get("status") == st)],
                key=lambda r: -r["res"].get("score", 0))
    for n, r in enumerate(rs, 1):
        x = r["res"]
        w.writerow([n, safe(x["name"]), safe(x["email"]), x["phone"], x["category"], x.get("score"), x["exp"],
                    "; ".join(x.get("matched", [])), "; ".join(x.get("missing", [])), x.get("status"), safe(x.get("notes"))])
    return Response(o.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=screening_report.csv"})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "adduser":  # python app.py adduser <name> <password>
        q("REPLACE INTO users VALUES(?,?,?)", (sys.argv[2], generate_password_hash(sys.argv[3]), "Recruiter")); print("User saved.")
    else:
        app.run(debug=True)