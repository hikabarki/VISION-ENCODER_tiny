"""
Keural Vision Encoder — GUI Evaluation (Flask)
Run: python3.10 scripts/gui_eval.py
Open: http://localhost:7860
"""
import sys, os, glob, io, base64, json
sys.path.insert(0, "/home/mkd-s002/vision-encoder/poc")

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

import alignment.hf_wrapper
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = "/home/mkd-s002/vision-encoder/poc/outputs/run"
LEVEL_COLORS = {0: ("#2196F3", "Global"), 1: ("#FF9800", "Region"), 2: ("#F44336", "Detail")}
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_cache = {"ckpt": None, "model": None}
_text_enc = None

def get_model(ckpt_path):
    if _cache["ckpt"] == ckpt_path and _cache["model"] is not None:
        return _cache["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(ckpt_path, dtype=load_dtype)
    model = model.to(device).eval()
    model.encoder.pool_metadata.data[:] = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    for block in model.encoder.blocks:
        rope = block.attn.rope
        i = torch.arange(0, rope.half_dim, 2, dtype=torch.float32)
        rope.inv_freq.data.copy_(1.0 / (rope.base ** (i / rope.half_dim)))
    _cache["ckpt"] = ckpt_path
    _cache["model"] = model
    return model

def get_text_enc():
    global _text_enc
    if _text_enc is None:
        _text_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _text_enc

def list_checkpoints():
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    return [os.path.basename(c) for c in ckpts]

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def process_image(ckpt_path, pil_img, captions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = get_model(ckpt_path)

    pv = transform(pil_img).unsqueeze(0).to(device, dtype=load_dtype)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=load_dtype)
        if device == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        out = model(pv, token_budget=256)
        img_emb = F.normalize(out.pooled.float().cpu(), dim=-1)
        sal_map = model.encoder.tokenizer.saliency(model.encoder.tokenizer.stem(pv))
        spatial_meta = out.spatial_metadata[0].cpu().float().numpy()
        level_ids = out.level_ids[0].cpu().numpy()

    # Replace NaN embeddings with zeros so JSON stays valid
    if img_emb.isnan().any():
        img_emb = torch.zeros_like(img_emb)

    txt_embs = F.normalize(torch.tensor(get_text_enc().encode(captions)), dim=-1)
    sims = (img_emb @ txt_embs.T).squeeze(0).numpy()
    # Replace any NaN scores with 0.0
    sims = np.nan_to_num(sims, nan=0.0)
    sorted_idx = np.argsort(sims)[::-1]
    scores = [{"rank": int(i+1), "score": float(sims[idx]), "caption": captions[idx]}
              for i, idx in enumerate(sorted_idx)]

    counts = {lv: int((level_ids == lv).sum()) for lv in (0, 1, 2)}

    # 3-panel vertical figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 16))
    W_img, H_img = pil_img.size
    sal_np = sal_map[0, 0].float().cpu().numpy()
    sal_resized = np.array(Image.fromarray((sal_np * 255).astype(np.uint8)).resize(pil_img.size, resample=Image.BILINEAR)) / 255.0

    axes[0].imshow(pil_img); axes[0].set_title("Original Image", fontsize=12); axes[0].axis("off")

    axes[1].imshow(pil_img)
    axes[1].imshow(sal_resized, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f"Saliency Map  (mean={sal_np.mean():.3f}  std={sal_np.std():.3f})", fontsize=10)
    axes[1].axis("off")
    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(pil_img)
    axes[2].set_title(f"Token Placement  Global={counts[0]}  Region={counts[1]}  Detail={counts[2]}", fontsize=10)
    for level in (0, 1, 2):
        mask = level_ids == level
        if mask.sum() == 0: continue
        cx = spatial_meta[mask, 0] * W_img; cy = spatial_meta[mask, 1] * H_img
        tw = spatial_meta[mask, 2] * W_img; th = spatial_meta[mask, 3] * H_img
        color, label = LEVEL_COLORS[level]
        for i in range(len(cx)):
            axes[2].add_patch(patches.Rectangle((cx[i]-tw[i]/2, cy[i]-th[i]/2), tw[i], th[i],
                linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.15))
            axes[2].plot(cx[i], cy[i], ".", color=color, markersize=3, alpha=0.7)
    axes[2].legend(handles=[patches.Patch(color=c, label=f"{l} ({counts[lv]})") for lv,(c,l) in LEVEL_COLORS.items()],
                   loc="upper right", fontsize=9, framealpha=0.8)
    axes[2].axis("off")
    plt.tight_layout()

    pooled_norm = float(out.pooled.float().norm().item())
    if np.isnan(pooled_norm):
        pooled_norm = 0.0

    return {"scores": scores, "counts": counts,
            "pooled_norm": pooled_norm,
            "fig_b64": fig_to_b64(fig)}

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html>
<head>
<title>Keural Vision Encoder - Eval</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
  h1 { color: #333; }
  .layout { display: flex; gap: 20px; }
  .left { width: 380px; flex-shrink: 0; }
  .right { flex: 1; }
  .card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  label { display: block; font-weight: bold; margin-bottom: 6px; color: #555; }
  select, textarea { width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
  textarea { resize: vertical; }
  input[type=file] { width: 100%; padding: 6px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
  button { width: 100%; padding: 12px; background: #1976D2; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; margin-top: 8px; }
  button:hover { background: #1565C0; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .result-block { margin-bottom: 24px; border-top: 2px solid #1976D2; padding-top: 12px; }
  .result-block h3 { color: #1976D2; margin: 0 0 10px; }
  .scores table { width: 100%; border-collapse: collapse; }
  .scores td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 14px; }
  .scores tr:first-child td { font-weight: bold; background: #e3f2fd; }
  .bar { display: inline-block; background: #1976D2; height: 12px; min-width: 2px; border-radius: 2px; }
  .stats { font-size: 13px; color: #555; margin-top: 8px; }
  img.viz { width: 100%; border-radius: 4px; }
  #status { color: #666; font-style: italic; margin-top: 8px; }
  .spinner { display: none; color: #1976D2; font-weight: bold; margin-top: 8px; }
  .summary-box { background: white; border-radius: 8px; padding: 16px; margin-top: 32px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); border-top: 3px solid #1976D2; }
  .summary-box h2 { color: #1976D2; margin: 0 0 12px; font-size: 15px; letter-spacing: 0.5px; text-transform: uppercase; }
  .summary-box table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .summary-box th { background: #e3f2fd; padding: 7px 10px; text-align: left; font-weight: bold; color: #333; border-bottom: 2px solid #1976D2; }
  .summary-box td { padding: 6px 10px; border-bottom: 1px solid #eee; }
  .summary-box tr:hover td { background: #f5f5f5; }
  .rank1 { color: #2e7d32; font-weight: bold; }
  .summary-footer { font-size: 12px; color: #888; margin-top: 8px; }
</style>
</head>
<body>
<h1>Keural Vision Encoder - Checkpoint Evaluator</h1>
<div class="layout">
  <div class="left">
    <div class="card">
      <label>Checkpoint</label>
      <select id="ckpt">
        {% for c in checkpoints %}<option value="{{c}}" {% if loop.last %}selected{% endif %}>{{c}}</option>{% endfor %}
      </select>
      <button onclick="refreshCkpts()" style="margin-top:6px;padding:6px;font-size:13px;">Refresh Checkpoints</button>
    </div>
    <div class="card">
      <label>Upload Images (select multiple)</label>
      <input type="file" id="images" multiple accept="image/*">
    </div>
    <div class="card">
      <label>Captions to Compare (one per line)</label>
      <textarea id="captions" rows="8">a cat sitting on a sofa
a dog running in a park
a horse in a field
a red car on a street
a person walking</textarea>
    </div>
    <button id="runBtn" onclick="runEval()">Run Eval</button>
    <div class="spinner" id="spinner">Processing... please wait</div>
    <div id="status"></div>
    <div id="summary-left"></div>
  </div>
  <div class="right" id="results"></div>
</div>
<script>
let allResults = [];

async function refreshCkpts() {
  const r = await fetch('/checkpoints');
  const data = await r.json();
  const sel = document.getElementById('ckpt');
  sel.innerHTML = data.map((c,i) => `<option value="${c}" ${i===data.length-1?'selected':''}>${c}</option>`).join('');
}

async function runEval() {
  const files = document.getElementById('images').files;
  if (!files.length) { alert('Please select at least one image.'); return; }
  const captions = document.getElementById('captions').value.trim().split('\\n').filter(x=>x.trim());
  if (!captions.length) { alert('Please enter at least one caption.'); return; }
  const ckpt = document.getElementById('ckpt').value;

  document.getElementById('runBtn').disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('status').textContent = '';
  document.getElementById('results').innerHTML = '';
  document.getElementById('summary-left').innerHTML = '';
  allResults = [];

  for (const file of files) {
    const fd = new FormData();
    fd.append('image', file);
    fd.append('captions', JSON.stringify(captions));
    fd.append('checkpoint', ckpt);
    try {
      const r = await fetch('/eval', {method:'POST', body: fd});
      const data = await r.json();
      if (data.error) { appendResult(file.name, null, data.error); continue; }
      appendResult(file.name, data);
    } catch(e) {
      appendResult(file.name, null, e.toString());
    }
  }

  document.getElementById('runBtn').disabled = false;
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('status').textContent = `Done - ${files.length} image(s) processed`;
  if (allResults.length > 1) renderSummary();
}

function renderSummary() {
  const avgNorm = (allResults.reduce((s,r) => s + r.pooled_norm, 0) / allResults.length).toFixed(4);
  const rows = allResults.map(r => {
    const top = r.scores[0];
    const fname = r.name.length > 18 ? r.name.slice(0,16) + '…' : r.name;
    return `<tr>
      <td style="font-size:12px;color:#555;">${fname}</td>
      <td class="rank1" style="font-size:12px;">${top.caption}</td>
      <td style="font-size:12px;white-space:nowrap;">${top.score > 0 ? '+' : ''}${top.score.toFixed(4)}</td>
      <td style="font-size:12px;">${r.pooled_norm.toFixed(3)}</td>
    </tr>`;
  }).join('');

  document.getElementById('summary-left').innerHTML = `
    <div class="summary-box">
      <h2>All Results Summary</h2>
      <table>
        <tr><th>Image</th><th>Top Caption</th><th>Score</th><th>Norm</th></tr>
        ${rows}
      </table>
      <div class="summary-footer">
        ${allResults.length} images &nbsp;|&nbsp; Avg norm: ${avgNorm}
      </div>
    </div>`;
}

function appendResult(name, data, err) {
  const div = document.createElement('div');
  div.className = 'result-block';
  if (err) {
    div.innerHTML = `<h3>${name}</h3><p style="color:red">${err}</p>`;
  } else {
    allResults.push({name, ...data});
    const maxScore = Math.max(...data.scores.map(s=>s.score));
    const minScore = Math.min(...data.scores.map(s=>s.score));
    const range = maxScore - minScore || 1;
    const rows = data.scores.map(s => {
      const barW = Math.max(4, Math.round(((s.score - minScore)/range)*120));
      return `<tr><td>#${s.rank}</td><td>${s.score>0?'+':''}${s.score.toFixed(4)}</td>
              <td><span class="bar" style="width:${barW}px"></span></td>
              <td>${s.caption}</td></tr>`;
    }).join('');
    div.innerHTML = `
      <h3>${name}</h3>
      <div style="display:flex;gap:16px;align-items:flex-start;">
        <div style="flex:0 0 340px;">
          <div class="scores"><table><tr><td>Rank</td><td>Score</td><td>Bar</td><td>Caption</td></tr>${rows}</table></div>
          <div class="stats">
            Global: ${data.counts[0]} tokens &nbsp;|&nbsp; Region: ${data.counts[1]} tokens &nbsp;|&nbsp; Detail: ${data.counts[2]} tokens<br>
            Pooled norm: ${data.pooled_norm.toFixed(4)}
          </div>
        </div>
        <div style="flex:1;">
          <img class="viz" src="data:image/png;base64,${data.fig_b64}">
        </div>
      </div>`;
  }
  document.getElementById('results').appendChild(div);
}
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML, checkpoints=list_checkpoints())

@app.route("/checkpoints")
def checkpoints():
    return jsonify(list_checkpoints())

@app.route("/eval", methods=["POST"])
def eval_image():
    try:
        f = request.files["image"]
        captions = json.loads(request.form["captions"])
        ckpt_name = request.form["checkpoint"]
        ckpt_path = os.path.join(OUTPUT_DIR, ckpt_name)
        pil_img = Image.open(f.stream).convert("RGB")
        result = process_image(ckpt_path, pil_img, captions)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": f"{e}\n{traceback.format_exc()}"})

if __name__ == "__main__":
    print("Starting Keural Eval GUI at http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=False)
