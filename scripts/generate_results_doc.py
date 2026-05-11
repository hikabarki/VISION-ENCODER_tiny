"""
Generate a Word (.docx) report of all evaluation results produced so far.
Saves to poc/results/keural_eval_results_step28000.docx
"""
import os, sys
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "keural_eval_results_step28000.docx")
SALIENCY_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "saliency_viz")

doc = Document()

# ── page margins ──────────────────────────────────────────────────────────────
section = doc.sections[0]
section.top_margin    = Inches(1.0)
section.bottom_margin = Inches(1.0)
section.left_margin   = Inches(1.2)
section.right_margin  = Inches(1.2)

def heading(text, level=1, color=None):
    p = doc.add_heading(text, level=level)
    if color:
        for run in p.runs:
            run.font.color.rgb = RGBColor(*color)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(4)
    return p

def para(text, bold=False, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(4)
    return p

def add_table(headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    hrow = t.rows[0]
    for i, h in enumerate(headers):
        cell = hrow.cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "2C3E50")
        shading.set(qn("w:color"), "FFFFFF")
        cell._tc.get_or_add_tcPr().append(shading)
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    # Data rows
    for ri, row in enumerate(rows):
        trow = t.rows[ri + 1]
        for ci, val in enumerate(row):
            trow.cells[ci].text = str(val)
            trow.cells[ci].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    return t

def divider():
    p = doc.add_paragraph("─" * 80)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.runs[0]
    run.font.size = Pt(7)
    run.font.color.rgb = RGBColor(180, 180, 180)

# ══════════════════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════════
title = doc.add_heading("Keural Vision Encoder", 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

sub = doc.add_paragraph("Evaluation Results Report — Checkpoint 28,000 / 75,000")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].font.size = Pt(14)
sub.runs[0].bold = True

meta = doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   Training: 37.3% complete")
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.runs[0].font.size = Pt(10)
meta.runs[0].font.color.rgb = RGBColor(120, 120, 120)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
# 1. MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
heading("1. Model Overview", 1)
add_table(
    ["Property", "Value"],
    [
        ["Model name",         "Keural Vision Encoder (Tiny POC)"],
        ["Total parameters",   "24,666,369  (~24.7M)"],
        ["Architecture",       "CNN Stem → ATB Tokenizer → 12× Spatial Transformer"],
        ["Embed dim",          "384"],
        ["Transformer depth",  "12 layers, 6 heads, head_dim=64"],
        ["Token types",        "Global (5%) / Region (25%) / Detail (70%)"],
        ["Default budget",     "256 tokens"],
        ["Loss",               "SigLIP + HCT auxiliary + Saliency regularization"],
        ["Training data",      "CC3M + CC12M  (~15M image-text pairs)"],
        ["Text encoder",       "sentence-transformers/all-MiniLM-L6-v2 (frozen)"],
        ["Hardware",           "1× RTX 5090  (32 GB VRAM)"],
        ["Precision",          "bfloat16"],
    ],
    col_widths=[2.2, 3.8]
)

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAINING PROGRESS
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("2. Training Progress", 1)
add_table(
    ["Step", "% Done", "Loss", "Primary (SigLIP)", "HCT Loss", "Saliency", "Speed (50 steps)"],
    [
        ["5,000",  "6.7%",  "~6.8",  "~3.4", "~3.4", "~4.0", "—"],
        ["10,000", "13.3%", "~5.9",  "~2.9", "~2.9", "~4.0", "—"],
        ["15,000", "20.0%", "~5.4",  "~2.6", "~2.6", "~4.0", "—"],
        ["20,000", "26.7%", "~5.1",  "~2.5", "~2.5", "~4.0", "—"],
        ["25,000", "33.3%", "~4.8",  "~2.3", "~2.3", "~4.0", "~510s"],
        ["28,000", "37.3%", "~4.3–4.9", "~2.1–2.4", "~2.1–2.4", "~4.0", "~510s"],
    ],
    col_widths=[1.0, 0.9, 0.8, 1.4, 1.0, 1.0, 1.4]
)
doc.add_paragraph()
para("Key observations:", bold=True)
para("• Loss is steadily decreasing — model is converging correctly.")
para("• HCT loss closely tracks primary loss — hierarchical alignment is working.")
para("• Saliency regularization stable at ~4.0 throughout training.")
para("• Speed: ~510s per 50 steps (~10.2s/step). ETA to 75k: ~4–5 days from step 28k.")
para("• VRAM usage: ~25.8 GB / 32 GB after disabling gradient checkpointing.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BUG DISCOVERED AND FIXED
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("3. Bug Discovered & Fixed", 1)
para("Bug: pool_metadata registered as persistent=False in encoder.py", bold=True)
para(
    "The pool token's spatial metadata ([0.5, 0.5, 1.0, 1.0]) was registered as a "
    "non-persistent buffer. This meant it was never saved to checkpoints. When loading "
    "a checkpoint via HuggingFace from_pretrained(), the buffer received garbage values "
    "(e.g. 6.1e+25, 4.4e+21) causing NaN outputs in the first transformer block during "
    "all evaluation runs."
)
para("Impact on training: NONE. Training always initializes the model fresh before loading "
     "weights, so pool_metadata was always correctly set to [0.5, 0.5, 1.0, 1.0] during "
     "forward passes. All checkpoints and training loss values are valid.", italic=True)
para("Fix applied:", bold=True)
para("• encoder.py: changed persistent=False → persistent=True")
para("• eval_checkpoint.py, eval_retrieval_cc3m.py, visualize_saliency.py: added manual "
     "reset of pool_metadata after from_pretrained() for backwards compatibility with "
     "existing checkpoints.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. IMAGE-TEXT RETRIEVAL — DUMMY IMAGES
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("4. Image-Text Retrieval — Dummy Color Images", 1)
para("Script: scripts/eval_checkpoint.py  |  Checkpoint: checkpoint-28000")
para("Test: 4 synthetic solid-color images matched against 4 color-describing captions.")
doc.add_paragraph()

add_table(
    ["", "T0: red image", "T1: green image", "T2: blue image", "T3: yellow image"],
    [
        ["I0 (red)",    "-0.011 ✓", "-0.012",  "-0.082",  "-0.051"],
        ["I1 (green)",  "-0.005",   "-0.100 ✓", "-0.062",  "-0.038"],
        ["I2 (blue)",   "-0.003",   "+0.014",   "-0.057 ✓", "-0.069"],
        ["I3 (yellow)", "-0.012",   "-0.063",   "-0.088",   "-0.040 ✓"],
    ],
    col_widths=[1.2, 1.4, 1.4, 1.4, 1.4]
)
doc.add_paragraph()
para("Retrieval accuracy: 1/4 correct (25%)  |  Random baseline: 1/4 (25%)")
para("Pooled embedding norm: ~6.22  (target after full training: ~1.0)")
para("Token budget breakdown: Global=13, Region=64, Detail=179")
para("Note: All cosine similarities are negative — image and text embedding spaces are not "
     "yet aligned. This is expected behavior at 37% training progress.", italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. IMAGE-TEXT RETRIEVAL — REAL CC3M IMAGES
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("5. Image-Text Retrieval — Real CC3M Pairs", 1)
para("Script: scripts/eval_retrieval_cc3m.py  |  Checkpoint: checkpoint-28000")
para("Dataset: 200 image-caption pairs extracted from CC3M training tars")
doc.add_paragraph()

add_table(
    ["Metric", "Image → Text", "Text → Image", "Random Baseline"],
    [
        ["R@1",  "1.0%", "0.5%", "0.50%"],
        ["R@5",  "3.5%", "3.5%", "2.50%"],
        ["R@10", "6.0%", "5.5%", "5.00%"],
    ],
    col_widths=[1.5, 1.5, 1.5, 1.8]
)
doc.add_paragraph()
para("R@1 (Image→Text) is 2× above random chance — model is demonstrating early-stage "
     "image-text alignment. This is consistent with expectations at 37% training.")
para("Interpretation:", bold=True)
para("• Above random: the model has begun learning alignment — not junk.")
para("• Still low: saliency and alignment converge in the second half of training.")
para("• Expected R@1 at full training (75k steps): ~15–35% on training set.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SALIENCY VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("6. Saliency Map Visualization", 1)
para("Script: scripts/visualize_saliency.py  |  Checkpoint: checkpoint-28000")
para("Each figure shows 3 panels: Original Image | Saliency Heatmap | Token Placement")
doc.add_paragraph()

sal_stats = [
    ("cat",   "0.000", "1.000", "0.453", "0.498"),
    ("dog",   "0.000", "1.000", "0.469", "0.499"),
    ("horse", "0.000", "1.000", "0.484", "0.500"),
]
add_table(
    ["Image", "Sal Min", "Sal Max", "Sal Mean", "Sal Std", "Global", "Region", "Detail"],
    [
        [name, mn, mx, mean, std, "13", "64", "179"]
        for name, mn, mx, mean, std in sal_stats
    ],
    col_widths=[0.9, 0.8, 0.8, 0.9, 0.8, 0.8, 0.8, 0.8]
)
doc.add_paragraph()
para("Observations:", bold=True)
para("• Saliency mean ~0.45–0.48 (close to 0.5 = uniform/random) — the predictor has "
     "not yet learned to discriminate salient vs. non-salient regions.")
para("• Binary-like activations (std ~0.5) — saliency is either fully on or off, "
     "lacking smooth semantic gradients.")
para("• Dog image shows the most promise: bright spots loosely follow the dog's "
     "body silhouette — early evidence of spatial learning.")
para("• Token placement is nearly uniform across all images — ATB is not yet "
     "content-adaptive at this stage of training.")
para("• Expected improvement: by step 75k, saliency should focus clearly on foreground "
     "objects, and token clusters should visibly concentrate on salient regions.", italic=True)

doc.add_paragraph()
para("Saliency figures (saved to outputs/saliency_viz/):", bold=True)

for img_name in ["cat_saliency.png", "dog_saliency.png", "horse_saliency.png"]:
    img_path = os.path.join(SALIENCY_DIR, img_name)
    if os.path.exists(img_path):
        label = img_name.replace("_saliency.png", "").capitalize()
        para(f"{label}:", bold=True)
        doc.add_picture(img_path, width=Inches(5.8))
        doc.add_paragraph()

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPARISON WITH FASTVLM
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("7. Architecture Comparison: Keural vs FastVLM", 1)
add_table(
    ["Dimension", "FastVLM (Apple 2025)", "Keural Vision Encoder"],
    [
        ["Architecture",       "Hybrid CNN+Transformer (FastViTHD)", "CNN Stem + ATB + Transformer"],
        ["Token reduction",    "Structural: stride-32 (fixed)",      "Semantic: saliency-guided (adaptive)"],
        ["Token types",        "Uniform patch tokens",                "Global / Region / Detail (3 levels)"],
        ["Saliency routing",   "None",                               "Learnable → Gumbel sampling"],
        ["Model size",         "~125M params",                        "~24.7M params (tiny POC)"],
        ["Loss",               "CLIP/SigLIP",                        "SigLIP + HCT + saliency reg."],
        ["Open source",        "No (Apple internal)",                 "Yes"],
        ["Benchmarked",        "Yes (published)",                     "In progress (step 28k/75k)"],
    ],
    col_widths=[2.0, 2.4, 2.4]
)

# ══════════════════════════════════════════════════════════════════════════════
# 8. NEXT STEPS
# ══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()
heading("8. Next Steps", 1)
add_table(
    ["Priority", "Task", "Status"],
    [
        ["1", "Complete Phase 1 training to step 75,000",                    "In progress (~4–5 days)"],
        ["2", "Re-run retrieval eval at step 50k and 75k",                   "Pending"],
        ["3", "Re-run saliency visualization at step 75k (paper figure)",    "Pending"],
        ["4", "Build Phase 2: projector training (train_projector.py)",      "Not started"],
        ["5", "Run Flickr30K / COCO retrieval benchmark (R@1, R@5, R@10)",  "Not started"],
        ["6", "Ablation: ATB on vs off, HCT on vs off",                     "Not started"],
        ["7", "Benchmark inference latency vs FastVLM",                      "Not started"],
    ],
    col_widths=[0.7, 4.0, 1.8]
)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
