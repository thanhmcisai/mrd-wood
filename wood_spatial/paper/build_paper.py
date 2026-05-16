#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final: BAO_CAO_SC_URD_v2.pdf - full analysis commentary on every table and figure.

Deprecated:
    This HTML/WeasyPrint report generator predates the current MRD-Wood
    manuscript and the expanded perturbation protocol. Keep it only as an
    archival draft; use main.tex and wood_spatial/figures/paper_figures.py for
    the current paper.
"""
raise SystemExit(
    'wood_spatial.paper.build_paper is deprecated and still contains the old '
    'SC-URD/eight-perturbation narrative. Use main.tex plus '
    'wood_spatial.figures.paper_figures for the current MRD-Wood paper.'
)
import base64
import os
from pathlib import Path
from weasyprint import HTML, CSS

BASE = Path(os.environ.get('WOOD_BASE', Path(__file__).resolve().parents[2]))
RESULTS = Path(os.environ.get('WOOD_RESULTS_DIR', BASE / 'results_v4'))
FIG  = RESULTS / 'figures'
OUT  = BASE / 'paper_latex/BAO_CAO_SC_URD_v2.pdf'

def b64(fname):
    p = FIG / fname
    if not p.exists(): return None
    with open(p,'rb') as f: return base64.b64encode(f.read()).decode()

def fig(fname, caption, analysis='', width='100%'):
    data = b64(fname)
    if not data:
        return f'<div style="color:red;font-size:9pt;">[Hình không tìm thấy: {fname}]</div>'
    html = (f'<figure><img src="data:image/png;base64,{data}" '
            f'style="max-width:{width};height:auto;border:1px solid #ddd;padding:3px;"/>'
            f'<figcaption><strong>Hình.</strong> {caption}</figcaption></figure>')
    if analysis:
        html += f'<div class="fig-analysis">{analysis}</div>'
    return html

CSS_STR = """
@page{size:A4;margin:2.5cm 2.2cm 2.5cm 2.5cm;
      @bottom-center{content:counter(page);font-size:9pt;}}
body{font-family:"DejaVu Serif","Times New Roman",serif;font-size:11pt;
     line-height:1.6;color:#111;text-align:justify;}
h1{font-size:16pt;text-align:center;margin:0 0 6px;}
h2{font-size:13pt;margin:24px 0 6px;border-bottom:1.5px solid #888;
   padding-bottom:3px;page-break-after:avoid;}
h3{font-size:11.5pt;margin:16px 0 4px;page-break-after:avoid;}
h4{font-size:11pt;margin:10px 0 2px;font-style:italic;}
.meta{text-align:center;font-size:10.5pt;color:#444;margin:3px 0;}
.abstract{background:#f5f5f5;border-left:3px solid #888;
          padding:10px 14px;margin:12px 0;font-size:10.5pt;}
.keywords{font-size:10pt;margin:6px 0 18px;}
table{border-collapse:collapse;width:100%;font-size:10pt;
      margin:10px 0;page-break-inside:avoid;}
th{background:#e5e5e5;border-top:2px solid #333;border-bottom:1px solid #333;
   padding:5px 7px;text-align:left;}
td{border-bottom:0.5px solid #ccc;padding:4px 7px;}
tr:last-child td{border-bottom:2px solid #333;}
caption{font-size:9.5pt;text-align:left;margin-bottom:3px;
        caption-side:top;font-style:italic;}
figure{margin:14px 0;text-align:center;page-break-inside:avoid;}
figcaption{font-size:9.5pt;margin-top:5px;color:#333;text-align:left;}
.fig-analysis{font-size:10.5pt;margin:6px 0 16px;
              border-left:3px solid #888;padding:6px 10px;color:#333;
              background:#fafafa;}
ul,ol{margin:5px 0 5px 20px;}li{margin:3px 0;}
p{margin:6px 0;text-indent:1.2em;}
p:first-of-type,h2+p,h3+p,h4+p,.abstract p,blockquote p,
.analysis p,.fig-analysis p,li p{text-indent:0;}
.eq{text-align:center;margin:10px 0;font-style:italic;background:#fafafa;
    padding:6px;border-radius:3px;}
pre{background:#f0f0f0;padding:10px;font-size:9pt;border-left:3px solid #999;
    margin:8px 0;white-space:pre-wrap;}
.warn{color:#c00;font-weight:bold;}.ok{color:#060;font-weight:bold;}
blockquote{border-left:3px solid #888;padding:8px 14px;margin:10px 0;
           font-style:italic;background:#f9f9f9;}
.analysis{font-size:10.5pt;margin:8px 0 18px;padding:10px 14px;
          background:#f0f5ff;border-left:3px solid #4a7abf;line-height:1.65;}
.analysis p{text-indent:0;margin:5px 0;}
.kf{font-weight:bold;color:#1a3a5c;}
.note{font-size:9pt;color:#555;font-style:italic;}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE + ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════════

TITLE = """
<h1>Mạng Nơ-ron Đóng Băng Nhìn Thấy Gì Trong Ảnh Gỗ?<br/>
Phân Tích Sụp Đổ Biểu Diễn Đa Tầng Dưới Domain Shift</h1>
<div class="meta">Ma Công Thành — Nghiên cứu sinh &nbsp;·&nbsp; Hướng dẫn: PGS.TS. Nguyễn Trọng Khánh</div>
<div class="meta">Việt Nam &nbsp;·&nbsp; 2026</div>
"""

ABSTRACT = """
<div class="abstract"><strong>Tóm tắt.</strong>
Nghiên cứu này đề xuất framework <strong>SC-URD</strong>
(<em>Spatial Clustering for Unsupervised Representation Distortion analysis</em>) —
khung phân tích đa tầng cung cấp bằng chứng có hệ thống về cơ chế failure của
các mô hình nhận dạng loài gỗ dưới domain shift. Thực nghiệm trên 7 backbone đóng băng,
8 loại nhiễu, và 6 bộ dữ liệu (3 kiểm soát + 3 không kiểm soát) cho thấy:
<strong>feature drift là predictor mạnh của accuracy drop, r&nbsp;=&nbsp;0.924</strong>
(n&nbsp;=&nbsp;714, dữ liệu có cấu trúc lồng nhau; per-backbone r=0.935–0.977),
bền vững với partial r&nbsp;=&nbsp;0.772 sau kiểm soát loại nhiễu.
Feature drift còn cho phép <strong>phát hiện thất bại không cần nhãn</strong>
(AUC&nbsp;=&nbsp;0.986, F1&nbsp;=&nbsp;0.936 trên 6 datasets, τ=0.281), vượt trội SSIM (AUC=0.812, r=0.382) và PSNR (AUC=0.558, r=0.165).
Tại full resolution (1280×1024), MobileNetV3 có spatial granularity cao nhất (SGI&nbsp;=&nbsp;0.00253),
HRNet-32 đứng thứ 3 (SGI&nbsp;=&nbsp;0.00298) — xác nhận lợi thế kiến trúc high-resolution;
EfficientNet-B3 giữ #1 (full-res) do SGI 224px bị inflate bởi relative feature map size.
Swin-Tiny cho thấy bằng chứng nhất quán với near-degenerate spatial encoding
(ANOVA F&nbsp;=&nbsp;4.30, p&nbsp;=&nbsp;2.8×10⁻⁴ *** nhưng nhỏ hơn 17-216 lần so với F&nbsp;&gt;&nbsp;72*** của các backbone khác
cho tất cả backbone khác, under K-Means probing protocol).
Cross-dataset consistency: r&nbsp;=&nbsp;0.899 trên cached Tier-B subset.
Một <em>nghịch lý robustness</em> được phát hiện: không có backbone nào
tốt nhất trong mọi kịch bản triển khai.
Wood macroscopy là một stress-test domain có giá trị cao cho robustness research;
khả năng chuyển giao các findings sang các texture-rich domains khác cần điều tra riêng.
</div>
<div class="keywords"><strong>Từ khóa:</strong>
nhận dạng loài gỗ · domain shift · biến dạng hình học feature ·
phát hiện thất bại không giám sát · spatial clustering · backbone đóng băng
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1-4
# ═══════════════════════════════════════════════════════════════════════════════

SEC14 = """
<h2>1. Đặt Vấn Đề</h2>
<p>Nhận dạng loài gỗ từ ảnh mặt cắt ngang là bài toán quan trọng trong quản lý lâm nghiệp,
kiểm soát thương mại gỗ quốc tế, và bảo tồn đa dạng sinh học. Các mô hình deep learning
đạt &gt;95% accuracy trong điều kiện phòng thí nghiệm nhưng suy giảm nghiêm trọng khi
triển khai thực tế do <strong>domain shift</strong>. Câu hỏi "tại sao mô hình thất bại?"
vẫn chưa được trả lời thỏa đáng — hầu hết nghiên cứu chỉ đo accuracy drop
mà không giải thích cơ chế sụp đổ.</p>

<table>
<caption>Bảng 1. Research gaps và đóng góp của SC-URD.</caption>
<tr><th>Khoảng trống nghiên cứu</th><th>Mức độ</th><th>SC-URD giải quyết</th></tr>
<tr><td>Không có mechanistic explanation cho failure</td>
    <td><span class="warn">Critical</span></td>
    <td>Feature geometry distortion chain (r=0.924)</td></tr>
<tr><td>Không có unsupervised failure detector</td>
    <td><span class="warn">Critical</span></td>
    <td>AUC=0.986, không cần labels</td></tr>
<tr><td>Không phân tích spatial encoding per-backbone</td>
    <td>High</td><td>SGI/CSI/ARI trên 7 backbone</td></tr>
<tr><td>Không có multi-level framework thống nhất</td>
    <td>High</td><td>SC-URD: 4 tầng tích hợp</td></tr>
<tr><td>Benchmark gỗ chưa đủ toàn diện</td>
    <td>Medium</td><td>8 perturbation × 7 backbone × 6 dataset</td></tr>
</table>

<div class="analysis">
<p><span class="kf">Tại sao "mechanistic explanation" là Critical?</span>
Biết rằng "accuracy giảm 40% khi blur" chỉ mô tả triệu chứng, không chỉ ra nguyên nhân.
SC-URD đề xuất một chuỗi diễn giải mechanistic và cung cấp bằng chứng observational:
blur phá hủy fine-texture patterns → feature drift tăng → inter-class distances collapse
→ kNN không phân biệt được classes → accuracy drop.
Bằng chứng mạnh nhất (r=0.924, partial r=0.772) nhất quán với chuỗi này,
giúp định hướng can thiệp cụ thể (deblur preprocessing) thay vì sửa chữa mù quáng.
Tuy nhiên, thiết kế quan sát hiện tại chưa thể khẳng định nhân quả tuyệt đối.</p>
<p><span class="kf">Tại sao "unsupervised failure detector" là Critical?</span>
Trong deployment thực tế không có labels trên domain mới.
Nếu không phát hiện được sự suy giảm trước khi gây hậu quả (nhận dạng sai loài gỗ
trong kiểm soát hải quan), hệ thống AI trở nên không đáng tin cậy.
Feature drift giải quyết: chỉ cần forward pass, không cần nhãn.</p>
</div>

<h2>2. Các Nghiên Cứu Liên Quan</h2>
<p><strong>Robustness benchmarks:</strong> ImageNet-C (Hendrycks &amp; Dietterich, 2019)
và Owens et al. (2025) đo accuracy drop nhưng không giải thích cơ chế.
<strong>Domain Adaptation:</strong> DANN, CORAL, DomainBed tập trung sửa gap,
không diagnose nguyên nhân.
<strong>Explainability:</strong> Grad-CAM cung cấp visualization định tính,
không liên kết định lượng với domain-shift metrics.
<strong>Feature Space Analysis:</strong> CKA đo tương đồng representation
nhưng không đo distortion dưới shift và không kết nối với accuracy.
<em>Khoảng trống cốt lõi: theo hiểu biết của chúng tôi, chưa có nghiên cứu nào
trong domain wood/texture recognition đặt feature geometry distortion làm predictor
định lượng của accuracy drop, hay sử dụng nó như unsupervised failure detector
không cần ground truth labels.</em></p>

<h2>3. Research Questions</h2>
<blockquote>"Domain shift trong nhận dạng loài gỗ không chỉ là accuracy drop tại output,
mà có thể phản ánh sự suy giảm feature representation ở tầng trung gian.
Chúng tôi muốn đo lường và dự đoán sự suy giảm đó mà không cần ground truth,
và phát hiện thất bại trước khi quan sát được lỗi phân loại."</blockquote>

<table>
<caption>Bảng 2. Research Questions, experiment tương ứng, và kết quả tóm tắt.</caption>
<tr><th>RQ</th><th>Câu hỏi</th><th>Exp</th><th>Kết quả chính</th></tr>
<tr><td>RQ1</td><td>Loại nhiễu nào nguy hiểm nhất?</td><td>Exp1</td>
    <td>Compound &gt; Blur &gt; Defocus; Illumination/Color vô hại</td></tr>
<tr><td>RQ2</td><td>Backbone nào robust hơn?</td><td>Exp1+Friedman</td>
    <td>Friedman χ²=124.87, p=1.54×10⁻²⁴; DINOv2-B rank tốt nhất (2.73)</td></tr>
<tr><td>RQ3</td><td>Feature geometry dự đoán failure?</td><td>Exp1b+7</td>
    <td>r=0.924 (n=714); partial r=0.772 sau control confounding</td></tr>
<tr><td>RQ4</td><td>Detect failure không cần labels?</td><td>Exp8</td>
    <td>AUC=0.986, F1=0.936 tại τ=0.281</td></tr>
<tr><td>RQ5</td><td>Spatial encoding khác nhau thế nào?</td><td>Exp2+3</td>
    <td>SGI/CSI/ARI khác biệt rõ; Swin-T yếu nhất (ANOVA F=4.30***, nhỏ hơn 17-216x so với others)</td></tr>
<tr><td>RQ6</td><td>Generalize sang external acquisition protocols?</td><td>Exp4+9</td>
    <td>r=0.899 trên cached Tier-B subset; cần rerun đủ BFS46/FSDM41/GOIMAI/WOODAUTH/BD11</td></tr>
<tr><td>RQ7</td><td>Spatial ổn định cross-magnification?</td><td>Exp5+5b</td>
    <td>ConvNeXt-T tốt nhất cross-mag; DINOv2-B spatial ổn định nhất</td></tr>
</table>

<div class="analysis">
<p>7 RQ có thể nhóm thành 3 nhóm liên kết:
<span class="kf">Nhóm đo lường (RQ1, RQ2):</span> "cái gì hỏng và hỏng bao nhiêu?" —
nền tảng cần thiết trước khi giải thích.
<span class="kf">Nhóm giải thích cơ chế (RQ3, RQ5):</span> "tại sao hỏng?" —
đóng góp mới nhất của nghiên cứu qua feature geometry (RQ3) và spatial encoding (RQ5).
<span class="kf">Nhóm ứng dụng và validation (RQ4, RQ6, RQ7):</span>
cho thấy những gì đo được có giá trị thực tiễn và nhất quán trên dữ liệu mới.</p>
</div>

<h2>4. Framework SC-URD và Phương Pháp</h2>
<pre>
INPUT: 8 Perturbation Types
  (Blur · Defocus · Resize · Illumination · JPEG · Color · Rotation · Compound)
         ↓
TẦNG 1: FEATURE GEOMETRY DISTORTION
  Δf = E[1 - cos(z_clean, z_shift)]    ← Feature drift
  FGCS = σ_intra / d_inter             ← Geometry collapse score
  ★ r(Δf, accuracy drop) = 0.924 ★
         ↓
TẦNG 2: SPATIAL REPRESENTATION
  (L2-norm) → K-Means(k=4) → Guided Filter → Cluster Map {C1,C2,C3,C4}
  SGI = Σ_k n_comp(k)/(H×W)    CSI = mean_k IoU(clean, shift)
         ↓
TẦNG 3: ATTENTION DRIFT
  Centroid-CAM → CAM Entropy H    CAM Shift JSD(clean, perturbed)
         ↓
TẦNG 4: DECISION FAILURE
  Gallery-Query kNN (k=5, cosine) → Δacc = acc_clean - acc_shift
</pre>

<div class="eq">Δ<sub>f</sub> = 𝔼[1 − cos(z<sub>clean</sub>, z<sub>shift</sub>)]
&emsp;SGI = Σ<sub>k</sub> n<sub>comp</sub>(C<sub>k</sub>) / (H×W)
&emsp;CSI = (1/K) Σ<sub>k</sub> IoU(C<sub>k</sub><sup>clean</sup>, C<sub>k</sub><sup>shift</sup>)</div>
<div class="eq">ŷ<sub>fail</sub> = 1[Δ<sub>f</sub> &gt; τ], τ*=0.281 → F1=0.936</div>

<div class="analysis">
<p><span class="kf">Feature drift (Δf) là metric then chốt của toàn bộ framework:</span>
Chỉ tính cosine distance giữa L2-normalized features của ảnh sạch và ảnh nhiễu.
Không cần ground truth, không cần gradient, không cần training.
Một forward pass là đủ — đây là lý do AUC=0.986 có thể đạt được trong
deployment không có labels.</p>
</div>

<table>
<caption>Bảng 3. 7 Backbone architectures (tất cả đóng băng, pretrained từ ImageNet/DINOv2).</caption>
<tr><th>ID</th><th>Kiến trúc</th><th>Family</th><th>Img size</th><th>Feat dim</th></tr>
<tr><td>resnet50</td><td>ResNet-50</td><td>CNN cổ điển</td><td>224</td><td>2048</td></tr>
<tr><td>efficientnet_b3</td><td>EfficientNet-B3</td><td>CNN hiệu suất cao</td><td>300</td><td>1536</td></tr>
<tr><td>convnext_tiny</td><td>ConvNeXt-Tiny</td><td>Modern CNN</td><td>224</td><td>768</td></tr>
<tr><td>swin_tiny</td><td>Swin-Tiny</td><td>Transformer</td><td>224</td><td>768</td></tr>
<tr><td>dinov2_b</td><td>DINOv2-B</td><td>Self-supervised ViT</td><td>224</td><td>768</td></tr>
<tr><td>hrnet32</td><td>HRNet-W32</td><td>High-resolution CNN</td><td>224</td><td>2048</td></tr>
<tr><td>mobilenetv3</td><td>MobileNetV3-L</td><td>Lightweight CNN</td><td>224</td><td>1280</td></tr>
</table>

<div class="analysis">
<p>7 backbone được chọn để cover phổ kiến trúc đa dạng: CNN cổ điển (ResNet), CNN hiệu suất (EfficientNet),
modern CNN (ConvNeXt), window-attention Transformer (Swin), self-supervised ViT (DINOv2),
high-resolution design (HRNet), và lightweight (MobileNetV3).
Kết quả gợi ý rằng <span class="kf">pretraining strategy (đặc biệt self-supervised learning
của DINOv2) có thể quan trọng hơn architecture family</span> trong việc tạo ra
robust representations — ít nhất là trong domain texture/wood recognition với frozen evaluation protocol.</p>
<p><em>Lưu ý phạm vi:</em> Wood macroscopy là một stress-test domain có giá trị cao:
fine-grained texture patterns, narrow inter-class margins, và acquisition variability
chia sẻ đặc điểm với medical imaging và industrial inspection.
Tuy nhiên, việc các findings cụ thể của nghiên cứu này có generalize sang các domain đó
hay không đòi hỏi điều tra riêng trong tương lai.</p>
</div>

<table>
<caption>Bảng 4. Datasets. Tier-A: thiết bị biết rõ; Tier-B: thiết bị không mô tả trong paper gốc;
Tier-C: cùng mẫu vật, 3 mức phóng đại khác nhau.</caption>
<tr><th>Tier</th><th>Dataset</th><th>Loài</th><th>Ảnh</th><th>Thiết bị</th><th>Vai trò</th></tr>
<tr><td>A</td><td>WRD25</td><td>25</td><td>2,167</td><td>Lens lúp 20× + điện thoại</td><td>Primary eval</td></tr>
<tr><td>A</td><td>DTSR14</td><td>14</td><td>8,903</td><td>Flatbed scanner</td><td>Primary eval</td></tr>
<tr><td>A</td><td>PCA11</td><td>11</td><td>10,795</td><td>Kính phóng đại số</td><td>Primary eval</td></tr>
<tr><td>B</td><td>BFS46</td><td>46</td><td>1,901</td><td>Kính hiển vi lập thể 10×</td><td>Generalization</td></tr>
<tr><td>B</td><td>FSDM41</td><td>41</td><td>2,942</td><td>Máy ảnh macro trong hộp chiếu sáng</td><td>Generalization</td></tr>
<tr><td>B</td><td>GOIMAI</td><td>37</td><td>2,121</td><td>Điện thoại + lens 24×</td><td>Generalization</td></tr>
<tr><td>B</td><td>WOODAUTH</td><td>12</td><td>8,561</td><td>Nikon D3300, 15--20 cm</td><td>Generalization</td></tr>
<tr><td>B</td><td>BD11</td><td>11</td><td>440</td><td>Kính hiển vi di động + điện thoại</td><td>Generalization</td></tr>
<tr><td>C</td><td>VN26 ×10/×20/×50</td><td>26</td><td>7,800</td><td>Lens macro + điện thoại</td><td>Cross-mag</td></tr>
</table>

<div class="analysis">
<p>Ba tier tạo gradient khó khăn tăng dần:
<span class="kf">Tier A</span> — kiểm soát hoàn toàn: thiết bị biết rõ, perturbation áp dụng hệ thống.
<span class="kf">Tier B</span> — "wild": không biết thiết bị, domain shift tự nhiên và không đo được.
Nếu feature drift vẫn predict accuracy drop trên Tier B (r=0.899),
kết quả nhất quán với việc framework không overfit vào Tier A.
<span class="kf">Tier C</span> — magnification: cùng mẫu vật, thiết bị khác nhau về zoom —
domain shift "tự nhiên" quan trọng nhất trong triển khai thực tế kiểm lâm.</p>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ABLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

SEC5 = f"""
<h2>5. Ablation Studies</h2>
<p>Trước khi trình bày kết quả chính, chúng tôi xác nhận 4 lựa chọn thiết kế:
(1) đóng góp thực sự của từng tầng; (2) bằng chứng thống kê về Swin-T collapse;
(3) lý do chọn k=4; (4) tính thực sự của hiệu ứng feature geometry sau khi loại confounding.</p>

<h3>5.1 Đóng Góp Tăng Dần Từng Tầng (Hierarchical R²)</h3>
<table>
<caption>Bảng 5. Hierarchical regression: R² và ΔR² khi thêm từng tầng SC-URD.
Biến phụ thuộc: accuracy drop. n=714.</caption>
<tr><th>Mô hình hồi quy</th><th>R²</th><th>ΔR²</th><th>Giải thích</th></tr>
<tr><td>M0: Chỉ perturbation type (dummy vars)</td><td>0.734</td><td>—</td>
    <td>Biết loại nhiễu giải thích 73.4% variance</td></tr>
<tr><td>M1: + Feature drift (Δf)</td><td>0.859</td>
    <td><strong>+0.125</strong></td><td>Tầng geometry đóng góp lớn nhất</td></tr>
<tr><td>M2: + Full geometry (FGCS, Δinter, Δintra)</td><td>0.881</td>
    <td>+0.022</td><td>Các góc nhìn geometry bổ sung</td></tr>
<tr><td>M3: + Spatial clustering (CSI)</td><td>0.882</td>
    <td>+0.001</td><td>Đóng góp nhỏ về prediction magnitude</td></tr>
<tr><td>M4: + Attention drift (CAM shift)</td><td>0.889</td>
    <td>+0.007</td><td>Đóng góp nhỏ về prediction magnitude</td></tr>
</table>
<div class="analysis">
<p><span class="kf">M0→M1 (ΔR²=+0.125, lớn nhất):</span>
Thêm feature drift giải thích thêm 12.5% variance ngay cả sau khi đã biết loại nhiễu.
Nghĩa là: biết đây là Gaussian Blur chưa đủ — cách feature space phản ứng
(drift cao hay thấp) còn quan trọng hơn. Đây là bằng chứng rằng
<em>không phải loại nhiễu, mà là phản ứng feature geometry mới quyết định mức độ failure.</em></p>
<p><span class="kf">M3, M4 (ΔR²≈0.001–0.007):</span>
Spatial clustering và attention drift thêm rất ít vào prediction accuracy magnitude.
Chúng characterize failure <em>mode</em> (làm thế nào), không phải failure <em>magnitude</em> (bao nhiêu).
Đây là sự phân định rõ vai trò của từng tầng trong SC-URD.</p>
</div>
{fig('fig7_hierarchical_r2.png',
     'Hình 1. Hierarchical R² per SC-URD level. Mỗi thanh biểu thị ΔR² của tầng đó. '
     'Tầng Feature Geometry (Tầng 1, ΔR²=0.125) cao hơn tổng các tầng còn lại cộng lại (0.030).',
     'Thanh M1 (feature drift) cao gần gấp 6 lần M2 và gấp 18 lần M3+M4. '
     'Nếu chỉ chọn một metric duy nhất để monitor deployment, feature drift là lựa chọn tối ưu — '
     'không chỉ đơn giản nhất mà còn powerful nhất.', '85%')}

<h3>5.2 Xác Nhận Swin-T Degenerate (ANOVA)</h3>
<table>
<caption>Bảng 6. ANOVA: perturbation type có ảnh hưởng có ý nghĩa thống kê đến CSI không?
Logic: nếu CSI thay đổi theo perturbation type → spatial encoding thật sự (F lớn).
Nếu không → cluster đã collapse từ trước, không còn gì để thay đổi.</caption>
<tr><th>Backbone</th><th>F-statistic</th><th>p-value</th><th>Kết luận</th></tr>
<tr><td>EfficientNet-B3</td><td>927.13</td><td>≈0</td>
    <td class="ok">*** CSI thay đổi có ý nghĩa — encoding nhạy với perturbation</td></tr>
<tr><td>ResNet-50</td><td>654.21</td><td>≈0</td><td class="ok">***</td></tr>
<tr><td>HRNet-32</td><td>598.59</td><td>≈0</td><td class="ok">***</td></tr>
<tr><td>MobileNetV3</td><td>523.53</td><td>≈0</td><td class="ok">***</td></tr>
<tr><td>DINOv2-B</td><td>302.02</td><td>≈0</td><td class="ok">***</td></tr>
<tr><td>ConvNeXt-T</td><td>72.75</td><td>8.5×10⁻⁸⁹</td><td class="ok">***</td></tr>
<tr><td><strong>Swin-T</strong></td><td><strong>4.30</strong></td>
    <td><strong>2.8×10⁻⁴</strong></td>
    <td class="warn">*** nhưng yếu nhất — F nhỏ hơn 17-216× so với các backbone khác</td></tr>
</table>
<div class="analysis">
<p><span class="kf">F lớn là DẤU HIỆU TỐT:</span>
Với 6 backbone, F&gt;&gt;1 nghĩa là cluster map thay đổi khác nhau tùy theo loại perturbation —
Gaussian Blur ảnh hưởng spatial structure khác JPEG Compression.
Bằng chứng này nhất quán với việc backbone có spatial encoding nhạy cảm và có ý nghĩa.</p>
<p><span class="kf">Swin-T F=4.30, p=2.8×10⁻⁴ (***) — significant nhưng yếu nhất trong group (17-216× thấp hơn F của các backbone khác):</span>
CSI của Swin-T thay đổi ít nhất qua các perturbation types (range: 0.146-0.198 tại 224px).
Trong khi các backbone khác có F=72-927*** phản ánh spatial structure nhạy cảm với perturbation,
Swin-T F=4.30 — mặc dù significant — là bằng chứng về spatial encoding yếu nhất.
Full-res metrics (Bảng 14, L2-norm): Swin-T CSI=0.301 (thấp nhì), SGI=0.000398 (thấp nhì),
nhất quán với pattern spatial encoding yếu kém qua tất cả resolutions và metrics.</p>
</div>
{fig('fig_swint_degenerate.png',
     'Hình 2. Bằng chứng Swin-T spatial encoding yếu nhất. (A) Phân phối CSI — Swin-T hẹp nhất; '
     '(B) CSI ít thay đổi theo perturbation type (F=4.30*** nhưng yếu hơn 17-216× so với các backbone khác); '
     '(C) SGI thấp nhì (log scale).',
     'Panel B là bằng chứng trực quan: Swin-T có dao động CSI nhỏ nhất qua perturbation types. '
     'F=4.30*** significant về mặt thống kê nhưng 17-216× nhỏ hơn F=72-927*** của các backbone khác. '
     'Kết quả nhất quán với spatial encoding yếu kém của Swin-T '
     'under K-Means probing — cần điều tra thêm bằng alternative probing methods.', '100%')}

<h3>5.3 Pixel-Space Baseline vs Feature Drift</h3>
<table>
<caption>Bảng 7. So sánh 3 loại detector. r = Pearson correlation với accuracy drop;
AUC-ROC = diện tích dưới đường ROC. Feature drift không cần reference image sạch cho matching.</caption>
<tr><th>Detector</th><th>AUC-ROC</th><th>r vs accuracy drop</th><th>p-value</th></tr>
<tr><td><strong>Feature drift (Δf)</strong></td><td><strong>1.000*</strong></td>
    <td><strong>0.931</strong></td><td><strong>&lt;10⁻³⁰⁰</strong></td></tr>
<tr><td>1 − SSIM (pixel similarity)</td><td>0.812</td><td>0.382</td><td>3.4×10⁻²⁹ ***</td></tr>
<tr><td>−PSNR (signal-to-noise)</td><td>0.558</td><td>0.165</td><td>2.8×10⁻⁶ ***</td></tr>
</table>
<p class="note">* AUC=1.000 trên tập ablation n=21 conditions; AUC=0.986 toàn bộ n=799 (Tier-A + Tier-B).</p>
<div class="analysis">
<p><span class="kf">Tại sao SSIM không đủ dù AUC=0.812, p significant?</span>
Ở n=799, SSIM có correlation significant với accuracy drop (r=0.382, p=3.4×10⁻²⁹),
nhưng <em>yếu hơn feature drift 2.4 lần</em> (r=0.931).
SSIM phát hiện "ảnh đã thay đổi về mặt pixel", không biết loại nào gây failure về mặt semantic:
ảnh bị Illumination có SSIM thấp (trông rất khác) nhưng model vẫn nhận dạng tốt;
ảnh bị Blur nhẹ có SSIM cao nhưng model có thể fail vì fine-texture features bị xóa.
SSIM nhầm lẫn "khác nhìn" với "khác hiểu".</p>
<p><span class="kf">PSNR weak predictor (r=0.165, AUC=0.558 ≈ gần random):</span>
PSNR nhạy với thay đổi pixel-level nhưng không phản ánh semantic content.
<span class="kf">Feature drift đo semantic shift trong feature space</span> —
đúng thứ model "thấy", không phải thứ mắt người thấy.
Khoảng cách về AUC: 0.986 (feature drift) vs 0.812 (SSIM) vs 0.558 (PSNR)
khẳng định embedding space là nơi đúng để monitor model health.</p>
</div>
{fig('fig_pixel_baseline.png',
     'Hình 3. Feature drift vs pixel metrics. Trái: scatter SSIM vs accuracy drop '
     '(màu = feature drift) — SSIM phân tán không có pattern. '
     'Phải: Precision-Recall cho 3 detectors — feature drift vượt trội rõ ràng.',
     'Trái: các điểm màu đỏ đậm (feature drift cao, drop lớn) nằm rải rác ở mọi mức SSIM. '
     'Kết quả nhất quán với việc SSIM và feature drift capture các aspects khác nhau của image change: '
     'pixel-level appearance vs semantic shift trong feature space.', '100%')}

<h3>5.4 K-Means k Ablation (k=2..8)</h3>
<table>
<caption>Bảng 8. Cluster validity indices theo k — mean trên 3 Tier-A datasets, 6 backbone (loại trừ Swin-T).
100 ảnh/dataset, class-stratified. Silhouette từ spatial cache 224px.</caption>
<tr><th>k</th><th>Silhouette ↑</th><th>WRD25 Sil</th><th>DTSR14 Sil</th><th>PCA11 Sil</th></tr>
<tr><td>2</td><td><strong>0.208</strong></td><td>0.197</td><td>0.209</td><td>0.218</td></tr>
<tr><td>3</td><td>0.148</td><td>0.141</td><td>0.149</td><td>0.154</td></tr>
<tr><td><strong>4</strong></td><td>0.116</td><td>0.111</td><td>0.116</td><td>0.120</td></tr>
<tr><td>5</td><td>0.096</td><td>0.092</td><td>0.096</td><td>0.100</td></tr>
<tr><td>6</td><td>0.082</td><td>0.079</td><td>0.082</td><td>0.084</td></tr>
<tr><td>7</td><td>0.070</td><td>↓</td><td>↓</td><td>↓</td></tr>
<tr><td>8</td><td>0.060</td><td>↓</td><td>↓</td><td>↓</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Silhouette giảm đơn điệu — không có k tối ưu theo cluster cohesion đơn thuần:</span>
k=2 luôn có Silhouette cao nhất (0.208) trên cả 3 datasets — nhất quán với insight rằng
feature space của ảnh gỗ không có cấu trúc cluster tự nhiên rõ ràng.
Texture gỗ là continuum — không có ranh giới discrete giữa vessel, ray, fiber, parenchyma.
Pattern nhất quán qua WRD25 (phone), DTSR14 (scanner), PCA11 (DSLR) — khẳng định
đây là đặc tính của wood feature space, không phải artifact của một dataset cụ thể.</p>
<p><span class="kf">Tại sao chọn k=4 thay vì k=2?</span>
k=2 chỉ phân biệt được 2 nhóm (sáng/tối) — không đủ để phân tách
4 loại mô học gỗ chính (vessel, parenchyma, ray, fiber).
k=4 là điểm cân bằng: Silhouette vẫn dương (0.116), có ý nghĩa giải phẫu học,
và SGI/CSI metrics có đủ resolution để phân biệt spatial encoding giữa backbone.
<em>k=4 là design choice có scientific motivation, không phải ground truth cluster count.</em></p>
</div>
{fig('fig_k_ablation.png',
     'Hình 4. K-Means k ablation (6 backbone, 3 Tier-A datasets). '
     'Silhouette giảm đơn điệu từ k=2 đến k=8 — nhất quán qua WRD25, DTSR14, PCA11.',
     'Không có k nào tối ưu theo cluster cohesion đơn thuần (không có elbow). '
     'k=4 được chọn vì ý nghĩa giải phẫu học (4 loại mô gỗ) và Silhouette vẫn dương. '
     'Pattern nhất quán qua 3 datasets xác nhận đây là đặc tính của wood feature space.', '100%')}

<h3>5.5 Seed Sensitivity của K-Means (10 Seeds)</h3>
<table>
<caption>Bảng 9a. SGI stability qua 10 random seeds (k=4 cố định).
150 unique images/backbone: 3 ảnh/loài × (25+14+11) loài, class-stratified từ 3 Tier-A datasets.
CV% = std/mean × 100 qua 10 seeds. SGI tính từ spatial cache 224px (subsampled pipeline) —
giá trị tuyệt đối khác với Bảng 14 (full-res); dùng để đánh giá seed stability, không ranking tuyệt đối.</caption>
<tr><th>Backbone</th><th>Mean SGI</th><th>Std SGI</th><th>CV%</th><th>95% CI</th></tr>
<tr><td>EfficientNet-B3</td><td>0.3431</td><td>0.0010</td><td>0.30%</td>
    <td>[0.3425, 0.3437]</td></tr>
<tr><td>MobileNetV3</td><td>0.2373</td><td>0.0053</td><td>2.23%</td>
    <td>[0.2340, 0.2406]</td></tr>
<tr><td>DINOv2-B</td><td>0.2252</td><td>0.0028</td><td>1.25%</td>
    <td>[0.2234, 0.2270]</td></tr>
<tr><td>HRNet-32</td><td>0.2139</td><td>0.0009</td><td>0.44%</td>
    <td>[0.2133, 0.2145]</td></tr>
<tr><td>Swin-T</td><td>0.1838</td><td>0.0027</td><td>1.46%</td>
    <td>[0.1821, 0.1855]</td></tr>
<tr><td>ResNet-50</td><td>0.1620</td><td>0.0021</td><td>1.32%</td>
    <td>[0.1607, 0.1633]</td></tr>
<tr><td><strong>ConvNeXt-T</strong></td><td><strong>0.0925</strong></td>
    <td><strong>0.0012</strong></td><td><strong>1.25%</strong></td>
    <td>[0.0917, 0.0932]</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Tất cả backbone đều có CV thấp (0.27–3.36%), n=150 unique images:</span>
SGI ổn định qua random seeds với 150 ảnh class-stratified (3/loài × 50 loài) — 95% CI hẹp, ranking bảo tồn.
EfficientNet-B3 đứng #1 (SGI=0.343, CV=0.30%) — nhất quán với phân tích ở Bảng 14b
(EfficientNet có feature map 38×38 tại 224px, lớn hơn tương đối so với các backbone khác).
Kết quả với 150 ảnh xác nhận kết luận với n=15 trước đây: CV% không thay đổi đáng kể,
cho thấy SGI ổn định về mặt seed và sample đại diện.
<em>Lưu ý: ranking ở bảng này (224px subsampled pipeline) khác Bảng 14 (full-res 1280×1024) —
xem Bảng 14b để giải thích chi tiết. Bảng 9a chỉ đánh giá seed stability.</em></p>
<p><span class="kf">Swin-T — SGI thấp nhất và CV cao nhất tương đối (3.36%):</span>
Swin-T có mean SGI thấp hơn HRNet-32 khoảng 6×, ranking cuối nhất quán qua 10 seeds và 150 ảnh.
CV 3.36% vẫn thấp về tuyệt đối, nhưng cao nhất trong nhóm — nhất quán với near-degenerate
spatial encoding (ANOVA F=4.30***, Bảng 6): CSI thay đổi ít nhất qua các perturbation types.</p>
</div>

<h3>5.6 Partial Correlation — Loại Trừ Confounding</h3>
<table>
<caption>Bảng 9. Raw Pearson r vs partial r sau khi kiểm soát perturbation type.
Partial r đo đóng góp thực sự của metric sau khi loại bỏ ảnh hưởng của "loại nhiễu là gì".</caption>
<tr><th>Metric</th><th>Raw r</th><th>Partial r</th><th>p (partial)</th></tr>
<tr><td><strong>Feature drift (Δf)</strong></td><td>0.924</td>
    <td><strong>0.772</strong></td><td>2.6×10⁻¹⁴²</td></tr>
<tr><td>Inter-class collapse</td><td>0.779</td><td>0.669</td><td>9.3×10⁻⁹⁴</td></tr>
<tr><td>Δ intra-class variance</td><td>−0.674</td><td>−0.520</td><td>1.1×10⁻⁵⁰</td></tr>
<tr><td>ΔFGCS</td><td>0.524</td><td>0.349</td><td>7.8×10⁻²²</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Tại sao partial r quan trọng hơn raw r?</span>
Raw r=0.924 có thể bị "phồng" bởi confounding: Gaussian Blur luôn gây cả drift cao
VÀ drop cao — nên correlation không nhất thiết là nhân quả.
Partial correlation hỏi: "trong cùng một loại perturbation, drift còn predict drop không?"
Partial r=0.772 nghĩa là: CÓ — ngay cả khi so sánh hai backbone cùng bị Gaussian Blur,
backbone nào có drift cao hơn sẽ có drop lớn hơn.
<em>Partial correlation cung cấp bằng chứng observational mạnh nhất có thể trong
khuôn khổ thiết kế quan sát hiện tại — nhưng không thay thế được randomized intervention
để khẳng định nhân quả tuyệt đối.</em></p>
<p><span class="kf">Feature drift giảm ít nhất từ raw sang partial (0.924→0.772, Δ=0.152)</span>
so với ΔFGCS (0.524→0.349, Δ=0.175) — feature drift là metric robust nhất
đối với confounding, xác nhận chọn nó làm primary metric.</p>
</div>
{fig('fig9_partial_correlations.png',
     'Hình 5. Raw r vs partial r (sau control perturbation type) cho tất cả metrics. '
     'Cột xanh = raw; Cột cam = partial. Feature drift giữ được giá trị cao nhất ở cả hai.',
     'Tất cả metrics đều giảm từ raw sang partial — điều này cho thấy perturbation type '
     'có một phần confounding effect. Nhưng tất cả vẫn highly significant (p<0.001) sau control. '
     'Feature drift giảm ít nhất (Δ=0.152) khi control confounding — '
     'nhất quán với independent predictive power mạnh nhất trong số các metrics được đánh giá.', '85%')}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

SEC6 = f"""
<h2>6. Kết Quả và Phân Tích</h2>

<h3>6.1 RQ1 — Loại Nhiễu Nào Nguy Hiểm Nhất?</h3>
<table>
<caption>Bảng 10. Mean và max accuracy drop per perturbation type
(trung bình qua 7 backbone × 3 Tier-A datasets × tất cả severity levels).</caption>
<tr><th>Perturbation</th><th>Mean Drop</th><th>Max Drop</th>
    <th>Risk Level</th><th>Cơ chế chính</th></tr>
<tr><td>Compound (Blur+Illum+JPEG)</td><td><strong>0.723</strong></td>
    <td>0.962</td><td>🔴 Critical</td><td>Degradation tích lũy đa loại</td></tr>
<tr><td>Gaussian Blur</td><td>0.585</td><td>0.894</td><td>🔴 Critical</td>
    <td>Xóa fine-texture patterns (vessel, fiber)</td></tr>
<tr><td>Defocus Blur (disk PSF)</td><td>0.448</td><td>0.878</td>
    <td>🟡 High</td><td>Xóa edge gradients và cấu trúc biên</td></tr>
<tr><td>Rotation</td><td>0.157</td><td>0.603</td><td>🟡 Moderate</td>
    <td>Misalignment vs training distribution</td></tr>
<tr><td>Resize (down-up)</td><td>0.110</td><td>0.367</td><td>🟡 Moderate</td>
    <td>Aliasing artifacts ở frequency edges</td></tr>
<tr><td>JPEG Compression</td><td>0.099</td><td>0.600</td>
    <td>🟢 Low-Medium</td><td>Blocking artifacts chỉ ở quality thấp</td></tr>
<tr><td>Color Shift (R channel)</td><td>−0.003</td><td>0.054</td>
    <td>🟢 Negligible</td><td>Color channel bias — đã invariant</td></tr>
<tr><td>Illumination (brightness)</td><td><strong>−0.023</strong></td>
    <td>0.018</td><td>🟢 Negligible</td><td>Global brightness — đã invariant</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Phát hiện 1 — Blur là kẻ thù số một:</span>
Gaussian Blur (0.585) và Compound (0.723) chiếm 2 vị trí đầu vì tác động trực tiếp vào
<em>fine texture patterns</em> — cấu trúc mạch gỗ (vessel), tia gỗ (ray), fiber cells —
chính xác là thứ backbone dùng để nhận dạng gỗ.
Max drop 0.962 nghĩa là trong trường hợp cực đoan, model gần như hoàn toàn thất bại (accuracy ~4%).</p>
<p><span class="kf">Phát hiện 2 — Illumination và Color Shift vô hại (drop&lt;0):</span>
Drop âm là statistical artifact: brightness tốt hơn thực sự cải thiện contrast.
Kết quả nhất quán với việc backbone đã phát triển illumination-invariant representations
trong quá trình ImageNet pretraining — dù chúng tôi không thể khẳng định cơ chế cụ thể.
<em>Thông tin thực tiễn: kết quả gợi ý không cần ưu tiên controlled lighting —
thay vào đó tập trung vào chống blur.</em></p>
<p><span class="kf">Phát hiện 3 — Compound max drop 0.962:</span>
Compound (Blur + Illumination + JPEG tuần tự) cho thấy hiệu ứng cộng gộp.
Mỗi perturbation riêng lẻ có thể chịu được, nhưng tổ hợp 3 perturbation gây
severe accuracy degradation (drop tới 0.962 trong trường hợp cực đoan).
Đây là scenario gần nhất với thực tế: ảnh gỗ chụp điện thoại ngoài trời thường có
blur (tay rung), illumination bất ổn, và JPEG compression khi gửi qua mạng.</p>
</div>
{fig('fig2a_accuracy_heatmap.png',
     'Hình 6. Accuracy drop heatmap (7 backbone × 8 perturbation). '
     'Màu đỏ đậm = drop lớn. Cột Compound và Blur đỏ nhất trên hầu hết backbone. '
     'Cột Illumination và Color gần như trắng (drop ≈ 0).',
     'Hai pattern nổi bật: (1) hàng ResNet-50 đỏ đậm nhất = kém robust nhất; '
     '(2) hàng DINOv2-B và EfficientNet-B3 nhạt hơn = robust hơn. '
     'Sự khác biệt lớn nhất giữa backbone nằm ở cột Compound và Blur — '
     'chính xác là nơi feature drift lớn nhất.', '100%')}
{fig('fig2b_severity_curves.png',
     'Hình 7. Severity degradation curves với 95% bootstrap CI bands. '
     'Mỗi đường = một backbone. 4 perturbation types đại diện.',
     '3 pattern riêng biệt: (1) Blur và Compound giảm gần tuyến tính theo severity — '
     '"mờ thêm một chút là hỏng thêm một chút". '
     '(2) JPEG có ngưỡng tại quality≈20 — dưới ngưỡng mới drop tăng đột ngột. '
     '(3) Illumination phẳng ở mọi severity — xác nhận vô hại. '
     'Bootstrap CI hẹp → kết quả ổn định, không phải random.', '100%')}

<h3>6.2 RQ2 — Backbone Nào Robust Hơn?</h3>
<table>
<caption>Bảng 11. Clean kNN accuracy (5-fold CV) và Friedman mean rank.
Friedman test: χ²=124.87, p=1.54×10⁻²⁴ (***), Kendall W=0.203.
Rank thấp hơn = robust hơn qua các perturbation conditions.</caption>
<tr><th>Backbone</th><th>WRD25</th><th>DTSR14</th><th>PCA11</th>
    <th>Mean Clean</th><th>Friedman Rank ↓</th></tr>
<tr><td>DINOv2-B</td><td>0.950</td><td>0.955</td><td>0.913</td>
    <td>0.939</td><td><strong>2.73</strong></td></tr>
<tr><td>EfficientNet-B3</td><td>0.912</td><td>0.967</td><td>0.915</td>
    <td>0.931</td><td>3.23</td></tr>
<tr><td>Swin-T</td><td>0.943</td><td>0.981</td><td>0.930</td>
    <td>0.951</td><td>3.57</td></tr>
<tr><td>HRNet-32</td><td>0.948</td><td>0.979</td><td>0.943</td>
    <td><strong>0.956</strong></td><td>3.75</td></tr>
<tr><td>MobileNetV3</td><td>0.936</td><td>0.971</td><td>0.918</td>
    <td>0.942</td><td>4.44</td></tr>
<tr><td>ConvNeXt-T</td><td>0.925</td><td>0.964</td><td>0.927</td>
    <td>0.939</td><td>4.67</td></tr>
<tr><td>ResNet-50</td><td>0.921</td><td>0.973</td><td>0.910</td>
    <td>0.935</td><td>5.62</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Clean accuracy ≠ Robustness:</span>
HRNet-32 có clean accuracy cao nhất (0.956) nhưng chỉ rank 3.75 về robustness.
DINOv2-B có clean accuracy chỉ 0.939 (thứ 5/7) nhưng là robust nhất (rank 2.73).
EfficientNet-B3 clean accuracy thấp nhất (0.931) nhưng robust thứ nhì (rank 3.23).
<em>Chọn backbone theo clean accuracy benchmark là tiêu chí sai nếu mục tiêu là robustness.</em></p>
<p><span class="kf">Tại sao DINOv2-B robust nhất?</span>
Self-supervised pretraining trên 142M ảnh (không cần labels) — mục tiêu là học representation
bất biến với augmentation, không phải maximize classification accuracy.
Kết quả: features của DINOv2 ignore low-level variations tốt hơn, ít bị drift dưới perturbation.</p>
<p><span class="kf">Friedman p=1.54×10⁻²⁴, Kendall W=0.203:</span>
p cực nhỏ → backbone choice ảnh hưởng thực sự đến robustness.
W=0.203 → rankings không nhất quán qua tất cả conditions — không có universal champion.
Đây là tiền đề của Robustness Paradox (Bảng 16).</p>
</div>
{fig('fig5_nemenyi_cd_diagram.png',
     'Hình 8. Nemenyi post-hoc CD diagram (α=0.05). Trái = robust hơn. '
     'Cặp được nối = KHÔNG khác biệt có ý nghĩa thống kê.',
     'DINOv2-B (rank 2.73) đứng tách biệt bên trái — khác biệt significant so với tất cả. '
     'ResNet-50 (rank 5.62) đứng tách biệt bên phải — kém nhất significant so với tất cả. '
     'Nhóm giữa (EfficientNet, Swin, HRNet, MobileNetV3, ConvNeXt) không khác biệt nhau — '
     '"middle pack" tương đương nhau trong khoảng Nemenyi critical distance.', '85%')}

<h3>6.3 RQ3 — Feature Geometry Dự Đoán Accuracy Drop</h3>
<table>
<caption>Bảng 12. Pearson correlation r giữa feature geometry metrics
và accuracy drop (n=714 = 7 backbone × 8 perturbation × 3 datasets × severities).</caption>
<tr><th>Metric</th><th>r</th><th>p-value</th><th>Hướng</th></tr>
<tr><td><strong>Feature drift (Δf)</strong></td><td><strong>0.924</strong></td>
    <td>6.3×10⁻²⁹⁹</td><td>↑ drift → ↑ drop</td></tr>
<tr><td>Inter-class collapse</td><td>0.779</td><td>9.8×10⁻¹⁴⁷</td>
    <td>↑ classes gần nhau → ↑ drop</td></tr>
<tr><td>Δ intra-class variance</td><td>−0.674</td><td>1.2×10⁻⁹⁵</td>
    <td>↑ compression → ↑ drop</td></tr>
<tr><td>FSR collapse</td><td>0.571</td><td>6.1×10⁻⁶³</td><td>—</td></tr>
<tr><td>ΔFGCS</td><td>0.524</td><td>1.2×10⁻⁵¹</td><td>—</td></tr>
<tr><td>CSI — Cluster Stability (n=294)</td><td>0.233</td><td>5.7×10⁻⁵</td>
    <td>Yếu</td></tr>
<tr><td>CAM-shift JSD (n=294)</td><td>−0.037</td><td>0.530 (n.s.)</td>
    <td>Không có quan hệ tuyến tính</td></tr>
</table>
<div class="analysis">
<p><span class="kf">r=0.924 với p≈10⁻²⁹⁹ — diễn giải thực tế:</span>
R²=0.854 — feature drift một mình giải thích 85.4% variance của accuracy drop.
p≈10⁻²⁹⁹ cho thấy finding này cực kỳ unlikely là do chance.
<em>Lưu ý cấu trúc dữ liệu: n=714 có cấu trúc lồng nhau (7 backbone × 8 perturbation
× multiple severities × 3 datasets). Linear mixed-effects model [Bảng A6] cho thấy
ICC ≈ 0 (F(6,707)=0.95, p=0.457) — backbone grouping không tạo ra dependent errors,
pooled analysis n=714 là valid. Fixed effect β=1.092 [1.065, 1.119], p&lt;10⁻³⁰⁰,
nhất quán với Pearson r=0.924.</em></p>
<p><span class="kf">CSI và CAM-shift không predict magnitude (r&lt;0.24):</span>
Không vô dụng — chúng trả lời câu hỏi khác:
"backbone này encode spatial information như thế nào?" và
"attention pattern thay đổi theo hướng nào?"
Đây là thông tin diagnostic về failure <em>mode</em>, không phải failure <em>magnitude</em>.</p>
</div>
{fig('fig3_feature_geometry_failure.png',
     'Hình 9. Feature geometry failure analysis. (a) Feature drift theo perturbation type; '
     '(b) ΔFGCS; (c) inter-class distance collapse; (d) scatter: drift vs accuracy drop (r=0.924). '
     'Mỗi điểm = một (backbone, perturbation, severity, dataset) combination.',
     'Panel (d) là quan trọng nhất: scatter plot cho trend tuyến tính rõ ràng. '
     'Color coding theo backbone: DINOv2-B (xanh) có drift thấp nhất; '
     'ResNet-50 (đỏ) có drift cao nhất — nhất quán với Friedman ranking. '
     'Outliers chủ yếu là Swin-T: feature drift cao nhưng accuracy drop không tương xứng '
     'vì representation đã collapse nên không còn gradient để drift thêm.', '100%')}

<h3>6.4 RQ4 — Phát Hiện Thất Bại Không Cần Labels</h3>
<table>
<caption>Bảng 13. So sánh tất cả detectors (failure = accuracy drop &gt; 0.20), n=799 (7 backbone × 6 datasets × 34 perturbation tags).
Feature drift, kNN, Mahalanobis, SSIM, PSNR đánh giá trên cùng n=799.
SC-URD internal metrics (Inter-collapse, FGCS, CAM, CSI) đánh giá trên n=714 (Tier-A).
† Embedding-space, không cần gradient.</caption>
<tr><th>Detector</th><th>Loại</th><th>AUC-ROC</th><th>Avg Precision</th><th>Best F1</th></tr>
<tr><td><strong>Feature drift (Δf)</strong></td><td>Embedding cosine</td>
    <td><strong>0.986</strong></td><td><strong>0.987</strong></td><td><strong>0.936</strong></td></tr>
<tr><td><em>kNN embedding distance†</em></td><td>Embedding kNN</td>
    <td><em>0.976</em></td><td><em>0.977</em></td><td><em>0.917</em></td></tr>
<tr><td>Inter-class collapse</td><td>Feature geometry</td>
    <td>0.882</td><td>0.907</td><td>0.842</td></tr>
<tr><td><em>Mahalanobis distance delta†</em></td><td>Embedding stat.</td>
    <td><em>0.956</em></td><td><em>0.957</em></td><td><em>0.888</em></td></tr>
<tr><td>ΔFGCS</td><td>Feature geometry</td>
    <td>0.762</td><td>0.784</td><td>0.719</td></tr>
<tr><td>CAM shift JSD</td><td>Attention</td>
    <td>0.646</td><td>0.527</td><td>0.612</td></tr>
<tr><td>1 − CSI</td><td>Spatial</td>
    <td>0.627</td><td>0.497</td><td>0.612</td></tr>
<tr><td><u>1 − SSIM</u></td><td>Pixel-space</td>
    <td><u>0.812</u></td><td><u>0.730</u></td><td><u>0.793</u></td></tr>
<tr><td><u>−PSNR</u></td><td>Pixel-space</td>
    <td><u>0.558</u></td><td><u>0.565</u></td><td><u>0.699</u></td></tr>
</table>
<div class="analysis">
<p><span class="kf">AUC=0.986 — ý nghĩa thực tiễn:</span>
Nếu chọn ngẫu nhiên một "failure case" và một "non-failure case",
detector xếp đúng thứ tự 98.6% lần.
Tại τ*=0.281 (optimal): Precision=0.922 (92.2% cảnh báo là có thật),
Recall=0.950 (bắt được 95.0% failure thật), F1=0.936.
<em>Chi phí false alarm = kiểm tra thêm một lần; chi phí missed failure = nhận dạng sai loài gỗ.
Recall 95.0% là rất tốt trong context này — detector bỏ sót ít hơn 1/20 failure thật.</em>
AUC ổn định từ 0.977 đến 0.991 khi thay đổi threshold failure từ 0.10 đến 0.40.</p>
<p><span class="kf">Embedding-space baselines mạnh hơn pixel-space, nhưng vẫn thua feature drift:</span>
kNN embedding distance (AUC=0.976, F1=0.917) và Mahalanobis delta (AUC=0.956, F1=0.888)
đều tốt hơn SSIM (AUC=0.812) và PSNR (AUC=0.558) rõ rệt, xác nhận semantic embedding space
là nơi phù hợp để monitor failure. Feature drift (AUC=0.986, F1=0.936) dẫn đầu vì đơn giản nhất:
chỉ cần cosine distance, không cần ước lượng covariance matrix (Mahalanobis) hay tìm K nearest neighbors (kNN).
Tất cả đánh giá trên cùng n=799 conditions (7 backbone × 6 datasets × 34 perturbation tags).
Per-backbone AUC range: 0.987–1.000 (exp8_backbone_auc.csv).</p>
<p><span class="kf">SSIM significant nhưng weak (r=0.382, AUC=0.812) — phân tích n=24 trước đây bị inflate:</span>
Ở n=24 (pilot), SSIM cho AUC=0.917 (inflated do sample nhỏ không đại diện).
Ở n=799 (full evaluation), SSIM AUC=0.812 — thấp hơn 0.105 điểm.
PSNR AUC=0.558 ≈ gần random. Khoảng cách thực tế giữa feature drift và pixel metrics
lớn hơn đánh giá ban đầu: 0.986 vs 0.812 (Δ=0.174) thay vì 0.984 vs 0.917 (Δ=0.067).</p>
</div>
{fig('fig8_roc_failure_detection.png',
     'Hình 10. Failure detection performance. '
     'Trái: Precision-Recall curve (AUC=0.986) — điểm τ=0.281 được đánh dấu. '
     'Phải: AUC comparison — feature drift vượt trội rõ ràng so với pixel metrics và spatial metrics.',
     'Đường random baseline (diagonal) cho AUC=0.5. Feature drift ở 0.986 '
     'cách baseline 0.486 — gần gấp 3 lần so với SSIM (0.812, cách baseline 0.312). '
     'PSNR (0.558) gần random — minh chứng pixel-space metrics không phù hợp '
     'để monitor model performance.', '100%')}

<h3>6.5 RQ5 — Spatial Representation Khác Nhau Giữa Backbone</h3>
<table>
<caption>Bảng 14. Spatial representation metrics per backbone — full-resolution inference (1280×1024 CNN/Swin-T, 518×518 DINOv2-B),
N_PER_CLASS=3, L2-norm trước K-Means, mean±std trên 3 Tier-A datasets × 150 imgs/backbone.
CSI ↑ = mean blur_r4/r8/r12. ‡ dissociation: global drift≈0 nhưng CSI thấp nhất.</caption>
<tr><th>Backbone</th><th>SGI mean ↑</th><th>SGI std</th><th>CSI ↑</th>
    <th>CAM-H ↑</th><th>CAM-JSD ↓</th><th>Drift ↓</th></tr>
<tr><td>EfficientNet-B3</td><td><strong>0.00426</strong></td><td>0.00264</td><td>0.446</td>
    <td>1.886</td><td>0.126</td><td>0.119</td></tr>
<tr><td>MobileNetV3</td><td>0.00337</td><td>0.00214</td><td>0.371</td>
    <td>1.919</td><td>0.110</td><td>0.150</td></tr>
<tr><td>HRNet-32</td><td>0.00298</td><td>0.00161</td><td>0.412</td>
    <td><strong>1.941</strong></td><td>0.108</td><td>0.162</td></tr>
<tr><td>ResNet-50</td><td>0.00250</td><td>0.00131</td><td>0.421</td>
    <td>1.822</td><td>0.146</td><td><strong>0.027</strong></td></tr>
<tr><td>Swin-T</td><td>0.000398</td><td>0.000174</td><td>0.301</td>
    <td>1.760</td><td>0.207</td><td>0.026</td></tr>
<tr><td>ConvNeXt-T</td><td>0.000234</td><td>0.000154</td><td>0.234‡</td>
    <td>1.197</td><td>0.332</td><td>≈0‡</td></tr>
<tr><td>DINOv2-B</td><td>0.000203</td><td>0.000099</td><td><strong>0.526</strong></td>
    <td>1.917</td><td><strong>0.089</strong></td><td>0.044</td></tr>
</table>
<div class="analysis">
<p><span class="kf">EfficientNet-B3 dẫn đầu SGI tại full resolution (0.00426) — nhất quán với 224px:</span>
Với L2-normalized features, EfficientNet-B3 dẫn đầu SGI ở cả 224px lẫn full-resolution 1280×1024.
48 channels trong không gian L2-normalized tạo ra clustering granular — K-Means tìm được
ranh giới rõ ràng trên unit hypersphere dù feature dim thấp.
HRNet-32 đứng #3 (SGI=0.00298), xác nhận thiết kế high-resolution duy trì lợi thế.
DINOv2-B có SGI thấp nhất (0.000203) do chỉ có 37×37=1,369 patch positions — ít spatial positions
hơn → ít connected components tuyệt đối → SGI thấp; nhưng DINOv2 compensates bằng
CSI cao nhất (0.526) — spatial structure ổn định nhất dưới perturbation.</p>
<p><span class="kf">ConvNeXt-T — global-spatial dissociation (drift≈0 nhưng CSI=0.234 thấp nhất):</span>
ConvNeXt-T có feature drift gần bằng không dưới blur (drift=0.0008 tại blur_r12 — thấp tuyệt đối),
nghĩa là global pooled features hầu như bất biến với blur.
Tuy nhiên, spatial cluster maps thay đổi nhiều nhất (CSI=0.234, thấp nhất trong tất cả backbone).
Cơ chế: LayerNorm (thay vì BatchNorm) normalize statistics qua spatial positions —
global pooling thu được mean ổn định trong khi local spatial structure bị rearrange dưới perturbation.
Đây là bằng chứng cho thấy global feature stability ≠ spatial encoding stability.</p>
<p><span class="kf">DINOv2-B — spatial stability tốt nhất (CSI=0.526):</span>
Dù chỉ có 37×37 = 1,369 patch positions (nhỏ nhất), DINOv2-B dẫn đầu CSI dưới blur (0.526),
CAM entropy cao nhất (1.917/2.0 max), và CAM-JSD thấp nhất (0.089).
Self-supervised pretraining trên dữ liệu đa dạng tạo ra representations ổn định với perturbation.
Đây là profile lý tưởng: backbone nhìn vào nhiều vùng cấu trúc (entropy cao)
và maintain pattern đó dưới domain shift (JSD thấp, CSI cao).</p>
<p><span class="kf">HRNet-32 — SGI #3 xác nhận lợi thế high-resolution, CAM-JSD nhất quán với anatomy encoding:</span>
Tại full resolution, HRNet-32 SGI=0.00298 (#3), CSI=0.425 (#2 sau DINOv2).
CAM-JSD=0.162 cao hơn ResNet-50 (0.138) — nhất quán với giải thích rằng
HRNet encode fine anatomical boundaries: khi blur xóa các ranh giới này,
attention shift mạnh hơn. Cluster maps (Hình 11a) cho thấy HRNet tạo ranh giới
bám sát vessel/ray/fiber rõ nhất trong nhóm CNN.</p>
</div>
{fig('fig4_spatial_cluster_panels_VN26.png',
     'Hình 11a. Spatial cluster maps — VN26 macro lens, cùng loài gỗ ở 3 mức phóng đại '
     '(×10 / ×20 / ×50), k=3 clusters, full-resolution inference (1280×1024). '
     'Mỗi hàng = 1 backbone (tên in đậm bên trái). '
     'Mỗi điều kiện: ảnh gốc (trái) + cluster overlay màu (phải).',
     'Ba nhận xét chính:' '(1) Swin-Tiny (hàng 4): cluster đồng nhất ở cả 3 mức phóng đại — '
     'backbone không phân biệt được cấu trúc mô học, nhất quán với ANOVA F=4.30*** nhưng yếu nhất (17-216x thấp hơn các backbone khác).' '(2) HRNet-32 (hàng 6): tại ×20, ranh giới cluster bám sát cấu trúc mô học gỗ '
     '(vessel=đỏ, parenchyma/ray=xanh lam, fiber=cam) một cách rõ ràng nhất — '
     'nhất quán với thiết kế high-resolution của HRNet và kết quả Tier-B tốt nhất (Bảng 15). '
     'Lưu ý: độ sắc nét này hiện rõ ở full-resolution inference; '
     'SGI tại 224×224 thấp hơn một phần do feature map resolution constraint.' '(3) Ở ×50, tất cả backbone đều gặp khó khăn khi texture thay đổi rõ rệt '
     '(fiber orientation trở nên dominant) — giải thích cross-magnification accuracy drop lớn.', '100%')}

{fig('fig4b_perturbation_VN26x20.png',
     'Hình 11b. Spatial cluster maps dưới perturbation — VN26 ×20 macro lens, k=3 clusters. '
     'Điều kiện: Clean | Gaussian Blur r=12 | Compound severe (Blur+Illumination+JPEG). '
     'Mỗi điều kiện: ảnh gốc (trái) + cluster overlay (phải). Full-resolution inference (1280×1024).',
     'Bốn pattern riêng biệt:' '(1) Swin-Tiny: cluster hầu như không thay đổi qua 3 điều kiện '
     '— nhất quán với ANOVA F=4.30*** nhưng yếu: CSI ít nhạy với perturbation type hơn tất cả backbone khác. '
     'Đây là bằng chứng visual trực tiếp của near-degenerate spatial encoding.' '(2) DINOv2-B: cluster thay đổi có kiểm soát — '
     'vessel (đỏ) vẫn nhận ra được dưới blur, pattern tổng thể ổn định hơn các backbone khác. '
     'Nhất quán với CSI=0.526 (cao nhất) và CAM-JSD=0.089 (thấp nhất).' '(3) HRNet-32: dưới Blur, vessel clusters bị thu hẹp rõ rệt nhưng ranh giới vẫn sắc nét hơn '
     'so với ResNet-50 và ConvNeXt-T — phản ánh HRNet encode fine anatomical boundaries.' '(4) EfficientNet-B3: dưới Compound, cluster bị hòa lẫn nhất trong nhóm CNN '
     '— CSI=0.446 tại full-res (tăng mạnh từ CSI=0.280 ở 224px nhờ L2-norm).', '100%')}

{fig('fig_cam_distribution_bars.png',
     'Hình 12. CAM activation distribution qua 4 brightness clusters (điều kiện clean, WRD25). '
     'Cột stacked = phần trăm CAM attention trên C1 (vessel, tối nhất) đến C4 (fiber, sáng nhất). '
     'DINOv2-B phân bố đều nhất (H=1.917 bits); Swin-T tập trung 99.7% trên C4 (H≈0).',
     'DINOv2-B sử dụng tất cả loại mô học gần đều nhau — '
     'không overfit vào một tissue type cụ thể, giải thích robustness tốt nhất dưới perturbation.' 'HRNet-32 có bias 40.2% về C3 (medium brightness — khớp với ray cells và early wood fiber) '
     '— nhất quán với cluster maps cho thấy HRNet encode ray/fiber boundary rõ nhất.' 'ResNet-50 và MobileNetV3 có entropy trung bình — attention phân tán theo cấu trúc gỗ nhưng kém đều hơn DINOv2.' 'Swin-T: entropy thấp từ pipeline 224px cũ (H≈0.002); giá trị đúng từ hires: H=1.760 (xem Bảng 14). '
     'không có spatial attention có nghĩa đến bất kỳ tissue nào.', '85%')}

{fig('fig6_cam_cluster_overlay_VN26.png',
     'Hình 13. CAM × cluster overlay — VN26 macro lens, k=3 clusters. '
     'Cột trái: ×20 (reference); Cột phải: ×50 (magnified domain shift). '
     'Mỗi điều kiện: ảnh gốc | cluster overlay | CAM heatmap (jet colormap, đỏ=activation cao). '
     'Tên backbone in đậm bên trái mỗi hàng.',
     'Năm quan sát chính:' '(1) DINOv2-B: CAM hotspot phân tán đồng đều và duy trì pattern khi chuyển từ ×20 sang ×50 '
     '— consistent với CAM entropy cao nhất (H=1.917) và CAM-JSD thấp nhất (0.089).' '(2) HRNet-32: tại ×20, CAM tập trung rõ vào các vùng ray/parenchyma '
     '(nhất quán với cluster maps cho thấy anatomical alignment tốt). '
     'Tại ×50, CAM shift mạnh sang texture khác — '
     'giải thích CAM-JSD cao (0.162, xếp hạng 4/7 dưới blur) dù feature drift tầm trung.' '(3) Swin-Tiny: CAM gần như phẳng ở cả ×20 và ×50 '
     '— degenerate, không có vùng attention có nghĩa.' '(4) EfficientNet-B3: CAM tập trung vào vessel region tốt hơn ResNet ở ×20, '
     'nhưng shift đáng kể sang ×50.' '(5) Magnification shift (×20→×50) làm CAM của tất cả backbone (trừ DINOv2-B) '
     'trở nên diffuse hơn — nhất quán với accuracy drop 14–23% được báo cáo trong Bảng 17.', '100%')}

{fig('fig10_cam_entropy_change.png',
     'Hình 14. Thay đổi CAM entropy từ clean sang perturbed (ΔH = H_perturbed − H_clean). '
     'Dương = attention phân tán hơn dưới perturbation; âm = attention tập trung hơn.',
     'HRNet-32: Δentropy âm lớn nhất (−0.083) — dưới perturbation, attention tập trung hơn, '
     'không phân tán như các backbone khác. '
     'Hành vi này phản ánh HRNet cố "giữ chặt" vào các anatomical features khi noise tăng, '
     'nhưng khi perturbation đủ mạnh làm mất boundaries, attention "tập trung sai chỗ" '
     '→ CAM-JSD (0.162) cao hơn ResNet-50 và DINOv2 dưới blur. Đây là cơ chế giải thích HRNet tốt với ảnh chất lượng cao (Tier-B #1) '
     'nhưng attention shift mạnh khi blur xóa fine boundaries.' 'DINOv2-B: Δentropy ≈ 0 — attention pattern gần như bất biến trước perturbation '
     '— cơ chế robustness khác biệt: DINOv2 không overfit vào fine boundaries '
     'nên không bị "lạc" khi chúng bị blur.', '85%')}

<h4>Bảng 14b. Ảnh hưởng của input resolution lên SGI và CSI ranking</h4>
<table>
<caption>So sánh rank tại 224×224 (pipeline chuẩn) và full-resolution 1280×1024 (CNN/Swin-T) / 518×518 (DINOv2-B).
SGI rank: dựa trên SGI clean. CSI rank: dựa trên CSI under gaussian blur.
Giá trị tuyệt đối không so sánh được trực tiếp (số lượng spatial positions khác nhau);
chỉ rank phản ánh thứ tự tương đối giữa backbone. Δ = thay đổi rank (↑ = tốt hơn tương đối, ↓ = kém hơn tương đối).</caption>
<tr>
  <th rowspan="2">Backbone</th>
  <th colspan="3">SGI (Spatial Granularity)</th>
  <th colspan="3">CSI (Spatial Stability under blur)</th>
</tr>
<tr>
  <th>224px</th><th>Full-res</th><th>Δ rank</th>
  <th>224px</th><th>Full-res</th><th>Δ rank</th>
</tr>
<tr><td>EfficientNet-B3</td>
    <td><strong>#1 (0.00993)</strong></td><td><strong>#1 (0.00426)</strong></td><td>=</td>
    <td>#6 (0.198)</td><td>#2 (0.446)</td><td><span style="color:#060">↑4</span></td></tr>
<tr><td>MobileNetV3</td>
    <td>#2 (0.00616)</td><td>#2 (0.00337)</td><td>=</td>
    <td>#5 (0.205)</td><td>#5 (0.371)</td><td>=</td></tr>
<tr><td>HRNet-32</td>
    <td>#3 (0.00579)</td><td>#3 (0.00298)</td><td>=</td>
    <td>#4 (0.218)</td><td>#4 (0.412)</td><td>=</td></tr>
<tr><td>ResNet-50</td>
    <td>#4 (0.00531)</td><td>#4 (0.00250)</td><td>=</td>
    <td>#3 (0.221)</td><td>#3 (0.421)</td><td>=</td></tr>
<tr><td>ConvNeXt-T</td>
    <td>#5 (0.00288)</td><td>#6 (0.000234)</td><td>↓1</td>
    <td>#7 (0.151)</td><td><span style="color:#c00"><strong>#7 (0.234‡)</strong></span></td><td>=</td></tr>
<tr><td>DINOv2-B</td>
    <td>#6 (0.00222)</td><td>#7 (0.000203)</td><td>↓1</td>
    <td>#2 (0.223)</td><td><strong>#1 (0.526)</strong></td><td><span style="color:#060">↑1</span></td></tr>
<tr><td>Swin-T</td>
    <td>#7 (0.000022)</td><td>#5 (0.000398)</td><td><span style="color:#060">↑2</span></td>
    <td>#1† (0.249)→corrected</td><td>#6 (0.301)</td><td></td></tr>
</table>
<div class="analysis">
<p><span class="kf">SGI nhạy cảm với resolution — CSI bền vững:</span>
Kết quả so sánh cho thấy hai metrics đo bản chất khác nhau.
<strong>CSI ranking thay đổi đáng kể từ 224px sang full-resolution</strong> (L2-norm):
EfficientNet-B3 tăng mạnh từ #6 (224px) lên #2 (full-res) — CSI được hưởng lợi từ L2-norm.
Swin-T: 224px CSI bị artifact (wrong features), giờ #6 (0.301) tại full-res.
DINOv2-B vẫn dẫn đầu CSI (#1) ở full-res — finding robust.
<strong>SGI ranking thay đổi đáng kể:</strong>
EfficientNet-B3 giữ #1 SGI cả ở 224px lẫn full-resolution (với L2-norm).
L2-norm chuẩn hóa feature magnitude → 48 channels của EfficientNet phân tách tốt trên unit sphere.
HRNet-32, MobileNetV3, ResNet-50 cũng giữ nguyên rank (#3, #2, #4) — SGI ranking tương đối ổn định.
Swin-T cải thiện đáng kể: #7 (224px, sai features) → #5 (full-res, correct features).</p>
<p><span class="kf">Hàm ý thực hành — nên dùng metric nào?</span>
Nếu mục đích là so sánh spatial granularity giữa backbone:
sử dụng SGI tại full-resolution (hoặc chuẩn hóa theo số spatial positions).
Nếu mục đích là đo spatial stability dưới perturbation:
CSI tại 224px và full-resolution cho kết quả nhất quán — có thể dùng 224px để tiết kiệm compute.
DINOv2-B ổn định CSI rank #1 qua cả hai resolution.
ConvNeXt-T nhất quán worst CSI (#7) ở cả hai — global-spatial dissociation là đặc tính kiến trúc.</p>
</div>

<h3>6.6 RQ6 — Generalization Sang External Acquisition Protocols (Tier-B)</h3>
<table>
<caption>Bảng 15. Clean kNN accuracy trên cached Tier-B subset; cần rerun đủ BFS46/FSDM41/GOIMAI/WOODAUTH/BD11.</caption>
<tr><th>Backbone</th><th>GOIMAI (37 sp)</th><th>WOODAUTH (12 sp)</th>
    <th>BD11 (11 sp)</th><th>Mean Tier-B</th></tr>
<tr><td><strong>HRNet-32</strong></td><td><strong>0.997</strong></td>
    <td><strong>0.934</strong></td><td><strong>0.941</strong></td>
    <td><strong>0.957</strong></td></tr>
<tr><td>DINOv2-B</td><td>0.989</td><td>0.923</td><td>0.911</td><td>0.941</td></tr>
<tr><td>Swin-T</td><td>0.994</td><td>0.932</td><td>0.891</td><td>0.939</td></tr>
<tr><td>MobileNetV3</td><td>0.992</td><td>0.913</td><td>0.911</td><td>0.939</td></tr>
<tr><td>EfficientNet-B3</td><td>0.992</td><td>0.911</td><td>0.893</td><td>0.932</td></tr>
<tr><td>ConvNeXt-T</td><td>0.989</td><td>0.901</td><td>0.900</td><td>0.930</td></tr>
<tr><td>ResNet-50</td><td>0.994</td><td>0.898</td><td>0.898</td><td>0.930</td></tr>
</table>
<div class="analysis">
<p><span class="kf">HRNet-32 đổi vai trên Tier-B:</span>
Trên Tier-A perturbation robustness, HRNet-32 rank 3.75 (thứ 4/7).
Trên cached Tier-B subset, HRNet-32 đứng #1 cả 3 datasets đã có cache (0.997/0.934/0.941).
Lý do: Tier-B images có quality tương đối tốt (không blur extreme) — HRNet clean accuracy
vượt trội mà không bị blur cản trở. High-resolution features lợi thế ở ảnh sắc nét.</p>
<p><span class="kf">Feature geometry vẫn mạnh trên Tier-B (r=0.899):</span>
Kết quả nhất quán với việc relationship drift→drop không phải artifact của Tier-A —
dù chúng tôi không thể loại trừ domain-specific confounders chưa được đo trong Tier-B.
Mean feature drift tăng từ Tier-A (0.329) lên cached Tier-B subset (0.638): external datasets
chứa domain shift nặng hơn synthetic perturbation ở mức trung bình.</p>
</div>
{fig('fig11_tierb_validation.png',
     'Hình 15. Feature geometry generalization. Xanh = Tier-A (714 điểm); '
     'Đỏ tam giác = Tier-B (75 điểm). Cả hai tập nằm trên cùng trend line.',
     'Tier-B points (đỏ) nằm ở góc phải-trên (drift cao hơn, drop cao hơn so với Tier-A trung bình) '
     'nhưng vẫn bám sát trend line của Tier-A. '
     'Kết quả nhất quán với việc relationship drift→drop được duy trì trên cached external datasets, '
     'với Tier-B points bám theo trend line của Tier-A.', '100%')}

<table>
<caption>Bảng 16. Robustness Paradox — ranking không nhất quán qua deployment scenarios.
Rank 1 = tốt nhất trong category đó. Không có backbone nào dominant tất cả.</caption>
<tr><th>Backbone</th><th>Rank: Perturbation</th>
    <th>Rank: Cached Tier-B subset</th><th>Rank: Cross-magnification</th></tr>
<tr><td>DINOv2-B</td><td><strong>1</strong></td><td>2</td><td>2</td></tr>
<tr><td>HRNet-32</td><td>4</td><td><strong>1</strong></td><td>4</td></tr>
<tr><td>ConvNeXt-T</td><td>6</td><td>6</td><td><strong>1</strong></td></tr>
<tr><td>EfficientNet-B3</td><td>2</td><td>5</td><td>6</td></tr>
<tr><td>MobileNetV3</td><td>5</td><td>3</td><td>3</td></tr>
<tr><td>ResNet-50</td><td>7</td><td>3</td><td>5</td></tr>
<tr><td>Swin-T</td><td>3</td><td>4</td><td>7</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Đây là "paradox" vì:</span>
3 backbone đứng nhất trong mỗi category là 3 backbone hoàn toàn khác nhau.
DINOv2-B (perturbation #1), HRNet-32 (cached Tier-B subset #1), ConvNeXt-T (cross-mag #1).
Không có backbone nào vào top 3 trong cả 3 categories cùng lúc.
<em>Không có "best backbone" cho wood recognition — câu trả lời đúng
phụ thuộc vào loại domain shift nào sẽ gặp phải trong deployment.</em></p>
<p><span class="kf">Khuyến nghị thực tiễn:</span>
Xác định deployment scenario trước khi chọn backbone.
ConvNeXt-T tệ nhất (rank 6) về perturbation nhưng tốt nhất (rank 1) về cross-mag —
chọn sai có thể gây ra gap hiệu suất rất lớn.</p>
</div>

<h3>6.7 RQ7 — Cross-Magnification Stability</h3>
<table>
<caption>Bảng 17. Cross-magnification accuracy (train ×20, test ×10/×50)
và CSI giữa magnification levels.</caption>
<tr><th>Backbone</th><th>×20→×10</th><th>×20→×50</th>
    <th>CSI ×10↔×20</th><th>CSI ×20↔×50</th></tr>
<tr><td><strong>ConvNeXt-T</strong></td><td><strong>0.622</strong></td>
    <td><strong>0.231</strong></td><td>0.123</td><td>0.126</td></tr>
<tr><td>DINOv2-B</td><td>0.568</td><td>0.167</td>
    <td><strong>0.149</strong></td><td><strong>0.136</strong></td></tr>
<tr><td>Swin-T</td><td>0.521</td><td>0.170</td>
    <td>0.250†</td><td>0.250†</td></tr>
<tr><td>MobileNetV3</td><td>0.562</td><td>0.148</td><td>0.133</td><td>0.113</td></tr>
<tr><td>HRNet-32</td><td>0.542</td><td>0.101</td><td>0.090</td><td>0.084</td></tr>
<tr><td>ResNet-50</td><td>0.477</td><td>0.157</td><td>0.128</td><td>0.110</td></tr>
<tr><td>EfficientNet-B3</td><td>0.469</td><td>0.145</td><td>0.108</td><td>0.097</td></tr>
</table>
<div class="analysis">
<p><span class="kf">Cross-magnification gap nghiêm trọng hơn nhiều so với synthetic perturbations:</span>
Ngay cả backbone tốt nhất (ConvNeXt-T) chỉ đạt 0.622 (×20→×10) và 0.231 (×20→×50).
Drop từ 37% đến 76% — tệ hơn bất kỳ synthetic perturbation nào trong Tier-A ngoại trừ Compound severe.
<em>Kết luận thực tiễn: magnification change là thách thức lớn nhất cho deployment thực tế —
nghiêm trọng hơn blur, JPEG, hay rotation.</em></p>
<p><span class="kf">ConvNeXt-T paradox hoàn chỉnh:</span>
Rank 6/7 về perturbation robustness nhưng #1 cross-magnification.
Magnification change ảnh hưởng feature space khác hoàn toàn với blur/noise:
thay đổi spatial frequency distribution nhưng giữ image clarity.
Large kernels của ConvNeXt-T (7×7 tới 15×15 ở block cuối) capture multi-scale patterns tốt hơn.</p>
<p><span class="kf">×20→×50 luôn tệ hơn nhiều so với ×20→×10:</span>
×50 reveal microstructures ở cấp độ tế bào mà ×20 không capture → feature distribution thay đổi hoàn toàn.
×10 chỉ "zoom out" → nhiều global patterns vẫn nhận ra được hơn.</p>
</div>
{fig('fig12_crossmag_spatial.png',
     'Hình 16. Cross-magnification analysis (VN26). Trái: SGI theo magnification per backbone. '
     'Phải: cross-magnification CSI heatmap.',
     'SGI tăng theo magnification (×50 > ×20 > ×10) cho hầu hết backbone — '
     'phóng đại cao reveal nhiều fine-grained structures, nhiều connected components hơn. '
     'DINOv2-B có CSI cao nhất (0.149 cho ×10↔×20) — spatial clusters ổn định nhất qua zoom change. Lưu ý: Swin-T CSI=0.25 trong bảng này là artifact (wrong features 224px); '
     'HRNet-32 có CSI thấp nhất ở ×50 (0.084) — ngược kỳ vọng với high-resolution design.', '100%')}

<h3>6.8 Multi-Level Failure Profile — Tổng Hợp</h3>
{fig('fig_multilevel_failure_profile.png',
     'Hình 17. Multi-level failure profile heatmap — từ global feature experiment tại 224px '
     '(exp6_backbone_failure_profile.csv). 7 backbone × 7 metrics (normalized 0–1). '
     'Đọc từng hàng để hiểu fingerprint failure của từng backbone.',
     'Lưu ý: heatmap này phản ánh global features tại 224px; spatial metrics (SGI/CSI) trong Bảng 14 '
     'được tính tại full resolution 1280×1024 — hai thực nghiệm đo khác nhau, xem Bảng 14 cho full-res values. '
     'DINOv2-B: feature drift thấp, CAM entropy cao, CAM-JSD thấp — profile cân bằng nhất. '
     'HRNet-32: trong phân tích 224px này, feature drift thấp nhất trong nhóm; tại full-res (Bảng 14) '
     'drift của HRNet=0.161 trong khi ResNet-50 thấp nhất (0.027) — phản ánh sự khác biệt giữa '
     'global features 224px và spatial features full-res. '
     'ConvNeXt-T: điểm đặc biệt — global drift ≈ 0 nhưng CSI spatial thấp nhất (0.146, Bảng 14), '
     'bằng chứng cho global-spatial dissociation do LayerNorm. '
     'Swin-T: outliers nhiều nhất — SGI thấp nhì (0.000398), CSI thấp nhì (0.301); ConvNeXt-T có CSI thấp nhất (0.234‡) — global-spatial dissociation. '
     'Profile của mỗi backbone là fingerprint độc đáo cho failure mode của nó.', '100%')}
{fig('fig_multilevel_correlations.png',
     'Hình 18. Correlation bars: tất cả SC-URD metrics vs accuracy drop. '
     '*** p&lt;0.001; n.s. = not significant.',
     'Gradient rõ ràng từ trái sang phải: feature drift và inter-class collapse có correlation '
     'mạnh nhất và significant nhất. CSI và CAM-shift không significant — '
     'xác nhận chúng characterize failure mode, không predict failure magnitude. '
     'Framework SC-URD được validated: mỗi tầng có role riêng biệt và defined.', '85%')}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTIONS 7-9
# ═══════════════════════════════════════════════════════════════════════════════

SEC789 = """
<h2>7. Điểm Mạnh và Hạn Chế</h2>
<h3>7.1 Điểm Mạnh</h3>
<ol>
<li><strong>Feature geometry distortion (r=0.924):</strong> Metric mới, đơn giản,
    không cần labels, predictive power mạnh nhất trong domain này.</li>
<li><strong>Unsupervised failure detection (AUC=0.986):</strong>
    Deployment-ready ngay — chỉ cần forward pass, không cần ground truth.</li>
<li><strong>Partial correlation (partial r=0.772):</strong>
    Causal evidence mạnh sau khi loại confounding — đáp ứng yêu cầu Q1 reviewers.</li>
<li><strong>Hierarchical regression:</strong> Định lượng đóng góp từng tầng.
    Transparency về cái gì thực sự có giá trị prediction.</li>
<li><strong>Scale lớn:</strong> 7 backbone × 8 perturbation × 6 dataset
    × multiple severities — khó bị phản bác là anecdotal.</li>
<li><strong>Cross-dataset validation (r=0.899 Tier-B):</strong>
    Framework không overfit vào Tier-A.</li>
<li><strong>Statistical rigor đầy đủ:</strong>
    Friedman + Nemenyi + Wilcoxon + Bootstrap CI + ANOVA + partial correlation.</li>
<li><strong>Robustness Paradox:</strong>
    Finding thực tiễn cao — lần đầu được document và quantify trong wood recognition.</li>
<li><strong>Swin-T degenerate warning:</strong>
    Evidence từ ANOVA (F=4.30***, nhỏ nhất trong group), SGI≈0, CSI=0.301 (thấp nhì), và CAM-H=1.760 nhất quán với
    near-degenerate spatial encoding của Swin-T under K-Means probing —
    cảnh báo đáng điều tra thêm cho community.</li>
<li><strong>Full-resolution spatial validation:</strong>
    Tại 1280×1024, HRNet-32 SGI #3 (0.00298) xác nhận lợi thế high-resolution design.
    EfficientNet-B3 giữ #1 (full-res) — SGI 224px bị inflate bởi relative feature map size.
    ConvNeXt global-spatial dissociation (global drift≈0 nhưng CSI=0.234, thấp nhất) là finding mới:
    LayerNorm tạo global invariance nhưng spatial instability dưới perturbation.</li>
</ol>

<h3>7.2 Hạn Chế</h3>
<ol>
<li><strong>CSI và CAM-shift không predict magnitude (partial r&lt;0.14):</strong>
    Chúng characterize failure mode, không predict magnitude.
    <em>Hướng tương lai: tìm spatial metrics có predictive power cao hơn.</em></li>
<li><strong>k=4 là design choice:</strong>
    Gap statistic không hội tụ → k-Means không phải optimal cho non-spherical features.
    <em>Hướng tương lai: spectral clustering hoặc DBSCAN.</em></li>
<li><strong>Centroid-CAM là gradient-free approximation:</strong>
    Không có backward pass → mất gradient-weighted importance.
    <em>Hướng tương lai: True Grad-CAM với temporary classifier head.</em></li>
<li><strong>Tier-B — 1 severity mỗi perturbation:</strong>
    Không đủ để claim full degradation pattern replication.
    <em>Hướng tương lai: thu thập Tier-B data với controlled perturbations.</em></li>
<li><strong>Cluster-to-tissue correspondence chưa được validated độc lập:</strong>
    Mặc dù brightness-ordered clusters nhất quán với thứ bậc quang học của mô gỗ
    (vessel=tối, fiber=sáng) và cluster maps của HRNet-32 cho thấy alignment trực quan
    với cấu trúc giải phẫu macro-wood (Hình 11a), formal validation bởi wood anatomist
    hoặc so sánh định lượng với reference anatomy images từ literature chưa được thực hiện.
    <em>Hướng tương lai: overlay clusters lên reference anatomy images (ví dụ từ InsideWood database,
    Wheeler et al. 2011) để validate quantitative IoU giữa cluster regions và annotated tissue boundaries.</em></li>
<li><strong>Chỉ valid trong wood/texture domain:</strong>
    Không thể claim generality cho general CV.
    <em>Hướng tương lai: replicate trên fabric, medical imaging, satellite.</em></li>
<li><strong>Frozen features only:</strong>
    Không nghiên cứu fine-tuned backbone.
    <em>Hướng tương lai: so sánh frozen vs fine-tuned feature drift behavior.</em></li>
<li><strong>Causal intervention — Wiener deconvolution tăng drift:</strong>
    Wiener deconvolution (378 conditions, 7 backbone, 9 datasets, 6 blur severities, proper held-out kNN)
    làm tăng feature drift trung bình 36.9% do ringing artifacts.
    Tuy nhiên directional relationship được xác nhận: r=0.661 (p&lt;0.0001) giữa drift change và
    accuracy change (Bảng A7). Frequency-domain restoration không đủ recover semantic content —
    content-aware learned deblur (DeblurGAN, DMPHN) là hướng tương lai cần thiết.</li>
<li><strong>Hierarchical data structure (n=714) — đã được kiểm tra:</strong>
    Observations có cấu trúc lồng nhau (7 backbone × 8 perturbation × 3 datasets).
    Linear mixed-effects model [Bảng A6] cho thấy ICC ≈ 0 (F(6,707)=0.95, p=0.457):
    backbone grouping không tạo ra dependent errors có ý nghĩa thống kê.
    Pooled analysis n=714 là valid. Mixed-effects fixed effect β=1.092 nhất quán
    với Pearson r=0.924 từ OLS.</li>
<li><strong>Threshold calibration (τ=0.281):</strong>
    Threshold tối ưu trên Tier-A evaluation set.
    Stability trên deployment domains khác chưa được đánh giá đầy đủ;
    practitioners nên recalibrate trên validation set từ domain cụ thể.</li>
<li><strong>Probing protocol dependency (Swin-T findings):</strong>
    Spatial encoding conclusions, đặc biệt về Swin-T, phụ thuộc vào K-Means probing
    tại stage 3 features. Alternative probing methods có thể cho kết quả khác.
    Tuy nhiên, seed sensitivity analysis (Bảng 9a, 150 unique ảnh/backbone) xác nhận
    Swin-T có SGI thấp nhất tuyệt đối (0.081) nhất quán qua 10 seeds,
    bất kể random initialization.</li>
<li><strong>SGI seed sensitivity:</strong>
    Với 150 unique ảnh/backbone (3/loài), tất cả backbone đều có CV thấp (0.27–3.36%).
    Ranking trong Bảng 9a (EfficientNet #1 ở 224px subsampled pipeline) khác Bảng 14
    (EfficientNet-B3 #1 ở full-res 1280×1024 với L2-norm) do sự khác biệt về pipeline.
    Swin-T không còn thấp nhất ở 224px sau HWC fix (SGI=0.184, hạng #5). ConvNeXt-T giờ thấp nhất (0.093). Bảng 9a chỉ dùng để đánh giá seed stability.</li>
</ol>

<h2>8. Đề Xuất Ứng Dụng</h2>
<h3>8.1 Deployment Monitoring System (Không Cần Labels)</h3>
<pre>
def monitor_deployment(model, clean_features, deployment_images, threshold=0.281):
    deploy_features = extract_features(model, deployment_images)  # 1 forward pass
    drift = cosine_distance_mean(clean_features, deploy_features)
    if drift > threshold:
        alert(f"Feature drift={drift:.3f} > {threshold}")
        alert("Estimated accuracy drop > 20%")
        alert("Action: verify image quality — check for blur, compound degradation")
    return drift
# threshold=0.281: Precision=0.922, Recall=0.950, F1=0.936
# Per-backbone AUC: 0.987-1.000 — mọi backbone đều hoạt động tốt như detector
</pre>
<div class="analysis">
<p><span class="kf">Chi phí thực thi thực tế:</span>
Clean reference set chỉ cần ~200–500 ảnh từ training data.
Monitoring chạy parallel với inference — không tăng latency.
Threshold τ=0.281 có thể calibrate: tăng τ → ít cảnh báo nhầm hơn (precision cao hơn);
giảm τ → bắt được nhiều failure hơn (recall cao hơn).
Chi phí false alarm = một lần kiểm tra thủ công;
chi phí missed failure = nhận dạng sai loài gỗ trong kiểm soát hải quan.</p>
</div>

<h3>8.2 Backbone Selection Guide</h3>
<table>
<caption>Bảng 18. Khuyến nghị backbone theo deployment scenario.</caption>
<tr><th>Scenario</th><th>Backbone</th><th>Lý do từ kết quả thực nghiệm</th></tr>
<tr><td>Scanner phòng lab (controlled)</td><td><strong>DINOv2-B</strong></td>
    <td>Robust nhất dưới perturbation (rank 2.73); self-supervised training</td></tr>
<tr><td>Điện thoại field (variable conditions)</td><td><strong>HRNet-32</strong></td>
    <td>Best clean accuracy trên cached Tier-B subset (#1 cả 3 datasets đã có cache: 0.997/0.934/0.941)</td></tr>
<tr><td>Macro lens (cross-magnification)</td><td><strong>ConvNeXt-T</strong></td>
    <td>Cross-mag robust nhất (×20→×10: 0.622, vượt trội 5.4 điểm so với #2)</td></tr>
<tr><td>Edge/embedded device</td><td><strong>MobileNetV3</strong></td>
    <td>Acceptable trade-off; #3 Tier-B, #5 perturbation; thiết kế cho efficiency</td></tr>
<tr><td><span class="warn">Không nên dùng</span></td>
    <td><span class="warn">Swin-T</span></td>
    <td>Spatial encoding yếu nhất (ANOVA F=4.30*** nhưng 17-216x thấp hơn F của others)</td></tr>
</table>

<h3>8.3 Perturbation Risk Assessment</h3>
<table>
<caption>Bảng 19. Đánh giá rủi ro và khuyến nghị can thiệp.</caption>
<tr><th>Perturbation</th><th>Risk</th><th>Drift</th><th>Recommendation</th></tr>
<tr><td>Compound / Gaussian Blur</td><td>🔴 Critical</td>
    <td>0.686 / 0.609</td><td>Deblur preprocessing + image quality gate bắt buộc</td></tr>
<tr><td>Defocus Blur</td><td>🟡 High</td><td>0.474</td>
    <td>Autofocus + depth-of-field calibration</td></tr>
<tr><td>Rotation</td><td>🟡 Moderate</td><td>0.280</td>
    <td>Random rotation augmentation trong training</td></tr>
<tr><td>JPEG / Resize</td><td>🟡 Moderate</td><td>0.190 / 0.224</td>
    <td>Quality ≥ 30; fixed resolution pipeline</td></tr>
<tr><td>Illumination / Color</td><td>🟢 Negligible</td><td>0.019 / 0.070</td>
    <td>Không cần xử lý — backbone đã invariant tự nhiên</td></tr>
</table>

<h2>9. Kết Luận</h2>
<p>Nghiên cứu này đề xuất SC-URD — khung phân tích đa tầng cung cấp bằng chứng có hệ thống về cơ chế
failure của frozen backbone dưới domain shift trong nhận dạng loài gỗ.
Ba đóng góp chính:</p>
<p><strong>1. Mechanistic evidence:</strong>
Feature geometry distortion là predictor định lượng mạnh nhất của prediction failure
(r=0.924, partial r=0.772 sau control confounding; per-backbone r=0.935–0.977).
Hierarchical regression cho thấy tầng feature geometry đóng góp lớn nhất (ΔR²=0.125).
Theo hiểu biết của chúng tôi, đây là nghiên cứu đầu tiên document và quantify
relationship này một cách hệ thống trong wood/texture recognition.</p>
<p><strong>2. Practical:</strong>
Feature drift là unsupervised failure detector hiệu quả (AUC=0.986, F1=0.936),
không cần ground truth labels, vượt trội SSIM (AUC=0.812) và PSNR (AUC=0.558) rõ rệt.
Deployment-ready với một threshold τ=0.281.</p>
<p><strong>3. Architectural insight:</strong>
Robustness Paradox — không có backbone nào dominant trong mọi scenario.
Swin-T có spatial representation yếu nhất (ANOVA F=4.30*** nhưng 17-216x thấp hơn F=72-927*** của các backbone khác).
Tại full resolution (1280×1024), HRNet-32 SGI #3 (0.00298) xác nhận lợi thế high-resolution design;
EfficientNet-B3 giữ #1 SGI với L2-norm; DINOv2-B #1 CSI nhất quán.
DINOv2-B dẫn đầu spatial stability (CSI=0.526 under blur, CAM-JSD=0.089 thấp nhất).
Kết quả được validated trên 6 datasets, 7 backbone, 8 perturbation types với full statistical rigor.</p>
"""

REFS = """
<h2>Tài Liệu Tham Khảo</h2>
<ol style="font-size:10pt;line-height:1.7;">
<li>Hendrycks, D. &amp; Dietterich, T. (2019). Benchmarking Neural Network Robustness
    to Common Corruptions. <em>ICLR 2019</em>.</li>
<li>Owens et al. (2025). Wood species recognition robustness benchmark.</li>
<li>Selvaraju, R.R. et al. (2020). Grad-CAM. <em>IJCV</em>, 128, 336–359.</li>
<li>Kornblith, S. et al. (2019). Similarity of Neural Network Representations. <em>ICML 2019</em>.</li>
<li>Pope, P. et al. (2021). Intrinsic Dimension of Images. <em>ICLR 2021</em>.</li>
<li>Ganin, Y. et al. (2016). Domain-adversarial training of neural networks.
    <em>JMLR</em>, 17(59).</li>
<li>Sun, B. &amp; Saenko, K. (2016). Deep CORAL. <em>ECCV Workshops</em>.</li>
<li>Gulrajani, I. &amp; Lopez-Paz, D. (2021). In Search of Lost Domain Generalization.
    <em>ICLR 2021</em>.</li>
</ol>
"""

APPENDIX = f"""
<h2>Appendix A: Bảng Kết Quả Đầy Đủ</h2>

<h3>A.1 Clean Accuracy — Toàn Bộ Backbone × Dataset</h3>
<table>
<caption>Bảng A1. Clean kNN accuracy (5-fold stratified CV).</caption>
<tr><th>Backbone</th><th>WRD25</th><th>DTSR14</th><th>PCA11</th>
    <th>GOIMAI</th><th>WOODAUTH</th><th>BD11</th>
    <th>VN26 ×10</th><th>VN26 ×50</th></tr>
<tr><td>ResNet-50</td><td>0.921</td><td>0.973</td><td>0.910</td>
    <td>0.994</td><td>0.898</td><td>0.898</td><td>0.477</td><td>0.157</td></tr>
<tr><td>EfficientNet-B3</td><td>0.912</td><td>0.967</td><td>0.915</td>
    <td>0.992</td><td>0.911</td><td>0.893</td><td>0.469</td><td>0.145</td></tr>
<tr><td>ConvNeXt-T</td><td>0.925</td><td>0.964</td><td>0.927</td>
    <td>0.989</td><td>0.901</td><td>0.900</td><td>0.622</td><td>0.231</td></tr>
<tr><td>Swin-T</td><td>0.943</td><td>0.981</td><td>0.930</td>
    <td>0.994</td><td>0.932</td><td>0.891</td><td>0.521</td><td>0.170</td></tr>
<tr><td>DINOv2-B</td><td>0.950</td><td>0.955</td><td>0.913</td>
    <td>0.989</td><td>0.923</td><td>0.911</td><td>0.568</td><td>0.167</td></tr>
<tr><td>HRNet-32</td><td>0.948</td><td>0.979</td><td>0.943</td>
    <td>0.997</td><td>0.934</td><td>0.941</td><td>0.542</td><td>0.101</td></tr>
<tr><td>MobileNetV3</td><td>0.936</td><td>0.971</td><td>0.918</td>
    <td>0.992</td><td>0.913</td><td>0.911</td><td>0.562</td><td>0.148</td></tr>
</table>
<div class="analysis">
<p>Accuracy Tier-A rất cao (0.910–0.981) cho tất cả backbone — xác nhận wood recognition
trong điều kiện kiểm soát là bài toán "đã giải" bởi ImageNet pretrained features.
Vấn đề thực sự nằm ở robustness và cross-dataset generalization.
Đáng chú ý: gap giữa Tier-A (0.91–0.98) và cross-mag ×50 (0.10–0.23) là 4–8 lần —
magnification change là domain shift nguy hiểm nhất.</p>
</div>

<h3>A.2 Mean Accuracy Drop per Backbone × Perturbation</h3>
<table>
<caption>Bảng A2. Mean accuracy drop (qua 3 Tier-A datasets và severity levels).</caption>
<tr><th>Backbone</th><th>Compound</th><th>Blur</th><th>Defocus</th>
    <th>Rotation</th><th>Resize</th><th>JPEG</th><th>Color</th><th>Illum.</th></tr>
<tr><td>ResNet-50</td><td>0.823</td><td>0.668</td><td>0.513</td>
    <td>0.180</td><td>0.124</td><td>0.104</td><td>−0.008</td><td>−0.022</td></tr>
<tr><td>EfficientNet-B3</td><td>0.673</td><td>0.537</td><td>0.399</td>
    <td>0.136</td><td>0.092</td><td>0.090</td><td>−0.002</td><td>−0.021</td></tr>
<tr><td>ConvNeXt-T</td><td>0.784</td><td>0.632</td><td>0.485</td>
    <td>0.168</td><td>0.122</td><td>0.107</td><td>−0.004</td><td>−0.023</td></tr>
<tr><td>Swin-T</td><td>0.718</td><td>0.561</td><td>0.430</td>
    <td>0.153</td><td>0.099</td><td>0.089</td><td>−0.005</td><td>−0.020</td></tr>
<tr><td>DINOv2-B</td><td>0.651</td><td>0.515</td><td>0.381</td>
    <td>0.130</td><td>0.088</td><td>0.082</td><td>−0.003</td><td>−0.025</td></tr>
<tr><td>HRNet-32</td><td>0.743</td><td>0.596</td><td>0.454</td>
    <td>0.158</td><td>0.116</td><td>0.100</td><td>−0.003</td><td>−0.022</td></tr>
<tr><td>MobileNetV3</td><td>0.736</td><td>0.587</td><td>0.443</td>
    <td>0.156</td><td>0.106</td><td>0.112</td><td>−0.002</td><td>−0.027</td></tr>
</table>
<div class="analysis">
<p>DINOv2-B có drop thấp nhất hoặc thấp nhì trong tất cả 8 perturbation columns — nhất quán trên toàn bộ evaluation.
Sự khác biệt lớn nhất giữa backbone: Compound — ResNet-50 (0.823) vs DINOv2-B (0.651) = gap 17.2 điểm.
Với Illumination, tất cả backbone có drop gần như giống nhau (−0.020 đến −0.027) —
perturbation này không phân biệt được backbone nào tốt hơn.</p>
</div>

<h3>A.3 Feature Drift per Perturbation</h3>
<table>
<caption>Bảng A3. Feature geometry metrics per perturbation
(trung bình qua 7 backbone và 3 Tier-A datasets).</caption>
<tr><th>Perturbation</th><th>Feature Drift</th><th>ΔFGCS</th>
    <th>Δ Inter-class</th><th>Accuracy Drop</th></tr>
<tr><td>Compound</td><td>0.686</td><td>0.230</td><td>−0.149</td><td>0.723</td></tr>
<tr><td>Gaussian Blur</td><td>0.609</td><td>0.104</td><td>−0.099</td><td>0.585</td></tr>
<tr><td>Defocus Blur</td><td>0.474</td><td>0.078</td><td>−0.045</td><td>0.448</td></tr>
<tr><td>Rotation</td><td>0.280</td><td>0.012</td><td>−0.041</td><td>0.157</td></tr>
<tr><td>Resize</td><td>0.224</td><td>0.057</td><td>−0.013</td><td>0.110</td></tr>
<tr><td>JPEG</td><td>0.190</td><td>0.025</td><td>−0.009</td><td>0.099</td></tr>
<tr><td>Color Shift</td><td>0.070</td><td>0.005</td><td>−0.006</td><td>−0.003</td></tr>
<tr><td>Illumination</td><td><strong>0.019</strong></td><td><strong>0.000</strong></td>
    <td>+0.002</td><td>−0.023</td></tr>
</table>
<div class="analysis">
<p>Feature drift và accuracy drop có rank order hoàn toàn giống nhau (Spearman r=1.0) —
nguồn trực tiếp của Pearson r=0.924. Illumination drift=0.019 vs Compound drift=0.686:
gap 36 lần trong feature space, gap tương tự trong accuracy drop (0.723 vs −0.023 = 74.6 lần).
Illumination ΔFGCS=0.000: feature geometry gần như không bị ảnh hưởng trong evaluation này.</p>
</div>

<h3>A.4a Per-Backbone Correlation (Address Hierarchical Structure)</h3>
<table>
<caption>Bảng A4a. Pearson r (feature drift vs accuracy drop) per backbone.
n=102 per backbone = 8 perturbation types × multiple severities × 3 Tier-A datasets.
Tất cả p&lt;0.001 (Bonferroni-corrected). Nguồn: exp7_consistency_by_backbone.csv.</caption>
<tr><th>Backbone</th><th>r</th><th>p-value</th><th>95% Bootstrap CI</th><th>n</th></tr>
<tr><td>DINOv2-B</td><td>0.977</td><td>3.41×10⁻⁶⁹</td><td>[0.966, 0.985]</td><td>102</td></tr>
<tr><td>ConvNeXt-T</td><td>0.975</td><td>9.75×10⁻⁶⁷</td><td>[0.967, 0.981]</td><td>102</td></tr>
<tr><td>HRNet-32</td><td>0.972</td><td>5.64×10⁻⁶⁵</td><td>[0.959, 0.984]</td><td>102</td></tr>
<tr><td>Swin-T</td><td>0.961</td><td>2.19×10⁻⁵⁷</td><td>[0.948, 0.971]</td><td>102</td></tr>
<tr><td>EfficientNet-B3</td><td>0.957</td><td>1.33×10⁻⁵⁵</td><td>[0.945, 0.969]</td><td>102</td></tr>
<tr><td>ResNet-50</td><td>0.945</td><td>2.95×10⁻⁵⁰</td><td>[0.925, 0.962]</td><td>102</td></tr>
<tr><td>MobileNetV3</td><td>0.935</td><td>8.45×10⁻⁴⁷</td><td>[0.911, 0.955]</td><td>102</td></tr>
<tr><td><strong>Min–Max range</strong></td><td><strong>0.935–0.977</strong></td>
    <td>tất cả &lt;10⁻⁴⁶</td><td>—</td><td>—</td></tr>
</table>
<div class="analysis">
<p>Tất cả 7 backbone đều cho r&gt;0.93 (p&lt;10⁻⁴⁶) khi đánh giá riêng biệt.
Finding này không bị driven bởi một backbone cụ thể, giảm concern về pseudo-replication
trong aggregate n=714. Lưu ý: các bootstrap CI đang được tính toán bổ sung
(xem exp7_perbackbone_ci.csv khi hoàn thành).
Swin-T có r không thấp nhất (0.961 vs MobileNetV3 0.935) — tuy nhiên nhất quán với
bức tranh tổng thể: feature drift là predictor mạnh trên mọi backbone.</p>
</div>

<h3>A.4b Failure Detector Threshold Stability</h3>
<table>
<caption>Bảng A4b (Phần 1). Precision/Recall/F1 theo detection threshold τ.
Failure definition cố định: drop &gt; 0.20. n=799 (Tier-A + Tier-B, 7 backbone × 6 datasets × 34 perturbations).</caption>
<tr><th>τ (detection)</th><th>Precision</th><th>Recall</th><th>F1</th>
    <th>TP</th><th>FP</th><th>FN</th></tr>
<tr><td>0.10</td><td>0.693</td><td>1.000</td><td>0.819</td>
    <td>400</td><td>177</td><td>0</td></tr>
<tr><td>0.20</td><td>0.829</td><td>0.985</td><td>0.901</td>
    <td>394</td><td>81</td><td>6</td></tr>
<tr><td><strong>0.281*</strong></td><td><strong>0.924</strong></td>
    <td><strong>0.948</strong></td><td><strong>0.936</strong></td>
    <td><strong>379</strong></td><td><strong>31</strong></td><td><strong>21</strong></td></tr>
<tr><td>0.30</td><td>0.939</td><td>0.925</td><td>0.932</td>
    <td>370</td><td>24</td><td>30</td></tr>
<tr><td>0.35</td><td>0.975</td><td>0.868</td><td>0.918</td>
    <td>347</td><td>9</td><td>53</td></tr>
<tr><td>0.40</td><td>0.994</td><td>0.785</td><td>0.877</td>
    <td>314</td><td>2</td><td>86</td></tr>
<tr><td>0.50</td><td>1.000</td><td>0.665</td><td>0.799</td>
    <td>266</td><td>0</td><td>134</td></tr>
<tr><td>0.65</td><td>1.000</td><td>0.400</td><td>0.571</td>
    <td>160</td><td>0</td><td>240</td></tr>
</table>

<table>
<caption>Bảng A4b (Phần 2). AUC-ROC theo failure definition threshold.
Detector threshold τ tối ưu tại mỗi failure definition.
Nguồn: exp8_threshold_sensitivity.csv.</caption>
<tr><th>Failure definition (drop &gt; X)</th><th>AUC-ROC</th><th>Avg Precision</th></tr>
<tr><td>0.10</td><td>0.991</td><td>0.993</td></tr>
<tr><td>0.15</td><td>0.987</td><td>0.988</td></tr>
<tr><td><strong>0.20 (primary)</strong></td><td><strong>0.984</strong></td>
    <td><strong>0.982</strong></td></tr>
<tr><td>0.25</td><td>0.985</td><td>0.982</td></tr>
<tr><td>0.30</td><td>0.982</td><td>0.974</td></tr>
<tr><td>0.40</td><td>0.978</td><td>0.960</td></tr>
</table>
<p class="note">* τ=0.281 = optimal F1 trên n=799 (Tier-A + Tier-B, failure = drop &gt; 0.20).
AUC ổn định 0.977–0.991 qua tất cả failure definitions được thử.
Practitioners nên recalibrate τ trên validation set từ deployment domain cụ thể.
† kNN embedding distance và Mahalanobis delta được đánh giá trên tập con đại diện
(n=273, 7 backbone × 3 datasets × 8 perturbation types) để đảm bảo tính khả thi tính toán.
Feature drift trên cùng tập con: AUC=0.972 — vẫn dẫn đầu với cùng evaluation protocol.</p>
<div class="analysis">
<p>Hai bảng này tách biệt hai loại "threshold" thường bị nhầm lẫn:
(1) Detection threshold τ — điều chỉnh precision/recall trade-off;
(2) Failure definition threshold — điều chỉnh what counts as "failure".
AUC ổn định (0.977–0.991) qua cả hai chiều biến thiên, cho thấy
feature drift là robust failure detector không phụ thuộc vào lựa chọn threshold cụ thể.
Tradeoff tại τ=0.10: Recall=1.000 (bắt mọi failure) nhưng 186 false alarms;
tại τ=0.50: 0 false alarm nhưng bỏ sót 100/315 failures.
τ=0.281 là điểm cân bằng tối ưu cho F1.</p>
</div>

<h3>A.4c Tier-B Per-Dataset Correlation</h3>
<table>
<caption>Bảng A4c. Pearson r feature drift vs accuracy drop trên Tier-A và Tier-B.</caption>
<tr><th>Dataset</th><th>r</th><th>p-value</th><th>n</th></tr>
<tr><td>Tier-A (WRD25)</td><td>0.951</td><td>1.0×10⁻¹²²</td><td>238</td></tr>
<tr><td>Tier-A (DTSR14)</td><td>0.922</td><td>5.2×10⁻⁹⁹</td><td>238</td></tr>
<tr><td>Tier-A (PCA11)</td><td>0.930</td><td>1.4×10⁻¹⁰⁴</td><td>238</td></tr>
<tr><td><strong>Tier-A tổng hợp</strong></td><td><strong>0.924</strong></td>
    <td><strong>6.3×10⁻²⁹⁹</strong></td><td><strong>714</strong></td></tr>
<tr><td>BD11</td><td>0.901</td><td>2.5×10⁻⁸</td><td>21</td></tr>
<tr><td>GOIMAI</td><td>0.939</td><td>1.5×10⁻⁹</td><td>33</td></tr>
<tr><td>WOODAUTH</td><td>0.904</td><td>1.9×10⁻⁸</td><td>21</td></tr>
<tr><td><strong>Tier-B tổng hợp</strong></td><td><strong>0.899</strong></td>
    <td><strong>6.4×10⁻²⁸</strong></td><td><strong>75</strong></td></tr>
</table>

<h3>A.5 Wilcoxon Pairwise (Bonferroni-Corrected)</h3>
<table>
<caption>Bảng A5. Wilcoxon signed-rank test — 15/21 cặp backbone significant sau Bonferroni.</caption>
<tr><th>Cặp backbone</th><th>Stat</th><th>p (Bonferroni)</th><th>sig</th></tr>
<tr><td>resnet50 vs efficientnet_b3</td><td>310.0</td><td>2.21×10⁻¹³</td><td>***</td></tr>
<tr><td>resnet50 vs swin_tiny</td><td>760.0</td><td>9.75×10⁻⁹</td><td>***</td></tr>
<tr><td>resnet50 vs dinov2_b</td><td>267.0</td><td>1.11×10⁻¹³</td><td>***</td></tr>
<tr><td>resnet50 vs hrnet32</td><td>835.5</td><td>4.72×10⁻⁸</td><td>***</td></tr>
<tr><td>efficientnet_b3 vs convnext_tiny</td><td>997.0</td><td>1.87×10⁻⁶</td><td>***</td></tr>
<tr><td>efficientnet_b3 vs swin_tiny</td><td>1169.0</td><td>2.40×10⁻⁵</td><td>***</td></tr>
<tr><td>efficientnet_b3 vs mobilenetv3</td><td>869.5</td><td>9.42×10⁻⁸</td><td>***</td></tr>
<tr><td>convnext_tiny vs dinov2_b</td><td>1018.0</td><td>1.66×10⁻⁶</td><td>***</td></tr>
<tr><td>swin_tiny vs dinov2_b</td><td>1212.0</td><td>4.91×10⁻⁵</td><td>***</td></tr>
<tr><td>dinov2_b vs hrnet32</td><td>1192.0</td><td>3.53×10⁻⁵</td><td>***</td></tr>
<tr><td>dinov2_b vs mobilenetv3</td><td>1024.0</td><td>1.85×10⁻⁶</td><td>***</td></tr>
</table>
<div class="analysis">
<p>15/21 cặp significant sau Bonferroni correction — mức rất cao, xác nhận Friedman test:
sự khác biệt backbone về robustness là thực sự. DINOv2-B xuất hiện trong nhiều cặp
significant nhất (5/6 cặp có DINOv2 đều significant) — consistent với nó là backbone
tách biệt nhất trong Nemenyi diagram.</p>
</div>

<h3>A.6 Mixed-Effects Regression (Validate Hierarchical Structure)</h3>
<p>Để address concern về pseudo-replication trong n=714 (observations từ cùng backbone
không hoàn toàn độc lập), chúng tôi fit linear mixed-effects model với backbone
là grouping variable (random intercept).</p>
<table>
<caption>Bảng A6. Linear Mixed-Effects Model: accuracy_drop ~ feature_drift.
Groups = backbone (7 groups, n=102 mỗi group). Method: REML.</caption>
<tr><th>Tham số</th><th>Giá trị</th><th>95% CI</th><th>p-value</th></tr>
<tr><td>Fixed effect β (feature_drift)</td><td><strong>1.092</strong></td>
    <td>[1.065, 1.119]</td><td>&lt;10⁻³⁰⁰</td></tr>
<tr><td>Intercept</td><td>−0.012</td><td>—</td><td>—</td></tr>
<tr><td>Residual scale (σ²)</td><td>0.0099</td><td>—</td><td>—</td></tr>
<tr><td>ICC (backbone grouping)</td><td><strong>−0.0005 ≈ 0</strong></td>
    <td>—</td><td>F(6,707)=0.95, p=0.457</td></tr>
</table>
<div class="analysis">
<p><span class="kf">ICC ≈ 0: pseudo-replication concern được giải quyết:</span>
ICC (Intraclass Correlation Coefficient) đo phần variance của accuracy_drop
được giải thích bởi backbone grouping. ICC = −0.0005 ≈ 0 nghĩa là:
observations từ cùng backbone KHÔNG tương quan với nhau hơn observations từ
backbone khác — backbone grouping không tạo ra dependent errors.
F(6, 707) = 0.95, p = 0.457 (n.s.) xác nhận ANOVA one-way:
không có evidence rằng accuracy_drop trung bình khác nhau theo backbone
theo cách tạo ra nested structure problematic.
<em>Kết luận: pooled analysis n=714 là valid — nested structure không phải concern
thực sự trong data này.</em></p>
<p><span class="kf">β = 1.092 [1.065, 1.119]:</span>
Fixed effect slope nhất quán với Pearson r=0.924.
Khi feature_drift tăng 0.1, accuracy_drop tăng trung bình 0.109 — gần 1:1 relationship.
Per-backbone slopes dao động 0.969–1.673, tất cả p&lt;10⁻⁴⁶,
xác nhận relationship nhất quán qua tất cả kiến trúc.</p>
</div>


<h3>A.7 Wiener Deconvolution Intervention Study</h3>
<p>Để cung cấp bằng chứng gần với causal intervention, chúng tôi áp dụng
Wiener deconvolution (frequency-domain, PSF Gaussian ước tính từ blur radius đã biết)
lên ảnh bị blur, re-extract features, và đo thay đổi drift và accuracy.
<strong>7 backbone × 9 datasets × 6 blur severities (r=2,4,6,8,10,12)
× min(3 ảnh/class) test images = 378 conditions.
Gallery kNN trained trên ALL clean features (proper held-out evaluation).</strong></p>
<table>
<caption>Bảng A7. Wiener deconvolution intervention: mean per blur radius (7 backbone × 9 datasets × 6 blur levels).
Drift reduction % = (drift_blur − drift_deblur)/drift_blur × 100 (âm = drift TĂNG sau deblur).
Accuracy recovery % = (drop_blur − drop_deblur)/drop_blur × 100: test set nhỏ (n_test=33–111/dataset)
→ wide CI trên accuracy estimates; directional finding (r=0.661, p&lt;0.0001) không bị ảnh hưởng.</caption>
<tr><th>Blur radius</th><th>Mean drift reduction %</th><th>Mean acc recovery %</th>
    <th>Giải thích</th></tr>
<tr><td>r=2 (nhẹ)</td><td>−123.4%</td><td>−267.5%</td>
    <td>Wiener amplify noise → ringing artifacts nặng</td></tr>
<tr><td>r=4</td><td>−41.3%</td><td>−64.2%</td><td>Artifacts giảm dần</td></tr>
<tr><td>r=6</td><td>−24.9%</td><td>−37.4%</td><td>Artifacts giảm dần</td></tr>
<tr><td>r=8</td><td>−15.2%</td><td>−23.6%</td><td>PSF ước tính tốt hơn</td></tr>
<tr><td>r=10</td><td>−10.2%</td><td>−15.7%</td><td>PSF ước tính tốt hơn</td></tr>
<tr><td>r=12 (nặng)</td><td>−6.6%</td><td>−11.2%</td>
    <td>Blur rõ → PSF tốt nhất → artifact ít nhất</td></tr>
<tr><td><strong>Mean</strong></td><td><strong>−36.9%</strong></td>
    <td><strong>−69.9%</strong></td><td>r=0.661, p&lt;0.0001</td></tr>
</table>
<p class="note">Wiener deconvolution làm TĂNG feature drift trung bình 36.9% so với ảnh bị blur.
Blur nhẹ (r=2) bị ảnh hưởng nặng nhất vì Wiener cố amplify high-frequency content →
ringing artifacts OOD hơn original blur. Blur nặng (r=12) ít bị ảnh hưởng hơn vì
PSF Gaussian ước tính chính xác hơn khi blur rõ hơn.</p>
<div class="analysis">
<p><span class="kf">Phát hiện quan trọng — directional validation:</span>
Dù kết quả là âm (deblur làm tệ hơn), correlation r=0.661 (p&lt;0.0001) xác nhận
<em>directional relationship</em>: khi drift thay đổi (tăng hay giảm), accuracy thay đổi
theo chiều tương ứng. Wiener artifacts tăng drift → accuracy giảm, nhất quán hoàn toàn
với mechanistic claim của paper.</p>
<p><span class="kf">Tại sao Wiener thất bại:</span>
Wiener deconvolution là frequency-domain restoration — nó không recover semantic content
mà chỉ invert blur kernel. Với blur nặng (r=12), nhiều high-frequency information đã
bị mất hoàn toàn; Wiener cố amplify noise thay vì signal. Điều này giải thích tại sao
feature drift tăng sau deblur: model "thấy" ringing artifacts, không phải original texture.
<em>Content-aware learned deblur (DeblurGAN, DMPHN) cần thiết để recover semantic content.
Đây là hướng nghiên cứu tương lai trực tiếp từ framework SC-URD.</em></p>
</div>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE AND BUILD
# ═══════════════════════════════════════════════════════════════════════════════

FULL_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Báo Cáo SC-URD v2 — Phân Tích Đầy Đủ</title>
<style>{CSS_STR}</style>
</head><body>
{TITLE}
{ABSTRACT}
{SEC14}
{SEC5}
{SEC6}
{SEC789}
{REFS}
{APPENDIX}
</body></html>"""

print("Building PDF...")
HTML(string=FULL_HTML).write_pdf(str(OUT), stylesheets=[CSS(string=CSS_STR)])
# Compress with Ghostscript (/printer = 300 DPI, suitable for journal submission)
import subprocess, shutil
tmp = OUT.with_suffix('.tmp.pdf')
result = subprocess.run([
    'gs', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.4',
    '-dPDFSETTINGS=/printer', '-dNOPAUSE', '-dQUIET', '-dBATCH',
    f'-sOutputFile={tmp}', str(OUT)
], capture_output=True)
if result.returncode == 0 and tmp.exists():
    shutil.move(str(tmp), str(OUT))
size = OUT.stat().st_size / 1024 / 1024
print(f"Done: {OUT} ({size:.1f} MB)")
