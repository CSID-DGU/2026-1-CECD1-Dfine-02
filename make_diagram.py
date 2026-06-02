"""
make_diagram.py — 멘토 스캐폴딩 파이프라인 도식 (PPT 삽입용 PNG/SVG 생성)
실행: python make_diagram.py
출력: pipeline_diagram.png (2560x1200), pipeline_diagram_mpl.svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

W, H = 1280, 600
# figsize를 데이터단위/72로 맞춰 '데이터 1단위 = 1pt' → 폰트 size가 SVG px처럼 동작
fig, ax = plt.subplots(figsize=(W / 72, H / 72))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.set_position([0, 0, 1, 1])

def Y(y):  # SVG 좌표(위가 0) → matplotlib 좌표(아래가 0)
    return H - y

def box(x, y, w, h, fc, ec, rounding=12, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (x, Y(y + h)), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=1))

def txt(x, y, s, size, color="#2b3542", weight="normal", ha="center"):
    ax.text(x, Y(y), s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center")

def arrow(x1, y1, x2, y2, color="#5b6470", lw=2.5):
    ax.add_patch(FancyArrowPatch((x1, Y(y1)), (x2, Y(y2)),
        arrowstyle="-|>", mutation_scale=18, color=color, linewidth=lw,
        shrinkA=0, shrinkB=0))

def dash(x1, y1, x2, y2, color="#9cbef0", lw=2):
    ax.plot([x1, x2], [Y(y1), Y(y2)], color=color, lw=lw, ls=(0, (5, 4)))

# ── 제목 ──
txt(40, 40, "멘토 스캐폴딩 파이프라인", 24, "#1f2733", "bold", ha="left")
txt(40, 74, "유저 상황 → 멘토 3인 선택 → 병렬 응답 생성 → 다양성 평가 → 결과", 12.5, "#7a828d", ha="left")

# ── 메인 단계 박스 ──
box(40, 130, 170, 78, "#eef1f5", "#c9d0d9")
txt(125, 160, "유저 상황", 16, "#2b3542", "bold")
txt(125, 186, "카테고리·태그·상품", 11.5, "#7a828d")

box(290, 130, 200, 78, "#e7f0ff", "#9cbef0")
txt(390, 160, "① 멘토 선택", 16, "#1f4e9b", "bold")
txt(390, 186, "select_mentors.py", 11.5, "#5577aa")

box(570, 130, 200, 78, "#e6f6ec", "#94d3ab")
txt(670, 156, "② 응답 생성", 16, "#1d7a45", "bold")
txt(670, 179, "병렬 LLM 호출", 11.5, "#3f9d68")
txt(670, 197, "generate_responses.py", 10.5, "#5aa97d")

box(850, 130, 200, 78, "#f1ebfb", "#c3a7ec")
txt(950, 156, "③ 다양성 평가", 16, "#6a35b8", "bold")
txt(950, 179, "KR-SBERT 코사인 거리", 11.5, "#8a5fc9")
txt(950, 197, "임계값 0.25", 10.5, "#9d78d4")

box(1130, 130, 120, 78, "#2b3542", "#2b3542")
txt(1190, 160, "결과", 16, "#ffffff", "bold")
txt(1190, 186, "멘토3 + 응답", 10.5, "#c9d0d9")

# ── 메인 화살표 ──
arrow(214, 169, 286, 169)
arrow(494, 169, 566, 169)
arrow(774, 169, 846, 169)
arrow(1054, 169, 1126, 169)

# ── 멘토 선택 상세 (최애/맥락/반대) ──
dash(390, 208, 390, 262)
dash(200, 262, 580, 262)
dash(200, 262, 200, 300); dash(390, 262, 390, 300); dash(580, 262, 580, 300)

box(105, 300, 190, 96, "#ffe9ef", "#f3a7bd")
txt(200, 330, "최애", 18, "#c2335c", "bold")
txt(200, 358, "평소 가장 좋아하는 멘토", 11.5, "#9a5570")
txt(200, 378, "온보딩 퀴즈로 고정", 11, "#b56a86")

box(295, 300, 190, 96, "#e7f0ff", "#9cbef0")
txt(390, 330, "맥락", 18, "#1f4e9b", "bold")
txt(390, 358, "상황에 맞는 멘토", 11.5, "#4a6ba3")
txt(390, 378, "벡터 의미검색 (FAISS ANN)", 11, "#5577aa")

box(485, 300, 190, 96, "#fff2e2", "#f0c089")
txt(580, 330, "반대", 18, "#b9711a", "bold")
txt(580, 358, "일부러 결이 다른 멘토", 11.5, "#9c6a2c")
txt(580, 378, "정합점수 최저 아키타입", 11, "#b5832f")

box(105, 424, 570, 40, "#f7f8fa", "#dde2e8", rounding=10, lw=1)
txt(390, 446, "★ 3인이 서로 다른 소비 Type(1~5)을 갖도록 강제 → 응답 다양성 보장", 12, "#5b6470")

# ── 다양성 평가 상세 ──
dash(950, 208, 950, 300, color="#c3a7ec")
box(790, 300, 320, 120, "#faf7ff", "#d9c6f2")
txt(950, 330, "측정 지표", 13.5, "#6a35b8", "bold")
txt(805, 360, "· 다양성: 쌍별 코사인 거리 평균", 11.5, "#4b5260", ha="left")
txt(805, 384, "· 통과율 100% · Type 3종 100%", 11.5, "#4b5260", ha="left")
txt(805, 408, "· 응답 시간(선택/생성/전체)", 11.5, "#4b5260", ha="left")

# ── 푸터 ──
txt(40, 556, "최애 = 온보딩 고정 · 맥락 = 의미검색 · 반대 = 인지적 자극   |   3인 병렬 호출로 체감 응답시간 단축",
    11.5, "#a7adb6", ha="left")

fig.savefig("pipeline_diagram.png", dpi=150, facecolor="white")
fig.savefig("pipeline_diagram_mpl.svg", facecolor="white")
print("saved: pipeline_diagram.png, pipeline_diagram_mpl.svg")
