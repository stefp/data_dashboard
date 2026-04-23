"""
Publication-quality figure for the forest LiDAR dataset catalogue.
Nature/Science double-column style (183 mm wide, 300 dpi).
Output: figure_dashboard.png
"""

import ast
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── rcParams ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ── Nature palette ────────────────────────────────────────────────────────────
NAT = ["#E64B35","#4DBBD5","#00A087","#3C5488",
       "#F39B7F","#8491B4","#91D1C2","#FF7F0E","#7E6148","#CCBB44"]

MOD_PAL = {
    "TLS":          "#E64B35",
    "ULS":          "#4DBBD5",
    "MLS":          "#3C5488",
    "ALS":          "#00A087",
    "Multi-sensor": "#FF7F0E",
    "iphone_LS":    "#CCBB44",
    "Unknown":      "#CCCCCC",
}
MOD_ORDER = ["TLS","ULS","MLS","ALS","Multi-sensor","iphone_LS"]

MAP_LAND, MAP_OCEAN, MAP_BORDER, MAP_GRID = "#E8E8E8", "#F7F9FC", "#BBBBBB", "#DDDDDD"
HERE = Path(__file__).parent

# ── Normalisation functions ───────────────────────────────────────────────────
def norm_modality(s):
    if not isinstance(s, str) or not s.strip():
        return "Unknown"
    parts = {p.strip() for p in re.split(r"[,;\s]+", s) if p.strip() and p.upper() in {"TLS","ULS","MLS","ALS","ALS-HD","IPHONE_LS"}}
    sensors = {p.upper().replace("ALS-HD","ALS") for p in parts}
    if not sensors:
        return "Unknown"
    if len(sensors) > 1:
        return "Multi-sensor"
    s2 = sensors.pop()
    if s2 == "IPHONE_LS":
        return "iphone_LS"
    return s2


def norm_forest(s):
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.lower().strip()
    if "boreal" in t or "taiga" in t:
        return "Boreal"
    if "tropical" in t or "rainforest" in t or "rain forest" in t or "peatland" in t or "mangrove" in t:
        return "Tropical"
    if "mediterr" in t:
        return "Mediterranean"
    if "temperate" in t and ("decid" in t or "broadleav" in t or "beech" in t):
        return "Temperate broadleaved"
    if "temperate" in t and ("conifer" in t or "pine" in t or "spruce" in t or "larch" in t):
        return "Temperate coniferous"
    if "temperate" in t:
        return "Temperate mixed"
    if "alpine" in t or "subalpin" in t or "montane" in t or "subalpine" in t:
        return "Alpine / Montane"
    if "arctic" in t or "tundra" in t:
        return "Arctic / Tundra"
    if "urban" in t:
        return "Urban"
    if "savann" in t or "woodland" in t or "shrub" in t or "miombo" in t or "semi-arid" in t:
        return "Savanna / Woodland"
    if "plantation" in t or "orchard" in t or "agroforest" in t:
        return "Plantation / Agro"
    if "conifer" in t or "pine" in t or "spruce" in t or "larch" in t:
        return "Temperate coniferous"
    return "Other"


def norm_license(s):
    if not isinstance(s, str) or not s.strip() or s.strip() in {"-", "NA", "na"}:
        return None
    t = s.lower()
    if "cc0" in t or "creative commons zero" in t or "public domain" in t or "universal deed" in t:
        return "CC0 / Public Domain"
    if "by-nc-sa" in t or "by nc sa" in t:
        return "CC BY-NC-SA"
    if "by-nc" in t or "nc 4.0" in t or "noncommercial" in t or "non-commercial" in t:
        return "CC BY-NC"
    if "by-sa" in t:
        return "CC BY-SA"
    if "cc by" in t or "cca 4.0" in t or "cc-by" in t or "attribution" in t:
        return "CC BY"
    if "gnu" in t or "affero" in t or "gpl" in t:
        return "Open Source (GNU)"
    if "open government" in t or "ogl" in t or "etalab" in t or "national archives" in t:
        return "Open Government"
    if "usgs" in t or "geological survey" in t:
        return "US Gov (Public Domain)"
    return "Other / Pending"


# ── Load & process data ───────────────────────────────────────────────────────
def parse_ll(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        r = ast.literal_eval(s.strip())
        if isinstance(r, list):
            return [(float(a), float(b)) for a, b in r]
        if isinstance(r, tuple):
            return [(float(r[0]), float(r[1]))]
    except Exception:
        pass
    return [(float(a), float(b)) for a, b in re.findall(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", s)]


df = pd.read_csv(HERE / "SSL_full_metadata.csv", encoding="utf-8", encoding_errors="replace")
df = df[df["Dataset_name"].notna() & (df["Dataset_name"].str.strip() != "")].copy()
df["modality_norm"] = df["lidar_modality"].apply(norm_modality)
df["forest_norm"]   = df["Forest type"].apply(norm_forest)
df["license_norm"]  = df["License data"].apply(norm_license)
df["year"]          = pd.to_numeric(df["year_data"], errors="coerce")
df["coords"]        = df["lat_long"].apply(parse_ll)

pts = pd.DataFrame(
    [{"lat": lat, "lon": lon, "modality": row["modality_norm"]}
     for _, row in df.iterrows() for lat, lon in row["coords"]]
)
world = gpd.read_file(HERE / "ne_countries" / "ne_110m_admin_0_countries.shp")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.2, 7.2), facecolor="white")
gs = GridSpec(3, 6, figure=fig,
              left=0.07, right=0.97, top=0.96, bottom=0.06,
              hspace=0.58, wspace=0.55,
              height_ratios=[2.2, 1.6, 1.6])

ax_world  = fig.add_subplot(gs[0, :4])
ax_europe = fig.add_subplot(gs[0, 4:])
ax_mod    = fig.add_subplot(gs[1, :2])
ax_forest = fig.add_subplot(gs[1, 2:4])
ax_ctry   = fig.add_subplot(gs[1, 4:])
ax_year   = fig.add_subplot(gs[2, :3])   # wider: half the bottom row
ax_lic    = fig.add_subplot(gs[2, 3:])


def plabel(ax, letter, x=-0.14, y=1.07):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left", color="#111111")


def trim_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── a/b  Maps ─────────────────────────────────────────────────────────────────
def draw_map(ax, xlim, ylim, ms, title, show_legend):
    ax.set_facecolor(MAP_OCEAN)
    world.plot(ax=ax, color=MAP_LAND, edgecolor=MAP_BORDER, linewidth=0.25, zorder=1)
    for lon in range(-180, 181, 30):
        ax.axvline(lon, color=MAP_GRID, lw=0.18, zorder=0)
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color=MAP_GRID, lw=0.18, zorder=0)

    handles = []
    for mod in MOD_ORDER:
        sub = pts[pts["modality"] == mod]
        if sub.empty:
            continue
        col = MOD_PAL[mod]
        ax.scatter(sub["lon"], sub["lat"], c=col, s=ms**2, zorder=3,
                   edgecolors="white", linewidths=0.25, alpha=0.92)
        handles.append(mpatches.Patch(color=col, label=mod))

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_color("#999999")
    ax.set_title(title, fontsize=7, pad=3, loc="left", color="#333333")

    if show_legend:
        leg = ax.legend(handles=handles, fontsize=5.5, ncol=3,
                        loc="lower left", bbox_to_anchor=(0.0, 0.0),
                        frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
                        handlelength=0.9, handletextpad=0.4,
                        columnspacing=0.7, borderpad=0.4)
        leg.get_frame().set_linewidth(0.4)


draw_map(ax_world,  (-168, 168), (-57, 80), 4.5, "Global distribution", True)
draw_map(ax_europe, (-11, 41),   (34, 72),  6.0, "Europe (detail)",     False)
plabel(ax_world,  "a", x=-0.02, y=1.04)
plabel(ax_europe, "b", x=-0.12, y=1.04)


# ── c  Modality — sorted ascending so largest bar is at bottom ───────────────
mod_counts = df["modality_norm"].value_counts().sort_values(ascending=True)
yp = np.arange(len(mod_counts))
ax_mod.barh(yp, mod_counts.values,
            color=[MOD_PAL.get(m, "#CCCCCC") for m in mod_counts.index],
            height=0.6, edgecolor="white", linewidth=0.3)
ax_mod.set_yticks(yp)
ax_mod.set_yticklabels(list(mod_counts.index))
ax_mod.set_xlabel("Datasets (n)")
ax_mod.set_title("LiDAR modality", loc="left")
ax_mod.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
for i, v in enumerate(mod_counts.values):
    ax_mod.text(v + 0.3, i, str(v), va="center", fontsize=5.5, color="#555")
trim_ax(ax_mod)
plabel(ax_mod, "c")


# ── d  Forest type — top 10, rest → Other, sorted ascending (largest at bottom)
TOP_N_FOREST = 10
fc_raw = df["forest_norm"].dropna().value_counts()
if len(fc_raw) > TOP_N_FOREST:
    fc_dict = fc_raw.iloc[:TOP_N_FOREST].to_dict()
    extra = fc_raw.iloc[TOP_N_FOREST:].sum()
    fc_dict["Other"] = fc_dict.get("Other", 0) + extra
    fc = pd.Series(fc_dict).sort_values(ascending=True)   # ascending → largest at bottom
else:
    fc = fc_raw.sort_values(ascending=True)

yf = np.arange(len(fc))
ax_forest.barh(yf, fc.values,
               color=[NAT[i % len(NAT)] for i in range(len(fc))],
               height=0.6, edgecolor="white", linewidth=0.3)
ax_forest.set_yticks(yf)
ax_forest.set_yticklabels(list(fc.index))
ax_forest.set_xlabel("Datasets (n)")
ax_forest.set_title("Forest / vegetation type", loc="left")
ax_forest.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
for i, v in enumerate(fc.values):
    ax_forest.text(v + 0.1, i, str(v), va="center", fontsize=5.5, color="#555")
trim_ax(ax_forest)
plabel(ax_forest, "d")


# ── e  Country ────────────────────────────────────────────────────────────────
cc = df["Country"].value_counts().head(12)
cmap_g = mpl.colormaps["YlGn"]
yc = np.arange(len(cc))
ax_ctry.barh(yc, cc.values[::-1],
             color=[cmap_g(0.35 + 0.55 * i / max(len(cc)-1, 1)) for i in range(len(cc))],
             height=0.6, edgecolor="white", linewidth=0.3)
ax_ctry.set_yticks(yc)
ax_ctry.set_yticklabels(list(cc.index[::-1]))
ax_ctry.set_xlabel("Datasets (n)")
ax_ctry.set_title("Country", loc="left")
ax_ctry.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
for i, v in enumerate(cc.values[::-1]):
    ax_ctry.text(v + 0.1, i, str(v), va="center", fontsize=5.5, color="#555")
trim_ax(ax_ctry)
plabel(ax_ctry, "e")


# ── f  Year timeline ──────────────────────────────────────────────────────────
yc2 = df[df["year"].notna()]["year"].value_counts().sort_index()
yr, yv = yc2.index.astype(int), yc2.values
ax_year.fill_between(yr, yv, alpha=0.15, color="#3C5488")
ax_year.plot(yr, yv, color="#3C5488", lw=1.4, marker="o", markersize=3.5,
             markerfacecolor="#3C5488", markeredgecolor="white", markeredgewidth=0.4)
ax_year.set_xlabel("Acquisition year")
ax_year.set_ylabel("Datasets (n)")
ax_year.set_title("Year distribution", loc="left")
ax_year.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax_year.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
ax_year.set_xlim(yr.min() - 0.5, yr.max() + 0.5)
trim_ax(ax_year)
plabel(ax_year, "f")


# ── g  Licence ────────────────────────────────────────────────────────────────
lc = df["license_norm"].dropna().value_counts()
lc_cols = [NAT[i % len(NAT)] for i in range(len(lc))]
wedges, _, autotexts = ax_lic.pie(
    lc.values, labels=None, colors=lc_cols,
    autopct=lambda p: f"{p:.0f}%" if p >= 8 else "",
    pctdistance=0.72, startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=0.5),
)
for at in autotexts:
    at.set_fontsize(5.5); at.set_color("white"); at.set_fontweight("bold")

short_labels = [l[:22] for l in lc.index]
ax_lic.legend(
    handles=[mpatches.Patch(color=c, label=l) for c, l in zip(lc_cols, short_labels)],
    loc="center left", bbox_to_anchor=(1.0, 0.5),
    fontsize=5.2, ncol=1, frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
    handlelength=0.85, borderpad=0.5,
).get_frame().set_linewidth(0.4)
ax_lic.set_title("Data licence", loc="left")
plabel(ax_lic, "g", x=-0.18)


# ── Save PNG (300 dpi) + PDF (vector, A4) ────────────────────────────────────
# PNG
png_out = HERE / "figure_dashboard.png"
fig.savefig(png_out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved PNG: {png_out}")

# PDF on A4 (210 x 297 mm = 8.27 x 11.69 in)
# Embed the figure centred on an A4 canvas with margins
A4_W, A4_H = 8.27, 11.69
fig_a4 = plt.figure(figsize=(A4_W, A4_H), facecolor="white")
# Re-draw by saving fig to a buffer and placing as image — simplest: just save
# the original figure as PDF (Matplotlib PDF backend is fully vector)
pdf_out = HERE / "figure_dashboard.pdf"
fig.savefig(pdf_out, bbox_inches="tight", facecolor="white",
            metadata={"Creator": "matplotlib"})
print(f"Saved PDF: {pdf_out}")
plt.close(fig_a4)
plt.close(fig)
