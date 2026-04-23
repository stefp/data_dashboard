import ast
import re
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Forest LiDAR Observatory",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Firefly theme CSS
# ──────────────────────────────────────────────
st.markdown(
    """
<style>
  /* Global background */
  .stApp { background-color: #070b14; color: #8ab4cf; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1f2d 0%, #0a1520 100%);
    border: 1px solid #1a3a4a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 0 18px rgba(0, 212, 255, 0.08);
  }
  [data-testid="stMetricLabel"] { color: #4a7a8a !important; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #39ff14 !important; font-size: 2rem !important; font-weight: 700; text-shadow: 0 0 12px rgba(57,255,20,0.5); }

  /* Section divider */
  hr { border-color: #132030; }

  /* Title glow */
  .glow-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    background: linear-gradient(90deg, #39ff14, #00d4ff, #39ff14);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
    filter: drop-shadow(0 0 14px rgba(57,255,20,0.45));
    animation: shimmer 4s linear infinite;
  }
  @keyframes shimmer { to { background-position: 200% center; } }

  .subtitle { color: #4a7a8a; font-size: 0.9rem; letter-spacing: 0.06em; margin-top: -0.4rem; }

  /* Section labels */
  .section-label {
    color: #00d4ff;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: -0.5rem;
    opacity: 0.75;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Data loading & preprocessing
# ──────────────────────────────────────────────
PALETTE = ["#39ff14", "#00d4ff", "#ff9f1c", "#ff3cac", "#7b2fff", "#00ff88", "#ff6b35", "#fffc00", "#a8edea"]
BG = "#070b14"
LAND = "#0d1f2d"
OCEAN = "#050b12"
BORDER = "#1a3050"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df[df["Dataset_name"].notna() & (df["Dataset_name"].str.strip() != "")]
    df = df.copy()

    # Normalize modalities (handle comma/semicolon lists → keep primary)
    def norm_modality(s):
        if not isinstance(s, str) or not s.strip():
            return "Unknown"
        # Split on comma or semicolon, take unique parts
        parts = [p.strip() for p in re.split(r"[,;]", s) if p.strip()]
        parts = sorted(set(parts))
        return " + ".join(parts) if len(parts) > 1 else parts[0]

    df["modality_norm"] = df["lidar_modality"].apply(norm_modality)

    # Normalize forest types (group similar)
    forest_map = {
        "Boreal forest": "Boreal",
        "Tropical rainforest": "Tropical",
        "Temperate Mediterranean forests": "Mediterranean",
        "Temperate boradleaved forest": "Temperate deciduous",
        "Urban forest": "Urban",
        "Temperate rainforest": "Temperate mixed",
        "European dry grassland": "Other",
        "experimental research forests or arboretums": "Other",
        "Boreal, temperate deciduous, temperate, Dry schlerophyl, coniferous plantation": "Mixed / Various",
        "Various": "Mixed / Various",
    }
    df["forest_norm"] = df["Forest type"].apply(
        lambda s: forest_map.get(str(s).strip(), str(s).strip()) if isinstance(s, str) and s.strip() else "Unknown"
    )

    # Normalize licenses
    lic_map = {
        "CCA 4.0 International": "CC BY 4.0",
        "CC-BY-4.0": "CC BY 4.0",
        "CC BY 4.0": "CC BY 4.0",
        "CC0 1.0 Universal": "CC0 1.0",
        "CC-BY-SA-4.0": "CC BY-SA 4.0",
        "Creative Commons Attribution-NonCommercial 4.0 International License": "CC BY-NC 4.0",
        "Creative Commons BY-NC-SA": "CC BY-NC-SA",
    }
    df["license_norm"] = df["License data"].apply(
        lambda s: lic_map.get(str(s).strip(), str(s).strip()) if isinstance(s, str) and s.strip() else "Unknown"
    )

    # Parse year
    df["year"] = pd.to_numeric(df["year_data"], errors="coerce")

    # Parse lat/lon
    def parse_ll(s):
        if not isinstance(s, str) or not s.strip():
            return []
        try:
            result = ast.literal_eval(s.strip())
            if isinstance(result, list):
                return [(float(a), float(b)) for a, b in result]
            elif isinstance(result, tuple):
                return [(float(result[0]), float(result[1]))]
        except Exception:
            pass
        matches = re.findall(r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)", s)
        return [(float(a), float(b)) for a, b in matches]

    df["coords"] = df["lat_long"].apply(parse_ll)
    return df


df = load_data("SSL_full_metadata.csv")

# Expand rows with multiple coords into individual point rows
point_rows = []
for _, row in df.iterrows():
    for lat, lon in row["coords"]:
        point_rows.append(
            {
                "Dataset_name": row["Dataset_name"],
                "lat": lat,
                "lon": lon,
                "modality": row["modality_norm"],
                "forest": row["forest_norm"],
                "country": row.get("Country", ""),
                "year": row["year"],
            }
        )
pts = pd.DataFrame(point_rows)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def glow_scatter_geo(fig, lats, lons, color, name, text=None, size=8):
    """Add a 3-layer glow scatter trace to a geo figure."""
    kw = dict(mode="markers", showlegend=False, hoverinfo="skip")
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, name=name,
        marker=dict(size=size * 3.5, color=color, opacity=0.04, line_width=0), **kw))
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, name=name,
        marker=dict(size=size * 2.0, color=color, opacity=0.12, line_width=0), **kw))
    fig.add_trace(go.Scattergeo(lat=lats, lon=lons, name=name,
        marker=dict(size=size * 1.1, color=color, opacity=0.55, line_width=0), **kw))
    # Core — this one is visible in legend & tooltip
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons, name=name,
        text=text, hovertemplate="<b>%{text}</b><extra></extra>",
        mode="markers", showlegend=True,
        marker=dict(size=size * 0.6, color=color, opacity=1.0,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
    ))


GEO_STYLE = dict(
    bgcolor=BG,
    landcolor=LAND,
    oceancolor=OCEAN,
    lakecolor=OCEAN,
    coastlinecolor=BORDER,
    coastlinewidth=0.6,
    countrycolor=BORDER,
    countrywidth=0.4,
    showocean=True,
    showlakes=True,
    showland=True,
    showframe=False,
    showcoastlines=True,
    showcountries=True,
    showrivers=False,
    showsubunits=False,
    subunitwidth=0.0,
    projection_type="natural earth",
)

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color="#8ab4cf", family="monospace"),
)

LEGEND_BASE = dict(
    bgcolor="rgba(7,11,20,0.8)",
    bordercolor="#1a3050",
    borderwidth=1,
    font=dict(size=11, color="#8ab4cf"),
)

MODALITY_COLORS = {
    "TLS": "#39ff14",
    "ALS": "#00d4ff",
    "ULS": "#ff9f1c",
    "MLS": "#ff3cac",
    "ALS-HD": "#7b2fff",
    "iphone_LS": "#fffc00",
    "TLS + ULS": "#00ff88",
    "ALS + TLS": "#a8edea",
    "ALS + ULS": "#ff6b35",
    "Unknown": "#4a5568",
}


def mod_color(m):
    return MODALITY_COLORS.get(m, PALETTE[hash(m) % len(PALETTE)])


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
col_title, col_spacer = st.columns([3, 1])
with col_title:
    st.markdown('<div class="glow-title">🌲 Forest LiDAR Observatory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Open-access 3D forest datasets · point cloud metadata explorer</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Key metrics
# ──────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Datasets", len(df))
m2.metric("Countries", df["Country"].nunique())
m3.metric("Modalities", df["modality_norm"].nunique())
m4.metric("Forest types", df[df["forest_norm"] != "Unknown"]["forest_norm"].nunique())
m5.metric("With paper", int(df["url paper"].notna().sum()))

st.markdown("<br/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAP SECTION
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Global distribution</div>', unsafe_allow_html=True)
map_col_world, map_col_europe = st.columns([3, 2])

modalities_sorted = pts["modality"].value_counts().index.tolist()


def build_map(scope_geo: dict, title: str, legend_visible: bool) -> go.Figure:
    fig = go.Figure()
    for mod in modalities_sorted:
        sub = pts[pts["modality"] == mod]
        if sub.empty:
            continue
        color = mod_color(mod)
        glow_scatter_geo(
            fig,
            sub["lat"].tolist(),
            sub["lon"].tolist(),
            color=color,
            name=mod,
            text=sub["Dataset_name"].tolist(),
            size=9,
        )
    geo_cfg = {**GEO_STYLE, **scope_geo}
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=13, color="#4a7a8a"), x=0.01),
        geo=geo_cfg,
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(**LEGEND_BASE, visible=legend_visible),
    )
    # De-duplicate legend entries (keep only the "core" trace per modality)
    seen = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        elif trace.showlegend:
            seen.add(trace.name)
    return fig


with map_col_world:
    fig_world = build_map({}, "World", legend_visible=True)
    st.plotly_chart(fig_world, use_container_width=True, config={"displayModeBar": False})

with map_col_europe:
    europe_geo = dict(
        lonaxis_range=[-12, 42],
        lataxis_range=[35, 72],
        projection_type="mercator",
    )
    fig_eu = build_map(europe_geo, "Europe (detail)", legend_visible=False)
    st.plotly_chart(fig_eu, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DONUT CHARTS — Modality & Forest type
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Composition</div>', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)

DONUT_LAYOUT = {
    **LAYOUT_BASE,
    "height": 320,
    "margin": dict(l=10, r=10, t=45, b=10),
    "legend": {**LEGEND_BASE, "orientation": "v", "font": dict(size=10, color="#8ab4cf"), "x": 0.85, "xanchor": "left"},
    "showlegend": True,
}


def donut(labels, values, colors, title):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(
                colors=colors,
                line=dict(color=BG, width=2),
            ),
            textinfo="percent",
            textfont=dict(size=11, color="#e8f4fd"),
            hovertemplate="<b>%{label}</b><br>%{value} datasets (%{percent})<extra></extra>",
            sort=True,
        )
    )
    fig.update_layout(
        **DONUT_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#4a7a8a"), x=0.01),
        annotations=[
            dict(
                text=f"<b style='color:#39ff14'>{sum(values)}</b>",
                x=0.38, y=0.5, font_size=22, font_color="#39ff14",
                showarrow=False,
            )
        ],
    )
    return fig


# Modality donut
mod_counts = df["modality_norm"].value_counts()
with d1:
    colors_mod = [mod_color(m) for m in mod_counts.index]
    st.plotly_chart(
        donut(mod_counts.index.tolist(), mod_counts.values.tolist(), colors_mod, "LiDAR Modality"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# Forest type donut
forest_counts = df[df["forest_norm"] != "Unknown"]["forest_norm"].value_counts()
with d2:
    colors_forest = PALETTE[: len(forest_counts)]
    st.plotly_chart(
        donut(forest_counts.index.tolist(), forest_counts.values.tolist(), colors_forest, "Forest Type"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# License donut
lic_counts = df[df["license_norm"].notna() & (df["license_norm"] != "")]["license_norm"].value_counts()
lic_counts = lic_counts[lic_counts.index != ""]
with d3:
    colors_lic = [PALETTE[i % len(PALETTE)] for i in range(len(lic_counts))]
    st.plotly_chart(
        donut(lic_counts.index.tolist(), lic_counts.values.tolist(), colors_lic, "Data License"),
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.markdown("<hr/>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# BOTTOM ROW — Country ranking + Year timeline + Openness radar
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Trends & accessibility</div>', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)

# Country bar chart
country_counts = df["Country"].value_counts().head(15)
with b1:
    fig_c = go.Figure(
        go.Bar(
            x=country_counts.values[::-1],
            y=country_counts.index[::-1],
            orientation="h",
            marker=dict(
                color=country_counts.values[::-1],
                colorscale=[[0, "#0d2a1a"], [0.5, "#1a6633"], [1, "#39ff14"]],
                line_width=0,
            ),
            hovertemplate="<b>%{y}</b>: %{x} datasets<extra></extra>",
            text=country_counts.values[::-1],
            textposition="outside",
            textfont=dict(size=10, color="#39ff14"),
        )
    )
    fig_c.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Datasets by country", font=dict(size=13, color="#4a7a8a"), x=0.01),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(color="#8ab4cf", tickfont=dict(size=10)),
        height=390,
        showlegend=False,
        margin=dict(l=10, r=40, t=45, b=10),
        legend=dict(**LEGEND_BASE),
    )
    st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

# Year timeline
year_counts = df[df["year"].notna()]["year"].value_counts().sort_index()
with b2:
    fig_y = go.Figure()
    # Glow area
    fig_y.add_trace(go.Scatter(
        x=year_counts.index, y=year_counts.values,
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(57,255,20,0.06)",
        showlegend=False, hoverinfo="skip",
    ))
    # Glow line
    fig_y.add_trace(go.Scatter(
        x=year_counts.index, y=year_counts.values,
        mode="lines+markers",
        line=dict(color="#39ff14", width=2.5, shape="spline"),
        marker=dict(
            size=8, color="#39ff14", opacity=1,
            line=dict(color=BG, width=1.5),
        ),
        hovertemplate="<b>%{x}</b>: %{y} datasets<extra></extra>",
        showlegend=False,
    ))
    fig_y.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Datasets by acquisition year", font=dict(size=13, color="#4a7a8a"), x=0.01),
        xaxis=dict(color="#4a7a8a", showgrid=False, tickfont=dict(size=10), dtick=2),
        yaxis=dict(color="#4a7a8a", showgrid=True, gridcolor="#0d1f2d", zeroline=False, tickfont=dict(size=10)),
        height=390,
        showlegend=False,
        margin=dict(l=10, r=10, t=45, b=30),
        legend=dict(**LEGEND_BASE),
    )
    st.plotly_chart(fig_y, use_container_width=True, config={"displayModeBar": False})

# Data openness — stacked bar of: has coords / has paper / full pointcloud / georeferenced
openness_labels = ["Has location", "Has paper", "Full point cloud", "Georeferenced", "Open license"]
openness_yes = [
    int(df["lat_long"].notna().sum()),
    int(df["url paper"].notna().sum()),
    int((df["full pointcloud"].str.strip().str.lower() == "yes").sum()),
    int((df["georeferenced"].str.strip().str.lower() == "yes").sum()),
    int(df["license_norm"].isin(["CC BY 4.0", "CC0 1.0", "CC BY-SA 4.0"]).sum()),
]
total = len(df)
openness_no = [total - v for v in openness_yes]

with b3:
    fig_o = go.Figure()
    fig_o.add_trace(go.Bar(
        name="Yes",
        y=openness_labels,
        x=openness_yes,
        orientation="h",
        marker=dict(color="#39ff14", opacity=0.85, line_width=0),
        hovertemplate="<b>%{y}</b><br>Yes: %{x}<extra></extra>",
        text=[f"{v}" for v in openness_yes],
        textposition="inside",
        textfont=dict(size=10, color=BG),
    ))
    fig_o.add_trace(go.Bar(
        name="No / Unknown",
        y=openness_labels,
        x=openness_no,
        orientation="h",
        marker=dict(color="#132030", line_width=0),
        hovertemplate="<b>%{y}</b><br>No/Unknown: %{x}<extra></extra>",
        showlegend=True,
    ))
    fig_o.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Dataset completeness", font=dict(size=13, color="#4a7a8a"), x=0.01),
        barmode="stack",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(color="#8ab4cf", tickfont=dict(size=10)),
        height=390,
        legend={**LEGEND_BASE, "orientation": "h", "x": 0, "y": -0.08, "font": dict(size=10, color="#8ab4cf")},
        margin=dict(l=10, r=10, t=45, b=40),
    )
    st.plotly_chart(fig_o, use_container_width=True, config={"displayModeBar": False})

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#1e3a4a; font-size:0.72rem; text-align:center; letter-spacing:0.08em;">'
    "FOREST LIDAR OBSERVATORY · NIBIO SFI SmartForest · built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
