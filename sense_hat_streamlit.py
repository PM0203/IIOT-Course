# sense_hat_streamlit_db.py
import json
import time
import re
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Sense HAT Live Dashboard", layout="wide")

# ---------- SETTINGS ----------
REFRESH_SECS = 3  # refresh every few seconds
DB_USER = "postgres"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "postgres"

# ---------- DB connection ----------
@st.cache_resource
def get_engine():
    """Create a SQLAlchemy engine directly (no secrets.toml needed)."""
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, pool_pre_ping=True)

def parse_payload_to_row(payload_text: str) -> dict:
    """Convert the payload column (JSON string) into separate sensor fields."""
    if payload_text is None:
        return {}

    # Try direct JSON parse
    try:
        obj = json.loads(payload_text)
    except Exception:
        try:
            # Sometimes payload may be double-encoded
            p = payload_text.strip()
            if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                obj = json.loads(p[1:-1])
            else:
                obj = {}
        except Exception:
            obj = {}

    if not isinstance(obj, dict):
        return {}

    # Map possible keys
    row = {}
    mappings = {
        "temperature_c": ["temperature_c", "temp_c", "temperature"],
        "humidity_pct": ["humidity_pct", "humidity", "hum_pct"],
        "pressure_hpa": ["pressure_hpa", "pressure", "press_hpa"],
        "yaw_deg": ["yaw_deg", "yaw"],
        "pitch_deg": ["pitch_deg", "pitch"],
        "roll_deg": ["roll_deg", "roll"],
    }

    for canon, keys in mappings.items():
        for k in keys:
            if k in obj:
                row[canon] = obj[k]
                break

    # Handle nested payloads like {"sensor": {...}}
    if not row:
        for v in obj.values():
            if isinstance(v, dict):
                for canon, keys in mappings.items():
                    for k in keys:
                        if k in v:
                            row[canon] = v[k]
                            break

    row["_raw_payload"] = payload_text
    return row

def load_all_from_db() -> pd.DataFrame:
    """Query mqtt_logs and extract sensor readings from payload JSON."""
    sql = text("""
        SELECT id, received_at, inserted_at, payload
        FROM public.mqtt_logs
        ORDER BY id ASC
    """)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    if not rows:
        return pd.DataFrame()

    out = []
    for r in rows:
        rec_id = r[0]
        received_at = r[1]
        inserted_at = r[2]
        ts = received_at or inserted_at or datetime.utcnow()
        payload_text = r[3]
        parsed = parse_payload_to_row(payload_text)
        parsed.update({
            "id": rec_id,
            "ts": ts,
            "payload_raw": payload_text
        })
        out.append(parsed)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)

    numeric_cols = ["temperature_c", "humidity_pct", "pressure_hpa", "yaw_deg", "pitch_deg", "roll_deg"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("ts").reset_index(drop=True)
    return df

# ---------- UI & plotting ----------
st.title("Sense HAT Dashboard (DB-backed)")

df_all = load_all_from_db()

if df_all.empty:
    st.warning("Database empty or no sensor fields parsed from payload.")
    time.sleep(REFRESH_SECS)
    st.rerun()

# ---------- Sidebar filters ----------
min_ts, max_ts = df_all["ts"].min(), df_all["ts"].max()
span = max_ts - min_ts
span_min = max(1, int(span.total_seconds() // 30))

window_options = ["Full recording", "Last 1 min", "Last 2 min", "Last 3 min", "Last 5 min"]
window_options = [opt for opt in window_options
                  if (opt == "Full recording") or (int(re.findall(r"\d+", opt)[0]) <= max(1, span_min))]

st.sidebar.header("Filters")
time_window = st.sidebar.radio("Time window (minutes)", window_options,
                               index=len(window_options) - 1)
smooth = st.sidebar.selectbox("Smoothing (downsample)", ["None", "1s mean", "5s mean"], index=1)

if time_window == "Full recording":
    df = df_all.copy()
else:
    m = int(re.findall(r"\d+", time_window)[0])
    start_time = max_ts - timedelta(minutes=m)
    df = df_all[df_all["ts"] >= start_time].copy()

def apply_resample(df_in):
    if smooth == "None":
        return df_in
    rule = {"1s mean": "1S", "5s mean": "5S"}[smooth]
    out = (df_in.set_index("ts")
                 .resample(rule)
                 .mean(numeric_only=True)
                 .reset_index())
    return out

plot_df = apply_resample(df)

# ---------- top summary ----------
latest = df.iloc[-1]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Temperature (°C)", f"{latest['temperature_c']:.2f}" if pd.notna(latest['temperature_c']) else "n/a")
k2.metric("Humidity (%)", f"{latest['humidity_pct']:.2f}" if pd.notna(latest['humidity_pct']) else "n/a")
k3.metric("Pressure (hPa)", f"{latest['pressure_hpa']:.2f}" if pd.notna(latest['pressure_hpa']) else "n/a")
k4.metric("Time span", f"{(df['ts'].max() - df['ts'].min()).total_seconds() / 60:.1f} min")

# ---------- Charts ----------
left, right = st.columns(2)

# Temperature
with left:
    fig_t = px.line(plot_df, x="ts", y="temperature_c", title="Temperature (°C)",
                    labels={"ts": "time", "temperature_c": "°C"})
    fig_t.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_t, use_container_width=True)

# Humidity
with right:
    fig_h = px.line(plot_df, x="ts", y="humidity_pct", title="Humidity (%)",
                    labels={"ts": "time", "humidity_pct": "%"})
    fig_h.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_h, use_container_width=True)

# Pressure
with left:
    fig_p = px.line(plot_df, x="ts", y="pressure_hpa", title="Pressure (hPa)",
                    labels={"ts": "time", "pressure_hpa": "hPa"})
    fig_p.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_p, use_container_width=True)

# Orientation: yaw/pitch/roll
with right:
    melt = plot_df.melt(id_vars=["ts"],
                        value_vars=["yaw_deg", "pitch_deg", "roll_deg"],
                        var_name="axis", value_name="deg")
    fig_o = px.line(melt, x="ts", y="deg", color="axis",
                    title="Orientation (Yaw / Pitch / Roll, °)",
                    labels={"ts": "time", "deg": "degrees", "axis": ""})
    fig_o.update_layout(legend_title_text="")
    st.plotly_chart(fig_o, use_container_width=True)

# --- rotation matrix (yaw about Z, pitch about Y, roll about X) ---
def rotation_matrix_from_euler(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[ np.cos(y), -np.sin(y), 0],
                   [ np.sin(y),  np.cos(y), 0],
                   [         0,          0, 1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [         0, 1,        0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1,         0,          0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    return Rz @ Ry @ Rx

# --- correct rectangular prism triangles (filled faces) ---
def rectangular_block_mesh(R, L=10.0, D=5.0, H=2.0):
    lx, ly, lz = L/2.0, D/2.0, H/2.0
    verts = np.array([
        [-lx,-ly,-lz],  # 0
        [ lx,-ly,-lz],  # 1
        [ lx, ly,-lz],  # 2
        [-lx, ly,-lz],  # 3
        [-lx,-ly, lz],  # 4
        [ lx,-ly, lz],  # 5
        [ lx, ly, lz],  # 6
        [-lx, ly, lz],  # 7
    ])
    verts_rot = (R @ verts.T).T

    # triangles (two per face) — ordering chosen for outward normals
    triangles = [
        # bottom (z-)
        (0,1,2), (0,2,3),
        # top (z+)
        (4,6,5), (4,7,6),
        # front (y+)
        (3,2,6), (3,6,7),
        # back (y-)
        (0,5,1), (0,4,5),
        # right (x+)
        (1,5,6), (1,6,2),
        # left (x-)
        (0,3,7), (0,7,4),
    ]
    I = [t[0] for t in triangles]
    J = [t[1] for t in triangles]
    K = [t[2] for t in triangles]
    return verts_rot, I, J, K

# --- plot: solid block + static origin arrow showing initial orientation ---
def plot_block_with_initial_arrow(yaw, pitch, roll, init_forward_vec, L=10.0, D=5.0, H=2.0):
    R = rotation_matrix_from_euler(yaw, pitch, roll)
    verts_rot, I, J, K = rectangular_block_mesh(R, L, D, H)

    fig = go.Figure()

    # solid block (opaque)
    fig.add_trace(go.Mesh3d(
        x=verts_rot[:,0], y=verts_rot[:,1], z=verts_rot[:,2],
        i=I, j=J, k=K,
        color="lightgrey",
        opacity=1.0,
        flatshading=True,
        name="block",
        hoverinfo="skip"
    ))

    # Current facing line (from origin to block +X face center)
    front_body = np.array([L/2.0, 0.0, 0.0])
    front_world = (R @ front_body)
    fig.add_trace(go.Scatter3d(
        x=[0, front_world[0]], y=[0, front_world[1]], z=[0, front_world[2]],
        mode="lines",
        line=dict(color="darkred", width=6),
        name="current facing",
        showlegend=False
    ))

    # Static initial-origin arrow (thin) — drawn from origin to initial forward vector
    init_end = init_forward_vec * (max(L,D,H) * 0.6)
    fig.add_trace(go.Scatter3d(
        x=[0, init_end[0]], y=[0, init_end[1]], z=[0, init_end[2]],
        mode="lines+markers",
        line=dict(color="cyan", width=4, dash="dash"),
        marker=dict(size=3),
        name="initial orientation",
        showlegend=True
    ))
    # add a small label
    fig.add_trace(go.Scatter3d(
        x=[init_end[0]], y=[init_end[1]], z=[init_end[2]],
        mode="text", text=["initial"], showlegend=False
    ))

    rng = max(L, D, H) * 1.2
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-rng, rng], title="X"),
            yaxis=dict(range=[-rng, rng], title="Y"),
            zaxis=dict(range=[-rng, rng], title="Z"),
            aspectmode="manual",
            aspectratio=dict(x=L, y=D, z=H),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Block orientation — yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°"
    )
    return fig

# -------------------------
# Prepare plot_df (robust handling)
# -------------------------
df = plot_df.copy()

# convert ts to datetime if present
try:
    df['ts'] = pd.to_datetime(df['ts'])
except Exception:
    pass

df = df.reset_index(drop=True)
for col in ["yaw_deg", "pitch_deg", "roll_deg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = 0.0
df[["yaw_deg","pitch_deg","roll_deg"]] = df[["yaw_deg","pitch_deg","roll_deg"]].fillna(0.0)

n = len(df)
if n == 0:
    st.warning("plot_df is empty.")
else:
    # compute initial forward vector from the first sample (static arrow)
    row0 = df.iloc[0]
    R0 = rotation_matrix_from_euler(float(row0["yaw_deg"]), float(row0["pitch_deg"]), float(row0["roll_deg"]))
    # body +X unit vector in world coords for initial orientation
    init_forward_vec = R0 @ np.array([1.0, 0.0, 0.0])

    slider_key = f"idx_block_filled_{n}"
    idx = st.slider("Choose sample index (time)", 0, n-1, n-1, key=slider_key)
    row = df.iloc[int(idx)]
    yaw, pitch, roll = float(row["yaw_deg"]), float(row["pitch_deg"]), float(row["roll_deg"])
    ts_display = row["ts"] if 'ts' in row.index else idx

    st.markdown(f"**Timestamp:** {ts_display} — yaw {yaw:.1f}°, pitch {pitch:.1f}°, roll {roll:.1f}°")
    fig3d = plot_block_with_initial_arrow(yaw, pitch, roll, init_forward_vec, L=10.0, D=5.0, H=2.0)
    st.plotly_chart(fig3d, use_container_width=True)

# ---------- Auto-refresh ----------
time.sleep(REFRESH_SECS)
st.rerun()
