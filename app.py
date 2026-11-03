import io
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(
    page_title="Ferrstron | Motor Test Intelligence",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
<style>
.main {background:#f6f8fb;}
.block-container {padding-top:1.4rem; max-width:1450px;}
.hero {padding:24px 28px; border-radius:16px; color:white; background:linear-gradient(115deg,#10243a,#176b87);}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:8px 0 0 0; opacity:.9;}
.small-note {color:#637083; font-size:.85rem;}

div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce4ec;
    padding: 14px;
    border-radius: 12px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #334155 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #10243a !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #176b87 !important;
}
</style>
<div class="hero">
<h1>⚙️ Motor Test Intelligence</h1>
<p>Engineering analytics prototype · motor performance, test comparison and report generation</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Independent demonstration prototype. Not affiliated with Ferrstron and not "
    "validated for production engineering decisions."
)

st.sidebar.header("Workspace")
st.sidebar.write(
    "Upload one or two motor test datasets, or start with the included "
    "illustrative sample files."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Expected columns**")
st.sidebar.code(
    "rpm, torque_nm, voltage_v, current_a, temperature_c",
    language="text"
)

ALIASES = {
    "rpm": ["rpm", "speed", "speed_rpm", "motor_rpm"],
    "torque_nm": ["torque_nm", "torque", "torque (nm)", "torque_nm "],
    "voltage_v": ["voltage_v", "voltage", "voltage (v)", "volt"],
    "current_a": ["current_a", "current", "current (a)", "amps"],
    "temperature_c": [
        "temperature_c", "temperature", "temp", "temp_c", "temperature (c)"
    ],
    "real_power_w": [
        "real_power_w", "electrical_input_w", "input_power_w",
        "measured_real_power_w", "real_power", "input_power"
    ],
}


def normalize_column_name(name):
    """Normalize headers and aliases to a consistent snake_case form."""
    text = str(name).strip().lower()
    text = text.replace("°", "")
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def standardize(df):
    """Normalize input headers, validate required fields and report row cleaning."""
    original_rows = len(df)
    df = df.copy()

    normalized_columns = [normalize_column_name(column) for column in df.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        duplicates = sorted({
            column for column in normalized_columns
            if normalized_columns.count(column) > 1
        })
        raise ValueError(
            "Column names become duplicates after normalization: "
            + ", ".join(duplicates)
            + ". Please rename the duplicate columns in the uploaded file."
        )
    df.columns = normalized_columns

    # Normalize aliases using exactly the same rules as uploaded headers.
    alias_lookup = {}
    for target, choices in ALIASES.items():
        for alias in choices:
            alias_lookup[normalize_column_name(alias)] = target

    rename = {
        column: alias_lookup[column]
        for column in df.columns
        if column in alias_lookup
    }
    df = df.rename(columns=rename)

    required = ["rpm", "torque_nm", "voltage_v", "current_a"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns after header normalization: "
            + ", ".join(missing)
            + ". Check the uploaded file's column names."
        )

    if "temperature_c" not in df.columns:
        df["temperature_c"] = np.nan
    if "real_power_w" not in df.columns:
        df["real_power_w"] = np.nan

    numeric_columns = required + ["temperature_c", "real_power_w"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing_required_mask = df[required].isna().any(axis=1)
    nonfinite_required_mask = ~np.isfinite(df[required]).all(axis=1)
    invalid_required_mask = missing_required_mask | nonfinite_required_mask

    negative_mask = (
        (df["rpm"] < 0)
        | (df["voltage_v"] < 0)
        | (df["current_a"] < 0)
    ).fillna(False)

    valid_mask = (~invalid_required_mask) & (~negative_mask)
    cleaned = df.loc[valid_mask].copy()

    report = {
        "original": original_rows,
        "missing_or_non_numeric_required": int(missing_required_mask.sum()),
        "non_finite_required": int(
            (nonfinite_required_mask & ~missing_required_mask).sum()
        ),
        "negative_measurements": int(
            (negative_mask & ~invalid_required_mask).sum()
        ),
        "retained": len(cleaned),
        "removed": original_rows - len(cleaned),
    }

    return cleaned.reset_index(drop=True), report



def apply_power_calculation(df, power_mode, power_factor):
    """
    Calculate electrical input power and validate it before calculating
    efficiency. Invalid power rows are retained, clearly flagged, and receive
    NaN efficiency so they cannot silently distort summary statistics.
    """
    result = df.copy()

    if power_mode == "DC":
        result["electrical_input_w"] = (
            result["voltage_v"] * result["current_a"]
        )
    elif power_mode == "Single-phase AC":
        result["electrical_input_w"] = (
            result["voltage_v"] * result["current_a"] * power_factor
        )
    elif power_mode == "Three-phase AC (balanced, line-line voltage)":
        result["electrical_input_w"] = (
            math.sqrt(3) * result["voltage_v"]
            * result["current_a"] * power_factor
        )
    elif power_mode == "Measured real power column":
        result["electrical_input_w"] = result["real_power_w"]
    else:
        raise ValueError(f"Unsupported electrical power method: {power_mode}")

    result["mechanical_power_w"] = (
        result["torque_nm"] * 2 * math.pi * result["rpm"] / 60
    )

    electrical_power = pd.to_numeric(
        result["electrical_input_w"], errors="coerce"
    )
    finite_power = np.isfinite(electrical_power)
    positive_power = electrical_power > 0
    valid_power = finite_power & positive_power

    result["power_validation_flag"] = np.select(
        [
            ~finite_power,
            finite_power & ~positive_power
        ],
        [
            "Review: missing or non-finite input power",
            "Review: input power must be greater than zero"
        ],
        default="—"
    )

    result["efficiency_pct"] = np.nan
    result.loc[valid_power, "efficiency_pct"] = (
        100
        * result.loc[valid_power, "mechanical_power_w"]
        / electrical_power.loc[valid_power]
    )

    # Keep raw power readings for review; round only finite numeric values.
    calculated_columns = [
        "rpm", "torque_nm", "voltage_v", "current_a", "temperature_c",
        "real_power_w", "mechanical_power_w", "electrical_input_w",
        "efficiency_pct"
    ]
    for column in calculated_columns:
        if column in result.columns:
            result[column] = result[column].replace(
                [np.inf, -np.inf], np.nan
            ).round(3)

    # Sort by RPM so uploaded row order cannot create zig-zag line charts.
    return result.sort_values(
        "rpm", kind="stable"
    ).reset_index(drop=True)



def load_file(uploaded_file):
    """Read a CSV or Excel upload."""
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def metrics(df):
    """Calculate summary metrics for one configuration."""
    valid_efficiency = df["efficiency_pct"].dropna()

    return {
        "Rows": len(df),
        "Peak torque (Nm)": df["torque_nm"].max(),
        "Peak speed (RPM)": df["rpm"].max(),
        "Peak mechanical power (kW)": (
            df["mechanical_power_w"].max() / 1000
            if len(df) else np.nan
        ),
        "Mean efficiency (%)": (
            valid_efficiency.mean()
            if len(valid_efficiency) else np.nan
        ),
        "Peak temperature (°C)": df["temperature_c"].max(),
    }


def flag_outliers(df):
    """
    Apply a simple 1.5 x IQR screen independently to one dataset.
    A statistical flag is a review prompt, not proof of a motor fault.
    """
    outliers = pd.Series(False, index=df.index)
    columns = [
        "torque_nm", "current_a", "temperature_c", "efficiency_pct"
    ]

    for column in columns:
        values = df[column].dropna()
        if len(values) >= 4:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1

            if iqr > 0:
                column_flags = (
                    (df[column] < q1 - 1.5 * iqr)
                    | (df[column] > q3 + 1.5 * iqr)
                )
                outliers |= column_flags.fillna(False)

    return outliers.fillna(False)


def observations(df, flags):
    """Generate cautious, dataset-specific engineering observations."""
    notes = []

    if len(df) == 0:
        return ["No valid rows were found."]

    efficiency = df["efficiency_pct"].dropna()

    if len(efficiency):
        notes.append(
            f"Mean calculated efficiency is {efficiency.mean():.1f}%; "
            "verify torque, RPM, voltage and current units and measurement "
            "boundaries."
        )

        if (efficiency > 100).any():
            notes.append(
                "Some calculated efficiency values exceed 100%. This is "
                "physically inconsistent for a conventional motor boundary "
                "and should trigger a units, sensor or measurement-boundary check."
            )

    invalid_power_count = int(
        (df["power_validation_flag"] != "—").sum()
    )
    if invalid_power_count:
        notes.append(
            f"{invalid_power_count} row(s) have missing, non-finite or "
            "non-positive electrical input power. Efficiency was not "
            "calculated for those rows; inspect the power readings and "
            "selected electrical measurement method."
        )

    if df["temperature_c"].notna().any():
        notes.append(
            f"Highest recorded temperature is "
            f"{df['temperature_c'].max():.1f} °C. Compare it with the "
            "component-specific limit supplied by the engineering team."
        )

    notes.append(
        f"{int(flags.sum())} row(s) were flagged by a simple IQR statistical "
        "outlier screen. A flag is not proof of a fault."
    )
    physical_count = int((df["efficiency_pct"] > 100).sum())
    if physical_count:
        notes.append(
            f"{physical_count} row(s) have calculated efficiency above 100% "
            "and are separately flagged for physical-plausibility review."
        )

    return notes


def make_pdf(title, df, summary, notes):
    """Build a PDF with combined configuration metrics, observations and data."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Normal"]
        ),
        Spacer(1, 12),
        Paragraph("Performance summary", styles["Heading2"])
    ]

    summary_data = [["Metric", "Value"]]
    for key, value in summary.items():
        if pd.isna(value):
            formatted_value = "—"
        elif isinstance(value, (int, float, np.number)):
            formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)

        summary_data.append([str(key), formatted_value])

    summary_table = Table(summary_data, colWidths=[280, 180], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163a52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5df")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f4f7")]),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.extend([summary_table, Spacer(1, 14)])

    story.append(
        Paragraph("Automated observations", styles["Heading2"])
    )
    for note in notes:
        if note in ("CONFIGURATION A", "CONFIGURATION B"):
            story.append(Paragraph(note, styles["Heading3"]))
        else:
            story.append(Paragraph("• " + note, styles["BodyText"]))
            story.append(Spacer(1, 5))

    story.extend([
        Spacer(1, 12),
        Paragraph("Data preview", styles["Heading2"])
    ])

    preview_columns = [
        "rpm", "torque_nm", "voltage_v", "current_a",
        "mechanical_power_w", "efficiency_pct"
    ]
    if "configuration" in df.columns:
        preview_columns.insert(0, "configuration")
    if "power_validation_flag" in df.columns:
        preview_columns.append("power_validation_flag")

    preview = df[preview_columns].head(24)
    table_data = [list(preview.columns)]
    table_data.extend(preview.fillna("").astype(str).values.tolist())

    preview_table = Table(table_data, repeatRows=1)
    preview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163a52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(preview_table)

    story.extend([
        Spacer(1, 12),
        Paragraph(
            "Important: This report is an analytical aid, not engineering "
            "certification. Validate measurements, units, test conditions and "
            "conclusions with qualified engineers.",
            styles["Italic"]
        )
    ])

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------
# 1. Upload test data
# ---------------------------------------------------------------------
st.subheader("1. Upload test data")

column_a, column_b = st.columns(2)

with column_a:
    file_a = st.file_uploader(
        "Motor / configuration A (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="a"
    )

with column_b:
    file_b = st.file_uploader(
        "Motor / configuration B (optional)",
        type=["csv", "xlsx", "xls"],
        key="b"
    )

# Select one data source at a time so uploaded and synthetic files cannot mix.
default_source = "Uploaded files" if file_a or file_b else "Illustrative sample data"
data_source = st.radio(
    "Data source",
    ["Uploaded files", "Illustrative sample data"],
    index=0 if default_source == "Uploaded files" else 1,
    horizontal=True
)

sample_a = pd.DataFrame({
    "rpm": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
    "torque_nm": [42, 41, 40, 38, 36, 34, 31, 28, 25, 21, 17],
    "voltage_v": [48] * 11,
    "current_a": [18, 20, 22, 24, 26, 28, 29, 30, 31, 30, 28],
    "temperature_c": [31, 33, 35, 38, 41, 44, 47, 50, 54, 58, 61]
})

sample_b = pd.DataFrame({
    "rpm": [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
    "torque_nm": [40, 39, 38, 37, 35, 33, 30, 27, 24, 20, 16],
    "voltage_v": [48] * 11,
    "current_a": [17, 19, 21, 23, 25, 27, 28, 29, 30, 29, 27],
    "temperature_c": [30, 32, 34, 36, 39, 42, 45, 48, 51, 55, 58]
})


if data_source == "Illustrative sample data":
    if file_a or file_b:
        st.warning(
            "Illustrative sample mode is selected; uploaded files are ignored "
            "for this run to prevent mixing synthetic and real data."
        )
    raw_a, raw_b = sample_a, sample_b
else:
    if file_a is None:
        st.info("Upload Configuration A to begin. Configuration B is optional.")
        st.stop()
    raw_a = load_file(file_a)
    raw_b = load_file(file_b) if file_b is not None else None

st.subheader("Electrical input power method")
power_mode = st.selectbox(
    "Choose the measurement / motor test setup",
    [
        "DC",
        "Single-phase AC",
        "Three-phase AC (balanced, line-line voltage)",
        "Measured real power column"
    ],
    help=(
        "For balanced three-phase AC, voltage_v must be line-to-line voltage "
        "and current_a must be line current."
    )
)
power_factor = 1.0
if power_mode in ("Single-phase AC", "Three-phase AC (balanced, line-line voltage)"):
    power_factor = st.number_input(
        "Power factor (PF)", min_value=0.0, max_value=1.0,
        value=0.90, step=0.01,
        help="Enter the measured power factor from the test instrument."
    )

try:
    a_raw, cleaning_a = standardize(raw_a)
    b_raw, cleaning_b = (
        standardize(raw_b) if raw_b is not None else (None, None)
    )
    a = apply_power_calculation(a_raw, power_mode, power_factor)
    b = (
        apply_power_calculation(b_raw, power_mode, power_factor)
        if b_raw is not None else None
    )
except Exception as error:
    st.error(f"Could not process the file: {error}")
    st.stop()

if a.empty:
    st.error("Configuration A contains no valid data rows after cleaning.")
    st.stop()

if b is not None and b.empty:
    st.warning(
        "Configuration B contains no valid data rows after cleaning. "
        "Only Configuration A will be analyzed."
    )
    b = None
    cleaning_b = None

st.subheader("Data-cleaning summary")
cleaning_rows = [{
    "Configuration": "A",
    "Original rows": cleaning_a["original"],
    "Rows retained": cleaning_a["retained"],
    "Rows removed": cleaning_a["removed"],
    "Missing / non-numeric required fields": cleaning_a["missing_required"],
    "Negative RPM / voltage / current": cleaning_a["negative_measurements"],
}]
if cleaning_b is not None:
    cleaning_rows.append({
        "Configuration": "B",
        "Original rows": cleaning_b["original"],
        "Rows retained": cleaning_b["retained"],
        "Rows removed": cleaning_b["removed"],
        "Missing / non-numeric required fields": cleaning_b["missing_or_non_numeric_required"],
        "Non-finite required values": cleaning_b["non_finite_required"],
        "Negative RPM / voltage / current": cleaning_b["negative_measurements"],
    })
st.dataframe(pd.DataFrame(cleaning_rows), use_container_width=True, hide_index=True)
st.caption(
    "Rows are removed if required fields are missing/non-numeric or if RPM, "
    "voltage or current is negative. Missing temperature is allowed."
)

st.success(
    f"Configuration A: {len(a)} valid rows"
    + (f" · Configuration B: {len(b)} valid rows" if b is not None else "")
)
st.caption(
    f"Electrical input method: {power_mode}. Calculations assume torque in Nm, "
    "speed in RPM, voltage in V and current in A. For three-phase AC, voltage "
    "must be line-to-line and current must be line current. Efficiency is "
    "mechanical output power ÷ electrical input power × 100."
)

# ---------------------------------------------------------------------
# 2. Performance overview
# ---------------------------------------------------------------------
st.subheader("2. Performance overview")

metrics_a = metrics(a)
metrics_b = metrics(b) if b is not None else None

metric_columns = st.columns(5)
overview_keys = [
    "Peak torque (Nm)",
    "Peak mechanical power (kW)",
    "Mean efficiency (%)",
    "Peak temperature (°C)",
    "Rows"
]

for column, key in zip(metric_columns, overview_keys):
    value = metrics_a[key]
    column.metric(
        key,
        "—" if pd.isna(value) else (
            f"{value:.1f}"
            if isinstance(value, (int, float, np.number))
            else str(value)
        )
    )

left, right = st.columns(2)

with left:
    figure = px.line(
        a,
        x="rpm",
        y="torque_nm",
        markers=True,
        title="Torque vs RPM · Configuration A"
    )
    figure.update_layout(
        xaxis_title="Speed (RPM)",
        yaxis_title="Torque (Nm)"
    )
    st.plotly_chart(figure, use_container_width=True)

with right:
    figure = px.line(
        a,
        x="rpm",
        y="mechanical_power_w",
        markers=True,
        title="Mechanical power vs RPM · Configuration A"
    )
    figure.update_layout(
        xaxis_title="Speed (RPM)",
        yaxis_title="Mechanical power (W)"
    )
    st.plotly_chart(figure, use_container_width=True)

left, right = st.columns(2)

with left:
    figure = px.line(
        a,
        x="rpm",
        y="efficiency_pct",
        markers=True,
        title="Calculated efficiency vs RPM · Configuration A"
    )
    figure.update_layout(
        xaxis_title="Speed (RPM)",
        yaxis_title="Efficiency (%)"
    )
    st.plotly_chart(figure, use_container_width=True)

with right:
    if a["temperature_c"].notna().any():
        figure = px.line(
            a,
            x="rpm",
            y="temperature_c",
            markers=True,
            title="Temperature vs RPM · Configuration A"
        )
        figure.update_layout(
            xaxis_title="Speed (RPM)",
            yaxis_title="Temperature (°C)"
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("No temperature column was supplied.")

# ---------------------------------------------------------------------
# 3. Data quality and anomaly screening
# ---------------------------------------------------------------------
st.subheader("3. Data quality and anomaly screening")

flags_a = flag_outliers(a)
anomaly_a = a.copy()
anomaly_a["configuration"] = "A"
anomaly_a["statistical_flag"] = flags_a.map({
    True: "Review",
    False: "—"
})
anomaly_a["physical_plausibility_flag"] = np.where(
    anomaly_a["efficiency_pct"] > 100, "Review: efficiency >100%", "—"
)
anomaly_a["engineering_review_flag"] = np.where(
    flags_a | (anomaly_a["efficiency_pct"] > 100), "Review", "—"
)

anomaly_frames = [anomaly_a]
total_flags = int(flags_a.sum())
flags_b = None

if b is not None:
    flags_b = flag_outliers(b)
    anomaly_b = b.copy()
    anomaly_b["configuration"] = "B"
    anomaly_b["statistical_flag"] = flags_b.map({
        True: "Review",
        False: "—"
    })
    anomaly_b["physical_plausibility_flag"] = np.where(
        anomaly_b["efficiency_pct"] > 100, "Review: efficiency >100%", "—"
    )
    anomaly_b["engineering_review_flag"] = np.where(
        flags_b | (anomaly_b["efficiency_pct"] > 100), "Review", "—"
    )
    anomaly_frames.append(anomaly_b)
    total_flags += int(flags_b.sum())

anomaly_df = pd.concat(anomaly_frames, ignore_index=True)

st.write(
    f"**{total_flags}** total row(s) flagged across all available "
    "configurations using an independent IQR screen for each dataset."
)

if b is not None:
    flag_col_a, flag_col_b = st.columns(2)
    flag_col_a.metric("Configuration A flags", int(flags_a.sum()))
    flag_col_b.metric("Configuration B flags", int(flags_b.sum()))

st.dataframe(
    anomaly_df,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------------------
# 4. Configuration comparison
# ---------------------------------------------------------------------
st.subheader("4. Configuration comparison")

if b is not None:
    comparison_keys = [
        "Peak torque (Nm)",
        "Peak speed (RPM)",
        "Peak mechanical power (kW)",
        "Mean efficiency (%)",
        "Peak temperature (°C)"
    ]

    comparison = pd.DataFrame({
        "Metric": comparison_keys,
        "Configuration A": [metrics_a[key] for key in comparison_keys],
        "Configuration B": [metrics_b[key] for key in comparison_keys]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    comparison_plot = pd.concat([
        a.assign(configuration="A"),
        b.assign(configuration="B")
    ], ignore_index=True).sort_values(
        ["rpm", "configuration"], kind="stable"
    )

    metric_choice = st.selectbox(
        "Compare performance metric",
        [
            "torque_nm",
            "mechanical_power_w",
            "efficiency_pct",
            "temperature_c"
        ]
    )

    if comparison_plot[metric_choice].notna().any():
        figure = px.line(
            comparison_plot,
            x="rpm",
            y=metric_choice,
            color="configuration",
            markers=True,
            title=f"Configuration A vs B · {metric_choice}"
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info(f"No valid values are available for {metric_choice}.")

    st.caption(
        "Comparison is meaningful only when both files use comparable test "
        "conditions, sensors, units and measurement procedures."
    )
else:
    st.info(
        "Upload a second file to compare two motor configurations."
    )

# ---------------------------------------------------------------------
# 5. Automated engineering observations
# ---------------------------------------------------------------------
st.subheader("5. Automated engineering observations")

notes = ["CONFIGURATION A"]
notes.extend(observations(a, flags_a))

if b is not None:
    notes.append("CONFIGURATION B")
    notes.extend(observations(b, flags_b))

for note in notes:
    if note in ("CONFIGURATION A", "CONFIGURATION B"):
        st.markdown(f"### {note}")
    else:
        st.write("• " + note)

# ---------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------
st.subheader("6. Export")

# Combined CSV: includes configuration labels and statistical flags.
csv_bytes = anomaly_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download processed data (CSV)",
    csv_bytes,
    file_name="motor_test_analysis.csv",
    mime="text/csv"
)

# Combined PDF summary: includes metrics and preview rows for A and B.
report_summary = {}

for key, value in metrics_a.items():
    report_summary[f"Configuration A - {key}"] = value

if b is not None:
    for key, value in metrics_b.items():
        report_summary[f"Configuration B - {key}"] = value

pdf_bytes = make_pdf(
    "Motor Test Intelligence Report",
    anomaly_df,
    report_summary,
    notes
)

st.download_button(
    "Download engineering summary (PDF)",
    pdf_bytes,
    file_name="motor_test_report.pdf",
    mime="application/pdf"
)

# ---------------------------------------------------------------------
# Calculation notes and limitations
# ---------------------------------------------------------------------
with st.expander("How are the calculations made?"):
    st.latex(
        r"P_{mechanical}(W)=T(Nm)\times\frac{2\pi\times RPM}{60}"
    )
    if power_mode == "DC":
        st.latex(r"P_{electrical}(W)=V\times I")
    elif power_mode == "Single-phase AC":
        st.latex(r"P_{electrical}(W)=V\times I\times PF")
    elif power_mode == "Three-phase AC (balanced, line-line voltage)":
        st.latex(r"P_{electrical}(W)=\sqrt{3}\times V_L\times I_L\times PF")
    else:
        st.write("Electrical input power is taken from the uploaded measured real-power column.")
    st.latex(
        r"Efficiency(\%)=\frac{P_{mechanical}}{P_{electrical}}\times100"
    )
    st.write(
        "The anomaly screen uses a simple 1.5×IQR rule independently for "
        "each configuration. It is an exploratory data-quality aid, not a "
        "trained fault-detection model."
    )

with st.expander("Prototype limitations and safe use"):
    st.markdown("""
    - Sample values are synthetic and are included only to demonstrate the workflow.
    - No Ferrstron confidential data, motor design files or internal specifications are included.
    - No claim is made that this software is approved for design release, safety certification or production decisions.
    - Confirm units, sensor calibration, test boundary and operating conditions before interpreting results.
    - Select the correct DC / single-phase AC / three-phase AC / measured-real-power method for the actual test setup.
    - For three-phase AC, use line-to-line voltage, line current and measured power factor for the balanced-load formula.
    - Efficiency above 100% is retained and separately flagged for physical-plausibility review rather than silently corrected.
    - Missing, non-finite or non-positive electrical input power is flagged; efficiency is not calculated for those rows.
    - Statistical outlier flags are prompts for investigation, not diagnoses of motor faults.
    """)
