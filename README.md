<div align="center">

<img src="docs/public/Glance.png" alt="Glance Logo" width="400"/>

<h3>Integrated 3D Rocket Design, Live Telemetry & Real-Time ML Predictive Analysis Platform</h3>

<p>
  <img src="https://img.shields.io/badge/License-GPLv3%20with%20restrictions-blue.svg" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version"/>
  <img src="https://img.shields.io/badge/GUI-PySide6-green" alt="PySide6"/>
  <img src="https://img.shields.io/badge/ML-PyTorch%20%7C%20CatBoost%20%7C%20Scikit--Learn-orange" alt="ML Stack"/>
  <img src="https://img.shields.io/badge/3D%20CAD-Parametric%20Simulation-purple" alt="3D CAD"/>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform"/>
</p>

<p><em>An advanced fork of <strong>Glance</strong>: Evolving from real-time telemetry visualization into a unified aerospace engineering environment featuring parametric 3D rocket design, 3-DOF flight simulation, and live multi-model machine learning analytics.</em></p>

<p>
  <a href="#-about-this-fork">About This Fork</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="https://glance.teamignition.space/">Upstream Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

<hr/>

<img src="docs/public/ign_logo_wht.png" alt="Team Ignition Logo" width="150"/>

<p><strong>Originally Developed by Team Ignition Software Department</strong><br/>
<em>Official Model Rocketry Team of Vellore Institute of Technology, Chennai</em><br/>
<strong>Forked &amp; Extended with Parametric 3D Rocket Design, Flight Dynamics &amp; Live ML Inference</strong></p>

</div>

---

## 🚀 About This Fork

> [!IMPORTANT]
> ### Evolution: From Telemetry Dashboard to Integrated Rocket Design & Live ML Analytics
> **This repository is a major fork of [Glance](https://github.com/teamignitionvitc/Glance) (originally created by Team Ignition).**
>
> * **Before (Upstream Glance):** The original Glance was created strictly as a **real-time telemetry visualization platform** — providing real-time data streaming (Serial, TCP, UDP), digital signal filtering (Kalman, Low-Pass, Moving Average, Median), modular GUI widgets (charts, gauges, status LEDs, maps), and session logging.
> * **Now (What We Have Added):** We transformed Glance into an end-to-end rocket design, simulation, and predictive intelligence suite:
>   1. **Interactive 3D Rocket Designer:** A parametric CAD-style airframe builder with real-time 3D mesh rendering (zero GPU/OpenGL driver dependencies via CPU rasterization), customizable nose cones (ogive, conical, parabolic, elliptical), body dimensions, fin geometry, and motor mounts.
>   2. **Integrated 3-DOF Flight Physics Simulation:** Real-time atmospheric flight dynamics (ISA model) calculating dynamic aerodynamic drag, thrust curves, propellant mass depletion, and live static stability margins (CP vs CG).
>   3. **Coupled Telemetry Stream:** The flight simulation emits a live 13-channel telemetry stream directly from your custom-designed rocket, bridging physical vehicle geometry with live sensor data.
>   4. **Live Real-Time Machine Learning Suite:** Streaming multi-model inference running 4 distinct algorithms sample-by-sample:
>      * **Deep Autoencoder (PyTorch):** Continuous nominal-baseline anomaly detection with channel-level residual error attribution.
>      * **Gaussian Process (GP):** Horizon forecasting with dynamic 95% confidence intervals and live self-evaluating coverage metrics.
>      * **LSTM Recurrent Network (PyTorch):** Time-series sequence classifier diagnosing health degradation stages (*Healthy*, *Degrading*, *Critical*).
>      * **CatBoost Regressor:** 0–100 system health index regression with explainable feature-importance attribution.
>   5. **In-App Model Training & Custom Datasets:** Train all 4 ML models on any custom CSV dataset right inside the GUI, with pre-trained models bundled for rocket flight profiles and the NASA C-MAPSS FD001 turbofan degradation benchmark.

---

## Overview

<table>
<tr>
<td width="60%">

### The Complete Aerospace Telemetry & Intelligence Pipeline

By uniting **parametric 3D rocket design**, **physics-based flight dynamics**, and **live ML predictive maintenance** with Glance's robust **real-time telemetry and signal-filtering core**, this platform enables a complete closed feedback loop:

* **Design**: Interactively adjust rocket dimensions, fin geometries, and motor configurations in a live 3D viewport.
* **Simulate & Stream**: Run high-fidelity 3-DOF flights or connect to live hardware feeds (Serial / TCP / UDP).
* **Predict & Diagnose**: Feed live sensor telemetry directly through 4 parallel machine learning algorithms to detect anomalies, forecast trends, classify degradation stages, and monitor overall health in real time.
* **Visualize & Log**: View live telemetry across customizable dashboards with hardware-accelerated graphing, Kalman filtering, and CSV/JSON recording.

</td>
<td width="40%">

### Core Capabilities

<ul>
<li><strong>Interactive 3D Rocket Designer</strong> (ogive/conical/parabolic/elliptical, fins, motor)</li>
<li><strong>3-DOF Flight Simulation</strong> with live static stability margin (CP vs CG)</li>
<li><strong>Live ML Predictive Maintenance</strong> (4 algorithms, arbitrary channel count)</li>
<li><strong>In-App Training</strong> on custom CSV/TXT datasets with auto-column discovery</li>
<li><strong>Pre-trained Bundles</strong> (Rocket flight &amp; NASA C-MAPSS turbofan)</li>
<li><strong>Multi-Channel Real-Time Plotting</strong> (PySide6 + PyQtGraph)</li>
<li><strong>Serial, TCP, UDP &amp; Dummy Streaming</strong></li>
<li><strong>Digital Signal Filtering</strong> (Kalman, Low-Pass, Moving Avg, Median)</li>
<li><strong>7+ Modular Widget Types</strong> (Graphs, Gauges, LEDs, GPS Maps, Tables)</li>
<li><strong>Automated Data Logging &amp; PDF Reports</strong></li>
<li><strong>Raw Telemetry Packet Inspector</strong></li>
</ul>

</td>
</tr>
</table>

---

## Features

<details open>
<summary><h3>🚀 3D Rocket Designer & Flight Physics Simulation</h3></summary>

Design parametric rocket airframes in an interactive 3D workspace and immediately simulate their aerodynamic flight trajectory. The simulation generates a rich 13-channel telemetry stream that flows directly into the dashboard and live ML models.

<table>
<tr>
<td width="50%">

**Parametric 3D CAD Airframe Designer**

- **Interactive 3D Viewport**: Orbit (left-click drag) and zoom (mouse wheel). Custom CPU software rasterization using `QPainter` — fast, responsive, and works everywhere without OpenGL or GPU driver dependencies.
- **Nose Cone Profiles**: Choose between Ogive, Conical, Parabolic, and Elliptical aerodynamic profiles with customizable length and diameter.
- **Body & Fins**: Configure body length, tube diameter, fin count (3, 4, etc.), root chord, tip chord, height/span, and sweep angle. The 3D surface mesh re-tessellates dynamically as you adjust dimensions.
- **Propulsion & Mass Configuration**: Set dry mass, propellant mass, motor burn duration, average thrust, and motor mount geometry.

</td>
<td width="50%">

**Aerodynamics & 3-DOF Flight Physics**

- **Live Stability Analysis**: Barrowman aerodynamic relations calculate the Center of Pressure (CP), while dynamic mass distribution tracks the Center of Gravity (CG). Real-time static stability margin (in calibers) with color-coded warnings if a design is unstable.
- **Atmospheric Model**: International Standard Atmosphere (ISA) exponential air density model and altitude-dependent speed of sound calculation.
- **Full Trajectory Simulation**: Solves 3-DOF equations of motion taking into account aerodynamic drag, motor thrust curve, and mass burn-off.
- **Physical Degradation Modeling**: Simulates nozzle erosion over motor burn time, causing chamber pressure bleed, nozzle temperature rise, and vibration — providing realistic failure modes for ML validation.
- **Closed-Loop Feedback**: Modify any airframe parameter and see the resulting changes in flight dynamics and ML verdicts instantly.

</td>
</tr>
</table>

</details>

<details open>
<summary><h3>🤖 Real-Time Machine Learning & Predictive Maintenance</h3></summary>

Four complementary machine learning algorithms run as **live streaming inference** on incoming telemetry — one sample in, four immediate verdicts out. Works with **arbitrary channel counts**, auto-detecting sensor fields from the data.

<table>
<tr>
<td width="50%">

**Four Live Inference Algorithms**

- **Deep Autoencoder (PyTorch)** — *What looks abnormal right now?*
  Trained on nominal flight data. Computes reconstruction loss on each sample; elevated errors trigger anomaly alerts and report per-channel residuals to isolate the root cause.
- **Gaussian Process (Scikit-Learn / SciPy)** — *Where is it heading, and can we trust the prediction?*
  Forecasts chosen trajectory channels ahead in time with a continuous 95% confidence interval band. Continuously self-evaluates against ground-truth data with live RMSE and interval coverage scores.
- **LSTM Recurrent Network (PyTorch)** — *Which degradation stage does the temporal sequence imply?*
  Classifies sequence trajectories over rolling temporal windows into discrete stages: **Healthy**, **Degrading**, or **Critical**.
- **CatBoost Regressor** — *What is the overall condition score, and what drives it?*
  Predicts an explainable 0–100 health index score and computes instant feature importance rankings.

</td>
<td width="50%">

**Flexible Data & Training Sources**

- **Three Telemetry Sources**:
  - *Live Telemetry*: Serial, TCP, or UDP hardware streams.
  - *Rocket Flight Simulation*: Telemetry generated live from your custom 3D rocket design.
  - *Dataset Replay*: Step-by-step playback of historical flight or test-stand CSV files.
- **In-App Training on Custom Datasets**:
  - Point to any CSV or TXT dataset; columns, sensors, units, cycles, and RUL metadata are automatically discovered.
  - Supports supervised run-to-failure training or unsupervised nominal baseline calibration.
  - Background worker thread (`TrainingWorker`) fits models without freezing the UI.
- **Pretrained Benchmark Bundles**:
  - *Rocket Flight Bundle*: Pretrained on simulated multi-regime rocket trajectories.
  - *NASA C-MAPSS FD001*: Turbofan degradation benchmark achieving AUROC 0.87, 94.8% forecast coverage, 87% stage accuracy, and R² 0.76.
- **Live Visual Analytics**:
  - KPI summary cards per algorithm, measured vs. predicted overlay plots with forecast bands, health trend history, circular dial gauge, and stage LED indicator.

</td>
</tr>
</table>

Build the bundled models with `python ml/build_pretrained.py`. The offline benchmark pipeline in `ml/` (`train_models.py`, `evaluate.py`) is also included.

</details>

<details>
<summary><h3>📊 Visualization & Display</h3></summary>

<table>
<tr>
<td width="50%">

**Interactive Graphs**

- Multi-parameter time-series plotting
- Zoom, pan, and crosshair inspection
- Auto-scaling with manual override
- Color-coded parameter traces
- Real-time legend updates

**Status Displays**

- Large-format instant value displays with alarm states
- Circular gauges with threshold zones
- LED indicators with color-coded status
- Statistical histograms for distribution analysis

</td>
<td width="50%">

**Data Tables & GPS**

- Searchable log tables with highlighting
- Timestamped data entries
- Multi-parameter comparison
- GPS mapping with satellite imagery
- Real-time location tracking
- Interactive map controls

**Advanced Features**

- Crosshair cursor with value readout
- Mouse-click data point selection
- Widget floating and docking
- Multi-tab dashboard support
- Undo/Redo functionality for widget operations
- Standalone raw telemetry viewer

</td>
</tr>
</table>

</details>

<details>
<summary><h3>🎛️ Signal Processing</h3></summary>

<table>
<tr>
<td width="50%">

**Available Filters**

<dl>
<dt><strong>Kalman Filter</strong></dt>
<dd>Optimal estimation for noisy signals with configurable process and measurement variance. Ideal for tracking applications and sensor fusion.</dd>

<dt><strong>Moving Average Filter</strong></dt>
<dd>Simple smoothing over configurable window size (2-100 samples). Effective for reducing random noise.</dd>

<dt><strong>Low-Pass Filter</strong></dt>
<dd>Exponential smoothing with adjustable alpha (0.01-1.0). Fast response with controllable lag.</dd>

<dt><strong>Median Filter</strong></dt>
<dd>Outlier rejection using median of N samples. Excellent for spike removal while preserving edges.</dd>
</dl>

</td>
<td width="50%">

**Filter Management**

- Per-parameter filter chains
- Enable/disable filters without restart
- Real-time filter parameter adjustment
- Filter state reset capability
- Visual filter status indicators
- Import/export filter configurations

**Processing Features**

- Apply filters to any parameter
- Chain multiple filters per parameter
- Raw and filtered data logging
- Zero-lag display options
- Configurable buffer management

</td>
</tr>
</table>

</details>

<details>
<summary><h3>🔌 Connectivity & Data Sources</h3></summary>

<table>
<tr>
<td width="50%">

**Serial Communication**

- RS232, RS485, USB-Serial support
- Auto-detection of available ports
- Configurable baudrate (300-10M)
- Manual refresh of port list
- Timeout configuration

**Network Protocols**

- TCP client connections
- UDP listener mode
- Configurable host and port
- Connection status monitoring
- Automatic reconnection

</td>
<td width="50%">

**Data Format Support**

- JSON array parsing
- CSV delimited data
- Raw binary bytes
- Binary struct (with type definitions)
- Bit-level data extraction
- Custom format definitions
- Little/big endian support
- Configurable channel count (1-1024)

**Simulation Mode**

- Built-in dummy data generator
- No hardware required for testing
- Configurable sine wave patterns
- Random noise injection

</td>
</tr>
</table>

</details>

<details>
<summary><h3>💾 Data Logging & Export</h3></summary>

<table>
<tr>
<td width="50%">

**Logging Features**

- CSV and JSON format support
- Parameter-selective logging
- Configurable buffer sizes (10-1000)
- Auto-generated timestamped filenames
- Manual buffer flush controls
- Organized logs directory structure

**Data Management**

- Raw and filtered value storage
- Millisecond-precision timestamps
- Elapsed time tracking
- Incremental file writing
- Low memory footprint

</td>
<td width="50%">

**Export Options**

- Standard CSV for Excel/MATLAB
- JSON Lines format for programming
- Human-readable timestamps
- Parameter metadata inclusion
- Configurable write frequency

**PDF Report Generation**

- Automated summary reports from logged data
- Statistical analysis (min, max, average)
- Session metadata and duration
- Professional formatting with charts
- Generate via **Ctrl+Shift+R** or **Data Logging → Generate Summary Report**

**Performance**

- Buffered I/O for efficiency
- Configurable flush intervals
- Background writing
- No dropped packets during logging
- Minimal CPU overhead

</td>
</tr>
</table>

</details>

<details>
<summary><h3>🔍 Monitoring & Debugging</h3></summary>

<table>
<tr>
<td width="50%">

**Raw Telemetry Monitor**

- Decimal, hexadecimal, ASCII, binary, and mixed display modes
- Real-time packet inspection
- Packet statistics (count, rate, bytes)
- Pause and resume capability
- Search and highlight functionality
- Screenshot capture
- Save to file

**VS Code-like Status Bar**

- Segmented layout for clear information hierarchy
- Live clock display
- Connection status with interactive button
- Packet count and data received metrics
- Active parameters and widget count
- UI update rate (FPS)
- Data logging indicator
- Quick access to Raw Telemetry viewer

</td>
<td width="50%">

**Debug Tools**

- Connection diagnostics
- Data format validation
- Parameter mapping verification
- Real-time packet rate monitoring
- Error message display
- Connection retry logic

**Advanced Telemetry**

- Multiple display format switching
- Packet numbering
- Timestamp display options
- Byte rate calculation
- Error counting
- Custom search filters

</td>
</tr>
</table>

</details>

<details>
<summary><h3>🎨 User Interface</h3></summary>

<table>
<tr>
<td width="50%">

**Modern Design**

- Professional "Apple-like" dark theme
- High-contrast elements with SF Pro typography
- Custom styled widgets with glassmorphism effects
- Smooth animations and transitions
- Gradient accents and refined color palettes
- Responsive layout with segmented controls

**Dashboard Management**

- 4-phase creation wizard
- Professional Welcome Screen with system icons
- Setup wizard with validation
- Widget pre-configuration
- Live dashboard phase

</td>
<td width="50%">

**Layout Control**

- Drag-and-drop widget positioning
- Floating and docked modes
- Multi-tab support with rename
- Tile evenly function
- Right-click context menus
- Widget resize and minimize
- Customizable dashboard titles

**Workflow**

- Project save/load system
- Unsaved changes tracking
- Configuration validation
- Error prevention dialogs
- Comprehensive keyboard shortcut support
- Command pattern for undo/redo operations

</td>
</tr>
</table>

</details>

<details>
<summary><h3>⚙️ Technical Architecture</h3></summary>

<table>
<tr>
<td width="50%">

**Backend Core**

- **Threaded Acquisition**: `DataSimulator` runs in a dedicated `QThread`, ensuring the UI remains responsive even at high data rates (100Hz+).
- **Abstraction Layer**: `DataReader` provides a unified interface for Serial, TCP, UDP, and Dummy sources, handling low-level socket/port management and error recovery with automatic reconnection.
- **Binary Parsing**: Uses Python's `struct` module for high-performance parsing of binary packets. Supports mixed data types (int8/16/32, uint8/16/32, float32/64) and binary struct definitions via parameter configuration.

**Data Flow**

1. **Source**: Hardware/Network sends data packet.
2. **Acquisition**: `DataReader` reads bytes/string.
3. **Parsing**: Data is converted to a normalized `list[float]`.
4. **Distribution**: `DataSimulator` emits `newData` Qt Signal.
5. **Processing**: Main thread applies active filters (Kalman/MA).
6. **Visualization**: Widgets update via optimized paint events.

</td>
<td width="50%">

**Data Logging Internals**

- **Buffered I/O**: `DataLogger` accumulates data in memory (default 100 samples) before performing a bulk write to disk. This minimizes filesystem overhead and prevents write-latency from affecting the acquisition loop.
- **Formats**:
  - **CSV**: Optimized for import into Excel/MATLAB with timestamp and elapsed time columns.
  - **JSON Lines**: Stream-friendly format for programmatic analysis with ISO timestamp formatting.
- **Report Generation**: PDF reports generated using `reportlab` library, analyzing logged CSV/JSON data to produce statistical summaries and visualizations.

**Signal Processing**

- **Filter Chain**: Filters are implemented as independent objects inheriting from `SignalFilter` base class.
- **State Management**: Filters maintain their own internal state (buffers, covariance matrices) which persists across updates but can be reset dynamically.
- **Real-time**: All filtering occurs in the main event loop, designed for low-latency (<1ms) processing per packet.
- **Filter Management**: `FilterManager` handles per-parameter filter chains, enabling/disabling filters without restart, and serialization for project save/load.

</td>
</tr>
</table>

</details>

---

## Screenshots

<div align="center">

<table>
<tr>
<td width="50%">
<img src="docs/public/Doc%20Images/welcome_screen_new.png" alt="Welcome Screen"/>
<p><em>Modern welcome screen with quick-start options</em></p>
</td>
<td width="50%">
<img src="docs/public/Doc%20Images/dashboard_config.png" alt="Connection Settings"/>
<p><em>Flexible data source configuration</em></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="docs/public/Doc%20Images/parameters_new.png" alt="Parameter Management"/>
<p><em>Parameter configuration with array index mapping</em></p>
</td>
<td width="50%">
<img src="docs/public/Doc%20Images/logging_config.png" alt="Data Logging"/>
<p><em>Comprehensive data logging with format selection</em></p>
</td>
</tr>
<tr>
<td colspan="2">
<img src="docs/public/Doc%20Images/dashboard_full_new.png" alt="Dashboard View"/>
<p><em>Real-time multi-widget dashboard layout</em></p>
</td>
</tr>
</table>

</div>

---

## System Requirements

<table>
<tr>
<td width="50%">

### Minimum

<table>
<tr><th>Component</th><th>Requirement</th></tr>
<tr><td>Operating System</td><td>Windows 10/11<br/>Linux (Ubuntu 20.04+)<br/>macOS 10.15+</td></tr>
<tr><td>Python</td><td>3.8 or higher</td></tr>
<tr><td>RAM</td><td>4 GB</td></tr>
<tr><td>Display</td><td>1280x720</td></tr>
<tr><td>Storage</td><td>500 MB free</td></tr>
</table>

</td>
<td width="50%">

### Recommended

<table>
<tr><th>Component</th><th>Requirement</th></tr>
<tr><td>Operating System</td><td>Windows 11<br/>Linux (Ubuntu 22.04+)<br/>macOS 12+</td></tr>
<tr><td>Python</td><td>3.10 or higher</td></tr>
<tr><td>RAM</td><td>8 GB or more</td></tr>
<tr><td>Display</td><td>1920x1080 or higher</td></tr>
<tr><td>Storage</td><td>1 GB+ for logging</td></tr>
</table>

</td>
</tr>
</table>

---

## Installation

### Method 1: Download & Run (Recommended)

The easiest way to use Glance is to download the standalone executable for your system. No Python installation required.

| Platform    | Download                                                                                                                                                                                                                          | Release Notes                                                                            |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Windows** | [![Windows](https://img.shields.io/badge/Windows-Download_.exe-0078D4?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/teamignitionvitc/Glance/releases/download/v1.0.0-alpha.1/Glance_Setup_Windows_v1.exe) | [v1.0.0-alpha.1](https://github.com/teamignitionvitc/Glance/releases/tag/v1.0.0-alpha.1) |
| **Linux**   | [![Linux](https://img.shields.io/badge/Linux-Download_.deb-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/teamignitionvitc/Glance/releases/download/v1.0.0-alpha.1/glance_1.0.0_amd64.deb)        | [v1.0.0-alpha.1](https://github.com/teamignitionvitc/Glance/releases/tag/v1.0.0-alpha.1) |
| **Linux (AppImage)**   | [![Linux](https://img.shields.io/badge/Linux-Download_.AppImage-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/teamignitionvitc/Glance/releases/download/v1.0.0-alpha.1/Glance-1.0.0-x86_64.AppImage)        | [v1.0.0-alpha.1](https://github.com/teamignitionvitc/Glance/releases/tag/v1.0.0-alpha.1) |

> **📘 Full Documentation:** For detailed setup guides and troubleshooting, see the [Official Documentation](docs/index.html).

### Method 2: Build from Source (Developers)

<details>
<summary><h3>Standard Installation</h3></summary>

```bash
# Clone the repository
git clone https://github.com/teamignitionvitc/Glance.git
cd Glance

# Install dependencies
pip install -r requirements.txt

# Launch application
python main.py
```

</details>

<details>
<summary><h3>Virtual Environment (Recommended)</h3></summary>

```bash
# Clone repository
git clone https://github.com/teamignitionvitc/Glance.git
cd Glance

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
python main.py
```

</details>

### Dependencies

<table>
<tr>
<td width="50%">

**Required**

```
PySide6>=6.4.0          # Qt6 UI framework
pyqtgraph>=0.13.0       # High-performance plotting
numpy>=1.21.0           # Numerical computing
pyserial>=3.5           # Serial communication
```

</td>
<td width="50%">

**ML Analysis tab**

```
torch                   # autoencoder + LSTM (inference and in-app training)
catboost                # health regressor
scikit-learn            # Gaussian Process
scipy                   # GP solver
joblib                  # model bundle I/O
pandas                  # dataset loading / column auto-detection
onnxruntime             # optional, for the offline ONNX artifacts
```

**Optional**

```
PySide6-WebEngine       # GPS map widgets
reportlab               # PDF report generation
```

**Installation:**

```bash
pip install PySide6-WebEngine reportlab
```

> The ML Analysis tab trains and runs models in-process, so `torch`, `catboost`,
> `scikit-learn` and `pandas` are needed for that tab to work. The rest of Glance runs
> without them. Rebuilding the offline benchmark artifacts additionally needs `onnx`
> and `lightgbm` — see `requirements.txt`.

</td>
</tr>
</table>

---

## Quick Start

<details open>
<summary><h3>1. Launch Application</h3></summary>

```bash
python main.py
```

</details>

<details>
<summary><h3>2. Choose Your Path</h3></summary>

<table>
<tr>
<td width="50%">

**First-Time Users**

1. Click **"Create New Dashboard"**
2. Follow the 4-phase wizard:
   - **Setup**: Configure connection
   - **Parameters**: Define data channels
   - **Widgets**: Pre-configure displays
   - **Dashboard**: Live monitoring

</td>
<td width="50%">

**Returning Users**

1. Click **"Load Project..."**
2. Select your `.json` configuration
3. Dashboard loads with:
   - Previous layout
   - Connection settings
   - Filter configurations
   - Logging preferences

</td>
</tr>
</table>

</details>

<details>
<summary><h3>3. Configure Connection</h3></summary>

<table>
<tr>
<th>Mode</th>
<th>Configuration</th>
</tr>
<tr>
<td><strong>Dummy (Testing)</strong></td>
<td>

```
Connection Mode: dummy
Data Format: json_array
Channel Count: 32
```

No hardware required

</td>
</tr>
<tr>
<td><strong>Serial</strong></td>
<td>

```
Connection Mode: serial
Serial Port: COM4 (auto-detected)
Baudrate: 115200
Data Format: json_array
Channel Count: 32
```

Use Refresh button to update port list

</td>
</tr>
<tr>
<td><strong>TCP/UDP</strong></td>
<td>

```
Connection Mode: tcp (or udp)
Host: 192.168.1.100
Port: 9000
Data Format: json_array
Channel Count: 32
```

Verify firewall settings

</td>
</tr>
</table>

</details>

<details>
<summary><h3>4. Define Parameters</h3></summary>

Navigate to **Edit → Manage Parameters** or press **Ctrl+P**

<table>
<tr>
<th>Property</th>
<th>Description</th>
<th>Example</th>
</tr>
<tr>
<td><strong>ID</strong></td>
<td>Unique identifier (no spaces)</td>
<td><code>temp_sensor</code></td>
</tr>
<tr>
<td><strong>Name</strong></td>
<td>Display name</td>
<td><code>Temperature</code></td>
</tr>
<tr>
<td><strong>Array Index</strong></td>
<td>Position in data packet (0-based)</td>
<td><code>0</code></td>
</tr>
<tr>
<td><strong>Sensor Group</strong></td>
<td>Category</td>
<td><code>Environmental</code></td>
</tr>
<tr>
<td><strong>Unit</strong></td>
<td>Measurement unit</td>
<td><code>°C</code></td>
</tr>
<tr>
<td><strong>Thresholds</strong></td>
<td>Alarm levels (ascending order)</td>
<td>Low Crit: -10<br/>Low Warn: 0<br/>High Warn: 80<br/>High Crit: 100</td>
</tr>
</table>

**Key Concept**: Array Index maps each parameter to a position in your incoming data array.

Example: If device sends `[23.5, 65.2, 1013.25, ...]`:

- Index 0 = Temperature (23.5)
- Index 1 = Humidity (65.2)
- Index 2 = Pressure (1013.25)

</details>

<details>
<summary><h3>5. Add Signal Filters (Optional)</h3></summary>

Navigate to **Filters → Add Filter to Parameter**

<table>
<tr>
<th>Filter Type</th>
<th>Use Case</th>
<th>Configuration</th>
</tr>
<tr>
<td><strong>Moving Average</strong></td>
<td>General smoothing</td>
<td>Window Size: 5-20</td>
</tr>
<tr>
<td><strong>Low Pass</strong></td>
<td>Fast response smoothing</td>
<td>Alpha: 0.3 (lower = smoother)</td>
</tr>
<tr>
<td><strong>Kalman</strong></td>
<td>Optimal estimation</td>
<td>Process Var: 0.01<br/>Measurement Var: 0.1</td>
</tr>
<tr>
<td><strong>Median</strong></td>
<td>Outlier rejection</td>
<td>Window Size: 5-11 (odd)</td>
</tr>
</table>

Filters can be chained and toggled without restarting.

</details>

<details>
<summary><h3>6. Add Widgets</h3></summary>

Click **"Add Widget..."** or press **Ctrl+W**

<table>
<tr>
<th>Widget Type</th>
<th>Parameters</th>
<th>Best For</th>
</tr>
<tr>
<td>Instant Value</td>
<td>1+</td>
<td>Large display of current value(s)</td>
</tr>
<tr>
<td>Time Graph</td>
<td>1+</td>
<td>Trend analysis</td>
</tr>
<tr>
<td>Gauge</td>
<td>1</td>
<td>Visual range indication</td>
</tr>
<tr>
<td>Histogram</td>
<td>1</td>
<td>Distribution analysis</td>
</tr>
<tr>
<td>LED Indicator</td>
<td>1+</td>
<td>Multi-status monitoring with per-parameter thresholds</td>
</tr>
<tr>
<td>Log Table</td>
<td>1+</td>
<td>Detailed inspection</td>
</tr>
<tr>
<td>Map</td>
<td>2 (Lat, Lon)</td>
<td>GPS tracking</td>
</tr>
</table>

**Widget Management**:

- Right-click for context menu
- Rename, Float/Dock, Close options
- Tile Evenly for auto-arrangement

</details>

<details>
<summary><h3>7. Start Data Logging</h3></summary>

Navigate to **Data Logging → Configure Logging...** or press **Ctrl+L**

<table>
<tr>
<th>Setting</th>
<th>Options</th>
<th>Recommendation</th>
</tr>
<tr>
<td>Format</td>
<td>CSV, JSON</td>
<td>CSV for Excel/MATLAB<br/>JSON for programming</td>
</tr>
<tr>
<td>Parameters</td>
<td>Select from list</td>
<td>Choose relevant parameters only</td>
</tr>
<tr>
<td>Buffer Size</td>
<td>10-1000</td>
<td>100 for most applications</td>
</tr>
<tr>
<td>File Path</td>
<td>Auto or custom</td>
<td>Auto-generated in logs/ folder</td>
</tr>
</table>

Start logging with **Ctrl+Shift+L**, stop with **Ctrl+Alt+L**

</details>

<details>
<summary><h3>8. Generate PDF Reports</h3></summary>

After logging data, generate professional PDF summary reports:

Navigate to **Data Logging → Generate Summary Report...** or press **Ctrl+Shift+R**

**Report Features:**

- Session metadata (duration, sample count, file size)
- Statistical summary for each parameter (min, max, average)
- Visual charts and graphs
- Professional formatting
- Export to PDF format

**Requirements:**

- Requires `reportlab` library (install with `pip install reportlab`)
- Must have logged data available

</details>

<details>
<summary><h3>9. Monitor Raw Data</h3></summary>

Open **View → Raw Telemetry Monitor** or press **Ctrl+M**

<table>
<tr>
<th>Display Mode</th>
<th>Format</th>
<th>Use Case</th>
</tr>
<tr>
<td><strong>Decimal</strong></td>
<td>`[12, 255, 0]`</td>
<td>Standard byte values</td>
</tr>
<tr>
<td><strong>Hex</strong></td>
<td>`0C FF 00`</td>
<td>Protocol debugging</td>
</tr>
<tr>
<td><strong>ASCII</strong></td>
<td>`...`</td>
<td>Text-based protocols</td>
</tr>
<tr>
<td><strong>Binary</strong></td>
<td>`00001100 ...`</td>
<td>Bit-level analysis</td>
</tr>
</table>

</details>

<details>
<summary><h3>10. Design a Rocket & Simulate Flight</h3></summary>

Navigate to **View → ML Analysis** or press **Ctrl+Shift+M**, then navigate to the **Rocket Flight Simulation** source:

1. **Adjust Geometry**: In the 3D airframe viewport, modify the nose cone shape (ogive, conical, parabolic, elliptical), length, body tube diameter, and fin dimensions.
2. **Review Aerodynamics**: Verify the Center of Pressure (CP), Center of Gravity (CG), and static stability margin (aim for 1.5–2.5 calibers).
3. **Simulate Flight**: Click **Start Flight Simulation** to run the 3-DOF aerodynamic trajectory and generate synthetic 13-channel telemetry.
4. **Inspect Closed-Loop Effects**: Notice how changes to the airframe shape immediately alter the flight dynamics and propagate into the ML analysis models.

</details>

<details>
<summary><h3>11. Run Real-Time ML Predictive Analysis</h3></summary>

From the **ML Analysis** window (`Ctrl+Shift+M`):

1. **Select Model**:
   - Choose a bundled model: **Pretrained · Rocket flight** or **Pretrained · NASA C-MAPSS turbofan**.
   - Or select **Train on my dataset…** to load your own CSV file.
2. **Choose Source**:
   - Select **Rocket flight simulation**, **Live telemetry** (Serial/TCP/UDP stream), or **Dataset replay**.
3. **Live Verdicts**:
   - **Autoencoder**: Monitor real-time anomaly scores and channel residuals.
   - **Gaussian Process**: Observe live trajectory forecasting with 95% confidence intervals and dynamic RMSE scoring.
   - **LSTM**: Check real-time health stage classification (*Healthy* → *Degrading* → *Critical*).
   - **CatBoost**: Track the overall 0–100 health index and view top driving features in the **Model & Insights** tab.

</details>

---

## Keyboard Shortcuts

Boost your productivity with these keyboard shortcuts:

<table>
<tr>
<th>Category</th>
<th>Action</th>
<th>Shortcut</th>
</tr>
<tr>
<td rowspan="6"><strong>File Operations</strong></td>
<td>New Dashboard</td>
<td><code>Ctrl + N</code></td>
</tr>
<tr>
<td>Load Project</td>
<td><code>Ctrl + O</code></td>
</tr>
<tr>
<td>Save Project</td>
<td><code>Ctrl + S</code></td>
</tr>
<tr>
<td>Save Project As</td>
<td><code>Ctrl + Shift + S</code></td>
</tr>
<tr>
<td>Connection Settings</td>
<td><code>Ctrl + Shift + C</code></td>
</tr>
<tr>
<td>Exit Application</td>
<td><code>Ctrl + Q</code></td>
</tr>
<tr>
<td rowspan="3"><strong>Edit</strong></td>
<td>Manage Parameters</td>
<td><code>Ctrl + P</code></td>
</tr>
<tr>
<td>Add Widget</td>
<td><code>Ctrl + W</code></td>
</tr>
<tr>
<td>Remove Widget</td>
<td><code>Ctrl + Shift + W</code></td>
</tr>
<tr>
<td rowspan="6"><strong>View &amp; Navigation</strong></td>
<td>Add New Tab</td>
<td><code>Ctrl + T</code></td>
</tr>
<tr>
<td>Rename Current Tab</td>
<td><code>Ctrl + R</code></td>
</tr>
<tr>
<td>Float/Dock Current Tab</td>
<td><code>Ctrl + Shift + F</code></td>
</tr>
<tr>
<td>Raw Telemetry Monitor</td>
<td><code>Ctrl + M</code></td>
</tr>
<tr>
<td>ML Analysis</td>
<td><code>Ctrl + Shift + M</code></td>
</tr>
<tr>
<td>Toggle Fullscreen</td>
<td><code>F11</code></td>
</tr>
<tr>
<td rowspan="2"><strong>Stream Control</strong></td>
<td>Pause/Resume Stream</td>
<td><code>Spacebar</code></td>
</tr>
<tr>
<td>Exit Fullscreen</td>
<td><code>ESC</code></td>
</tr>
<tr>
<td rowspan="5"><strong>Data & Filters</strong></td>
<td>Configure Logging</td>
<td><code>Ctrl + L</code></td>
</tr>
<tr>
<td>Start Data Logging</td>
<td><code>Ctrl + Shift + L</code></td>
</tr>
<tr>
<td>Stop Data Logging</td>
<td><code>Ctrl + Alt + L</code></td>
</tr>
<tr>
<td>Generate Summary Report</td>
<td><code>Ctrl + Shift + R</code></td>
</tr>
<tr>
<td>Manage Filters</td>
<td><code>Ctrl + F</code></td>
</tr>
<tr>
<td rowspan="2"><strong>Undo/Redo</strong></td>
<td>Undo</td>
<td><code>Ctrl + Z</code></td>
</tr>
<tr>
<td>Redo</td>
<td><code>Ctrl + Shift + Z</code> or <code>Ctrl + Y</code></td>
</tr>
<tr>
<td><strong>Help</strong></td>
<td>Show Keyboard Shortcuts</td>
<td><code>F1</code></td>
</tr>
</table>

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the **GNU General Public License v3.0**. See `LICENSE` for more information.

---

<div align="center">
<p>Copyright © 2025 Team Ignition Software Department</p>
</div>
