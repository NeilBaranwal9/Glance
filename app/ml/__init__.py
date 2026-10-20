"""Runtime machine-learning engine for the ML Analysis tab.

Unlike ``ml/`` (offline training on the C-MAPSS benchmark), everything here is
**N-sensor agnostic**: the number and names of channels come from whatever data
source is attached, and the models are sized to match at construction time.

No Qt import belongs in this package -- it must stay testable headlessly and
usable from a worker thread.
"""
