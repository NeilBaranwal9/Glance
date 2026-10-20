import sys, traceback
try:
    sys.argv = ['test']
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    from app.ui.main_window import MainWindow
    w = MainWindow()
    p = w.ml_analysis_page
    # The default model loads on a deferred timer, so let the loop run.
    for _ in range(40):
        app.processEvents()

    if p._models is None:
        print("ML page loaded but no pretrained bundle found; "
              "run: python ml/build_pretrained.py")
    else:
        # Exercise the live path: simulate a flight and stream a few samples through.
        p._prepare_flight()
        p._running = True   # _tick_stream is a no-op unless the run is active
        for _ in range(60):
            p._tick_stream()
        p.stop()
        s = p._engine.live_summary()
        print(f"ML page ok: {p._models.n_features} channels, "
              f"{s['samples']} samples streamed, health {s['health']:.0f}, stage {s['stage']}")
        state = p.get_save_state()
        p.load_save_state(state)

    print("Methods ran successfully.")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
