"""Lightweight omni.ui panel for grasp-copilot assistance."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence


class AssistUI:
    """Small omni.ui helper that mirrors gui_assist_demo semantics."""

    def __init__(
        self,
        *,
        on_ask: Callable[[], None],
        on_reset: Callable[[], None],
        on_choice: Callable[[str], None],
        on_mode_change: Callable[[str], None],
        show_logs: bool = False,
        initial_mode: str = "translate",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._on_ask = on_ask
        self._on_reset = on_reset
        self._on_choice = on_choice
        self._on_mode_change = on_mode_change
        self._show_logs = bool(show_logs)
        # Internal mode values match controllers.ModeManager / controller.set_mode:
        # "translate" | "rotate" | "gripper"
        self._mode = str(initial_mode).lower().strip() or "translate"
        self._choices: Sequence[str] = []
        # Track created omni.ui widgets explicitly since some omni.ui container
        # types (e.g. VStack) don't expose a stable `.children` API across versions.
        self._choice_widgets: List[object] = []
        self._log: List[str] = []
        self._ui = None
        if enabled:
            self._build_ui()

    def _build_ui(self) -> None:
        import omni.ui as ui  # type: ignore

        self._ui = ui
        # Main control window (user-facing).
        self.window = ui.Window("Grasp Copilot", width=380, height=340)
        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 10}):
                self.status_label = ui.Label("Status: idle", word_wrap=True)

                with ui.HStack(height=28):
                    ui.Label("Mode:", width=60)
                    self._mode_buttons = {}
                    for mode_value, label in (("translate", "Translate"), ("rotate", "Rotate"), ("gripper", "Gripper")):
                        btn = ui.Button(label, width=90, clicked_fn=lambda mm=mode_value: self._set_mode(mm))
                        self._mode_buttons[mode_value] = btn

                with ui.HStack(spacing=8):
                    ui.Button("Ask assistance", height=32, clicked_fn=self._on_ask)
                    ui.Button("Reset", height=32, clicked_fn=self._on_reset)

                ui.Label("Choices")
                # Keep choices as plain buttons (no scrolling). To avoid stale “black line”
                # artifacts when rebuilding, we recreate the container stack on each update.
                self._choice_frame = ui.Frame(height=0)
                with self._choice_frame:
                    self._choice_stack = ui.VStack(spacing=4, style={"margin": 4})

                # Optional workspace hint (no grid UI; keep only a compact bounds hint).
                self._bounds_label = ui.Label("", word_wrap=True)

        # Apply initial highlight
        self.set_mode(self._mode)

        # Separate log window (debug/engineer-facing).
        self.log_window = None
        self._log_frame = None
        self._log_stack = None
        if self._show_logs:
            self.log_window = ui.Window("Grasp Copilot Logs", width=520, height=420)
            with self.log_window.frame:
                with ui.VStack(spacing=6, style={"margin": 10}):
                    ui.Label("Log")
                    self._log_frame = ui.ScrollingFrame(height=360, style={"background_color": 0x151515ff})
                    with self._log_frame:
                        self._log_stack = ui.VStack(spacing=2, style={"margin": 6})

    def set_status(self, text: str) -> None:
        self._log.append(text)
        if len(self._log) > 200:
            self._log = self._log[-200:]
        if not self.enabled or self._ui is None:
            return
        self.status_label.text = f"Status: {text}"
        # Append to log area (if enabled)
        if self._show_logs and self._log_stack is not None:
            with self._log_stack:
                self._ui.Label(text, word_wrap=True)

    def set_bounds_hint(self, workspace_min, workspace_max) -> None:
        if not self.enabled or self._ui is None:
            return
        self._bounds_label.text = f"Workspace X[{workspace_min[0]:.2f},{workspace_max[0]:.2f}] Y[{workspace_min[1]:.2f},{workspace_max[1]:.2f}]"

    def _set_mode(self, mode: str) -> None:
        # Called from UI button clicks (should also notify controller).
        self.set_mode(mode)
        try:
            self._on_mode_change(self._mode)
        except Exception:
            pass

    def set_mode(self, mode: str) -> None:
        """Update the UI highlight to reflect the current mode (no controller side-effects)."""
        m = str(mode).lower().strip()
        if m in {"translation"}:
            m = "translate"
        if m in {"rotation"}:
            m = "rotate"
        if m not in {"translate", "rotate", "gripper"}:
            return
        self._mode = m
        if not self.enabled or self._ui is None:
            return
        # Highlight active mode button.
        active_style = {
            "background_color": 0x2D6CDFff,  # blue-ish
            "color": 0xFFFFFFFF,
            "border_width": 0,
        }
        inactive_style = {
            "background_color": 0x252525ff,
            "color": 0xDDDDDDff,
            "border_width": 1,
        }
        for mv, btn in (self._mode_buttons or {}).items():
            try:
                btn.style = active_style if mv == self._mode else inactive_style
            except Exception:
                # Some omni.ui builds don't expose `.style` as settable; ignore.
                pass

    def set_choices(self, choices: Sequence[str]) -> None:
        self._choices = choices
        if not self.enabled or self._ui is None:
            return
        ui = self._ui
        # Destroy any previous choice widgets.
        for w in list(self._choice_widgets):
            try:
                w.destroy()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._choice_widgets = []

        # Recreate the container stack to avoid visual artifacts in some omni.ui builds.
        try:
            self._choice_stack.destroy()  # type: ignore[attr-defined]
        except Exception:
            pass
        with self._choice_frame:
            self._choice_stack = ui.VStack(spacing=4, style={"margin": 4})

        try:
            self._choice_stack.visible = False
            self._choice_stack.visible = True
        except Exception:
            pass
        with self._choice_stack:
            for ch in choices:
                btn = ui.Button(
                    ch,
                    height=28,
                    clicked_fn=lambda cc=ch: self._on_choice(cc),
                    style={
                        "margin": 2,
                        # Best-effort: keep borders subtle to avoid “black line” artifacts.
                        "border_width": 1,
                    },
                )
                self._choice_widgets.append(btn)
