# TrueTone OLED logo

The existing speech-bubble and seven-bar waveform shape is preserved. The active palette is OLED black with a near-white symbol; the former green and terracotta colors are removed from the logo. The website still starts in light mode, with its existing OLED dark-mode toggle.

- `logo.png`: generated OLED master, 1,254 px square.
- `logo-96.png`: website header.
- `logo-192.png`: native startup screen, README and reusable web icon.
- `apple-touch-icon.png`: 180 px web shortcut icon.
- `favicon.ico`: browser/address-bar icon, containing 16, 32 and 48 px sizes.
- `TrueTone.icns`: macOS icon set; the installed bundle uses `TrueTone-OLED.icns` to refresh the previous icon reference.
- `archive/logo-forest.png`: preserved original master.
- `logo-oled-prompt.txt`: full edit prompt and generation provenance.

Created with the built-in image generation tool using the existing logo as the edit target. Export derivatives with `.venv/bin/python scripts/export_logo.py`. Browser icon URLs include a content hash to refresh cached icons.
