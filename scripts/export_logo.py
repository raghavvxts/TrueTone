"""Export the generated master to browser and macOS icon formats (macOS only)."""
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


if __name__ == '__main__':
    with Image.open(ASSETS / 'logo.png') as image:
        image = image.convert('RGBA')
        if image.width != image.height:
            raise SystemExit('The logo master must be square.')
        for size, name in [(96, 'logo-96.png'), (192, 'logo-192.png'), (180, 'apple-touch-icon.png')]:
            image.resize((size, size), Image.Resampling.LANCZOS).save(ASSETS / name)
        image.save(ASSETS / 'favicon.ico', sizes=[(16, 16), (32, 32), (48, 48)])
        with tempfile.TemporaryDirectory() as directory:
            iconset = Path(directory) / 'TrueTone.iconset'
            iconset.mkdir()
            for size in (16, 32, 128, 256, 512):
                for scale in (1, 2):
                    pixels = size * scale
                    suffix = '@2x' if scale == 2 else ''
                    image.resize((pixels, pixels), Image.Resampling.LANCZOS).save(iconset / f'icon_{size}x{size}{suffix}.png')
            subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o', str(ASSETS / 'TrueTone.icns')], check=True)
