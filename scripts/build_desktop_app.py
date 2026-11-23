"""Build the local macOS wrapper. No Python/model files are copied into the app."""
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    contents = destination / 'Contents'
    (contents / 'MacOS').mkdir(parents=True, exist_ok=True)
    (contents / 'Resources').mkdir(exist_ok=True)
    (contents / 'Resources/project-path.txt').write_text(str(ROOT))
    shutil.copy2(ROOT / 'assets/TrueTone.icns', contents / 'Resources/TrueTone.icns')
    shutil.copy2(ROOT / 'assets/logo-192.png', contents / 'Resources/logo.png')
    info = {
        'CFBundleExecutable': 'TrueTone', 'CFBundleIdentifier': 'com.truetone.desktop',
        'CFBundleName': 'TrueTone', 'CFBundleDisplayName': 'TrueTone',
        'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': '1.0',
        'CFBundleIconFile': 'TrueTone.icns', 'CFBundleVersion': '2', 'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'TrueTone uses your microphone to analyze recordings locally on your Mac.',
    }
    with (contents / 'Info.plist').open('wb') as file:
        plistlib.dump(info, file)
    subprocess.run(['xcrun', 'swiftc', str(ROOT / 'desktop/TrueTone.swift'),
                    '-o', str(contents / 'MacOS/TrueTone'), '-framework', 'Cocoa', '-framework', 'WebKit', '-framework', 'AVFoundation'], check=True)
    subprocess.run(['codesign', '--force', '--sign', '-', '--entitlements',
                    str(ROOT / 'desktop/TrueTone.entitlements'), str(destination)], check=True)
    # Refresh Finder after replacing an existing app's icon resources.
    subprocess.run(['/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister',
                    '-f', str(destination)], check=True)
    destination.touch()
    print(destination)


if __name__ == '__main__':
    if sys.platform != 'darwin':
        raise SystemExit('The Desktop wrapper requires macOS and Xcode Command Line Tools.')
    build(Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / 'Desktop/TrueTone.app')
