# TrueTone UI direction

A research listening tool, using the project's forest, ivory and burnt-orange palette. The interface should feel calm and editorial, with the recording and its result given equal space.

`style.css` is the source of truth for colors, typography, spacing and component treatments. Georgia introduces the subject and result; Arial keeps controls legible. Orange is reserved for small emphasis and focus. Components use modest corners, thin neutral borders, and no decorative shadows. OLED dark mode uses a pure black (#000000) page, neutral charcoal (#101010) surfaces and light text, without green-tinted backgrounds. The header toggle switches in place; keep current input and results intact. Small screens stack input before output; reduced-motion preferences disable transitions.

Use plain task language. Keep scores, transcripts and limitations accurate. Never add invented confidence, example waveforms or model explanations. Advanced settings belong behind a disclosure.

Direction informed by Brad Hussey's article: https://bradhussey.ca/web-design/make-ai-built-website-not-look-ai-generated/

The shared logo lives in `assets/`. Use the speech-bubble/waveform icon beside the TrueTone wordmark, as the browser favicon, and for the Desktop app and startup screen. Keep its aspect ratio and built-in safe margin.

The workspace is fluid across the available window width, with modest responsive edge padding. Avoid fixed desktop width caps that create large fullscreen gutters. Keep detailed emotion scores behind a disclosure. Distinguish classifier predictions from the separate polarity-contrast heuristic and emotion category matches.

Keep the introduction compact: no forced headline line break, a 42 px desktop heading cap, vertically centered instructions and small vertical padding. Recording and results take priority over promotional header space.
