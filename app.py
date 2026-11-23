"""TrueTone local microphone demo. Run with: python app.py"""
from functools import lru_cache
import base64
import os
from pathlib import Path

from inference import Predictor
import gradio as gr


@lru_cache(maxsize=1)
def predictor():
    return Predictor()


def analyze(audio, transcript, model, progress=gr.Progress()):
    try:
        score, words, duration, voice, emotion_scores, contrast = predictor().predict(audio, transcript, model, progress)
    except (ValueError, RuntimeError, OSError) as error:
        raise gr.Error(str(error)) from error
    verdict = 'Sarcastic' if score >= .5 else 'Not sarcastic'
    heading = 'Conflicting cues — review context' if contrast and score < .5 else f'Model predicts: {verdict.lower()}'
    details = (f'### {heading}\n'
               f'**Classifier prediction:** {verdict}\n\n'
               f'**Sarcasm score:** {score:.1%}\n\n'
               f'{duration:.1f}s recording · {model}\n\n'
               'A model estimate, not a verified label. Sarcasm can depend on context beyond this clip.')
    if contrast:
        details += (f'\n\n**Cue contrast:** {contrast}. This can occur in sarcasm. '
                    'This is a polarity heuristic, not a second classifier or a verified label.')
    return {'Sarcastic': score, 'Not sarcastic': 1-score}, words, details, voice, emotion_scores


ASSETS = Path(__file__).resolve().parent / 'assets'
LOGO_DATA = base64.b64encode((ASSETS / 'logo-96.png').read_bytes()).decode('ascii')
CSS = (Path(__file__).resolve().parent / 'ui/style.css').read_text()
THEME_JS = (Path(__file__).resolve().parent / 'ui/theme.js').read_text()
EMPTY_RESULT = '### No result yet\nRecord or upload a clip, then choose **Analyze recording**.'
THEME = gr.themes.Base(primary_hue='emerald', neutral_hue='stone', font=['Arial', 'sans-serif']).set(
    body_background_fill='var(--tt-paper)', body_background_fill_dark='var(--tt-paper)',
    block_background_fill='var(--tt-surface)', block_background_fill_dark='var(--tt-surface)',
    input_background_fill='var(--tt-surface)', input_background_fill_dark='var(--tt-surface)',
    block_label_background_fill='transparent', block_label_background_fill_dark='transparent')


def create_app():
    with gr.Blocks(title='TrueTone — Speech & Sarcasm', analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.HTML("""<header class="tt-masthead">
          <div class="tt-wordmark"><img class="tt-logo" src="data:image/png;base64,LOGO_DATA" alt="" aria-hidden="true">TrueTone</div>
          <nav class="tt-masthead-right" aria-label="Project"><span class="tt-project">PID-02</span>
          <button id="theme-toggle" type="button" role="switch" aria-label="Dark mode" aria-checked="false" title="Switch to dark mode">
            <span class="tt-switch-thumb" aria-hidden="true">
              <svg class="tt-switch-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
              <svg class="tt-switch-moon" viewBox="0 0 24 24" fill="currentColor"><path d="M20.8 14.2A9 9 0 0 1 9.8 3.2a9 9 0 1 0 11 11Z"/></svg>
            </span>
          </button>
          <a href="https://github.com/raghavvxts/TrueTone" target="_blank" rel="noopener noreferrer">View source</a></nav>
        </header>
        <section class="tt-hero" aria-labelledby="hero-title">
          <div><p class="tt-eyebrow">SPEECH &amp; SARCASM</p><h1 id="hero-title">A closer listen to sarcasm.</h1></div>
          <div class="tt-intro"><p>Record a line and see how the model reads its words and delivery.</p>
          <p class="tt-format">1–30 seconds / English speech</p></div>
        </section>""".replace('LOGO_DATA', LOGO_DATA), js_on_load=THEME_JS)
        with gr.Row(elem_id='workspace'):
            with gr.Column(min_width=280, elem_id='input-pane'):
                gr.HTML('<div class="tt-section-heading"><h2>Your recording</h2><span>Record or upload</span></div>')
                audio = gr.Audio(sources=['microphone', 'upload'], type='numpy', format='wav',
                                 label='Audio', show_label=False, elem_id='recording')
                with gr.Accordion('Adjust transcript or model', open=False, elem_id='options'):
                    transcript = gr.Textbox(label='Transcript (optional)', placeholder='Leave blank to transcribe automatically, or enter the exact spoken words.', lines=2, max_lines=5)
                    model = gr.Radio(['Text + speech', 'Speech reference'], value='Text + speech', label='Model')
                with gr.Row():
                    submit = gr.Button('Analyze recording', variant='primary', scale=3, min_width=180, elem_id='analyze')
                    clear = gr.ClearButton(value='Clear', scale=1, min_width=80, elem_id='clear')
                gr.Markdown('Stop recording before analyzing. Allow microphone access in your browser. Use the transcript option to correct any speech recognition errors.')
            with gr.Column(min_width=280, elem_id='output-pane'):
                gr.HTML('<div class="tt-section-heading"><h2>The model’s reading</h2><span>Model estimate</span></div>')
                result = gr.Markdown(EMPTY_RESULT, elem_id='result')
                scores = gr.Label(label='Model scores', num_top_classes=2, show_heading=False, container=False, elem_id='scores')
                voice = gr.Markdown('', elem_id='voice-profile')
                with gr.Accordion('Voice emotion scores', open=False, elem_id='emotion-details'):
                    emotion_scores = gr.Label(label='Voice emotion scores', num_top_classes=4, show_heading=False, container=False, elem_id='emotion-scores')
                recognized = gr.Textbox(label='Transcript used', placeholder='The spoken words will appear here.', interactive=False, lines=2, elem_id='recognized')
        gr.HTML("""<div class="tt-footnote"><p>MUStARD++ interface preview. Integration of the externally trained model is pending. Scores currently come from the local demo model.</p>
          <p class="tt-local">Audio processed locally<br>First use downloads pretrained models</p></div>""")
        with gr.Accordion('About the model and its limits', open=False, elem_id='method'):
            gr.Markdown('The local demo currently uses a classifier trained on the original MUStARD dataset. The externally trained MUStARD++ model has not yet been connected on this machine.\n\nLocal speech recognition supplies the transcript. The research notebook’s frozen encoders extract speech, prosody, emotion, text and sentiment features. A final logistic-regression classifier predicts sarcasm at a 0.5 threshold. Scores are uncalibrated. The speech reference also uses transcript word counts for speaking-rate features. This is a recording-based demo, not continuous streaming.\n\n[Project and evaluation results](https://github.com/raghavvxts/TrueTone)')
        outputs = [scores, recognized, result, voice, emotion_scores]
        prediction = submit.click(analyze, inputs=[audio, transcript, model], outputs=outputs, api_name='analyze', concurrency_limit=1, show_progress_on=[result])
        audio.change(lambda: ('', None, '', EMPTY_RESULT, '', None), outputs=[transcript, *outputs], cancels=[prediction], queue=False)
        for control in (transcript, model):
            control.input(lambda: (None, '', EMPTY_RESULT, '', None), outputs=outputs, cancels=[prediction], queue=False)
        clear.add([audio, transcript, *outputs])
    return demo


if __name__ == '__main__':
    create_app().queue(max_size=4).launch(server_name='127.0.0.1', server_port=int(os.environ.get('TRUETONE_PORT', '7860')),
        share=False, max_file_size='15mb', theme=THEME, css=CSS, favicon_path=ASSETS / 'favicon.ico',
        allowed_paths=[str(ASSETS)], head=f'<link rel="apple-touch-icon" href="/gradio_api/file={ASSETS / "apple-touch-icon.png"}"><meta name="theme-color" content="#203c34">')
