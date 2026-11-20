"""TrueTone local microphone demo. Run with: python app.py"""
from functools import lru_cache
import os
from pathlib import Path

from inference import Predictor
import gradio as gr


@lru_cache(maxsize=1)
def predictor():
    return Predictor()


def analyze(audio, transcript, model, progress=gr.Progress()):
    try:
        score, words, duration = predictor().predict(audio, transcript, model, progress)
    except (ValueError, RuntimeError, OSError) as error:
        raise gr.Error(str(error)) from error
    verdict = 'Sarcastic' if score >= .5 else 'Not sarcastic'
    details = (f'### {verdict}\n'
               f'**Sarcasm score:** {score:.1%}\n\n'
               f'{duration:.1f}s recording · {model}')
    return {'Sarcastic': score, 'Not sarcastic': 1-score}, words, details


CSS = (Path(__file__).resolve().parent / 'ui/style.css').read_text()
EMPTY_RESULT = '### No result yet\nRecord or upload a clip, then choose **Analyze recording**.'
THEME = gr.themes.Base(primary_hue='emerald', neutral_hue='stone', font=['Arial', 'sans-serif']).set(
    body_background_fill='var(--tt-paper)', body_background_fill_dark='var(--tt-paper)',
    block_background_fill='var(--tt-surface)', block_background_fill_dark='var(--tt-surface)',
    input_background_fill='var(--tt-surface)', input_background_fill_dark='var(--tt-surface)',
    block_label_background_fill='transparent', block_label_background_fill_dark='transparent')


def create_app():
    with gr.Blocks(title='TrueTone — Speech & Sarcasm', analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.HTML("""<header class="tt-masthead">
          <div class="tt-wordmark">TrueTone</div>
          <nav class="tt-masthead-right" aria-label="Project"><span class="tt-project">PID-02</span>
          <a href="https://github.com/raghavvxts/TrueTone" target="_blank" rel="noopener noreferrer">View source</a></nav>
        </header>
        <section class="tt-hero" aria-labelledby="hero-title">
          <div><p class="tt-eyebrow">SPEECH &amp; SARCASM</p><h1 id="hero-title">A closer listen<br>to sarcasm.</h1></div>
          <div class="tt-intro"><p>Record a line and see how the model reads its words and delivery.</p>
          <p class="tt-format">1–30 seconds / English speech</p></div>
        </section>""")
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
                gr.Markdown('Stop recording before analyzing. Use the transcript option to correct any speech recognition errors.')
            with gr.Column(min_width=280, elem_id='output-pane'):
                gr.HTML('<div class="tt-section-heading"><h2>The model’s reading</h2><span>Model estimate</span></div>')
                result = gr.Markdown(EMPTY_RESULT, elem_id='result')
                scores = gr.Label(label='Model scores', num_top_classes=2, show_heading=False, container=False, elem_id='scores')
                recognized = gr.Textbox(label='Transcript used', placeholder='The spoken words will appear here.', interactive=False, lines=2, elem_id='recognized')
        gr.HTML("""<div class="tt-footnote"><p>Research demo trained on 690 MUStARD clips. Scores are estimates; context, accents and background noise can change the reading.</p>
          <p class="tt-local">Audio processed locally<br>First use downloads pretrained models</p></div>""")
        with gr.Accordion('About the model and its limits', open=False, elem_id='method'):
            gr.Markdown('Local speech recognition supplies the transcript. The research notebook’s frozen encoders extract speech, prosody, emotion, text and sentiment features. A final logistic-regression classifier predicts sarcasm at a 0.5 threshold. Scores are uncalibrated. The speech reference also uses transcript word counts for speaking-rate features. This is a recording-based demo, not continuous streaming.\n\n[Project and evaluation results](https://github.com/raghavvxts/TrueTone)')
        prediction = submit.click(analyze, inputs=[audio, transcript, model], outputs=[scores, recognized, result], api_name='analyze', concurrency_limit=1)
        audio.change(lambda: ('', None, '', EMPTY_RESULT), outputs=[transcript, scores, recognized, result], cancels=[prediction], queue=False)
        clear.add([audio, transcript, scores, recognized, result])
    return demo


if __name__ == '__main__':
    create_app().queue(max_size=4).launch(server_name='127.0.0.1', server_port=int(os.environ.get('TRUETONE_PORT', '7860')),
        share=False, max_file_size='15mb', theme=THEME, css=CSS)
