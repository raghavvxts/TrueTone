"""TrueTone local microphone demo. Run with: python app.py"""
from functools import lru_cache
import os

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
               f'**Sarcasm score:** {score:.1%} · **Recording:** {duration:.1f}s\n\n'
               f'**Model:** {model}, logistic regression trained on MUStARD. '
               'This score is an uncalibrated model estimate, not certainty.')
    return {'Sarcastic': score, 'Not sarcastic': 1-score}, words, details


CSS = '''
.gradio-container {max-width:1120px!important;margin:auto!important;}
#brand h1 {font-size:3.4rem;letter-spacing:-.05em;color:#173f35;margin-bottom:.2rem;}
#brand p {font-size:1.1rem;color:#435d52;}
.dark #brand h1 {color:#b6dec3;}
.dark #brand p {color:#c2d0c5;}
.dark textarea::placeholder {color:#b9b9b3!important;opacity:1;}
#result {border-left:4px solid #c8784d;padding-left:1rem;min-height:8rem;}
footer {display:none!important;}
'''


def create_app():
    with gr.Blocks(title='TrueTone — Speech & Sarcasm', analytics_enabled=False, delete_cache=(3600, 3600)) as demo:
        gr.Markdown('# TrueTone\nHear the words. Read the tone.', elem_id='brand')
        gr.Markdown('Record **1–30 seconds of English speech** or upload a clip. Stop recording, then choose **Analyze recording**.')
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=['microphone', 'upload'], type='numpy', format='wav',
                                 label='Your recording')
                transcript = gr.Textbox(label='Transcript (optional)', placeholder='Leave blank for automatic transcription. Enter the spoken words to correct recognition errors.', lines=3, max_lines=5)
                model = gr.Radio(['Text + speech', 'Speech reference'], value='Text + speech', label='Model')
                with gr.Row():
                    submit = gr.Button('Analyze recording', variant='primary')
                    clear = gr.ClearButton(value='Clear')
            with gr.Column():
                result = gr.Markdown('### Ready when you are\nYour result will appear here.', elem_id='result')
                scores = gr.Label(label='Model scores', num_top_classes=2)
                recognized = gr.Textbox(label='Transcript used', interactive=False, lines=3)
        gr.Markdown('**Research demo:** trained on television dialogue. Everyday speech, accents, background noise and missing context can change the result. Audio is processed on this computer; the first run downloads pretrained models.')
        with gr.Accordion('How it works', open=False):
            gr.Markdown('Local speech recognition supplies the transcript. The research notebook’s frozen encoders extract speech, prosody, emotion, text and sentiment features. A final logistic-regression classifier predicts sarcasm at a 0.5 threshold. The speech reference also uses transcript word counts for speaking-rate features.\n\n[Project and evaluation results](https://github.com/raghavvxts/TrueTone)')
        prediction = submit.click(analyze, inputs=[audio, transcript, model], outputs=[scores, recognized, result], api_name='analyze', concurrency_limit=1)
        audio.change(lambda: ('', None, '', '### Ready when you are'), outputs=[transcript, scores, recognized, result], cancels=[prediction], queue=False)
        clear.add([audio, transcript, scores, recognized, result])
    return demo


if __name__ == '__main__':
    create_app().queue(max_size=4).launch(server_name='127.0.0.1', server_port=int(os.environ.get('TRUETONE_PORT', '7860')),
        share=False, max_file_size='15mb', theme=gr.themes.Soft(primary_hue='emerald', neutral_hue='stone'), css=CSS)
