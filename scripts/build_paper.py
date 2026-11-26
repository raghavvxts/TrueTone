"""Build the final two-column IEEE-style paper from verified evaluation CSVs.

Only the final PDF is written inside paper/. Source remains in this script.
"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figures import build_figures
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Table, TableStyle, Spacer, KeepTogether, Image
from reportlab.lib.utils import simpleSplit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/TrueTone_IEEE_Paper.pdf'
RESULTS=ROOT/'results/mustardpp'
W,H=letter;M=42;GAP=18;COL=(W-2*M-GAP)/2
styles={
    'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.7,alignment=TA_JUSTIFY,spaceAfter=6),
    'abstract':ParagraphStyle('abstract',fontName='Times-Bold',fontSize=9.2,leading=10.7,alignment=TA_JUSTIFY,spaceAfter=7),
    'section':ParagraphStyle('section',fontName='Times-Roman',fontSize=10,leading=12,alignment=TA_CENTER,spaceBefore=10,spaceAfter=6,keepWithNext=True),
    'sub':ParagraphStyle('sub',fontName='Times-Italic',fontSize=10,leading=12,spaceBefore=5,spaceAfter=4,keepWithNext=True),
    'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.3,leading=10,alignment=TA_CENTER,spaceBefore=5,spaceAfter=5,keepWithNext=True),
    'small':ParagraphStyle('small',fontName='Times-Roman',fontSize=8.3,leading=9.5,spaceAfter=5),
    'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=8.4,leading=9.8,spaceAfter=5,leftIndent=14,firstLineIndent=-14),
}
story=[]
def p(text,kind='body'):story.append(Paragraph(text,styles[kind]))
def sec(text):p(text,'section')
def sub(text):p(text,'sub')
def table(caption,rows,widths):
    cells=[[Paragraph(str(x),styles['small']) for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='CENTER')
    t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
    story.append(KeepTogether([Paragraph(caption,styles['caption']),t,Spacer(1,6)]))


def figure(filename,caption,height):
    story.append(KeepTogether([Image(str(ROOT/'results/paper'/filename),width=COL,height=height),
        Paragraph(caption,ParagraphStyle('figure-caption',parent=styles['caption'],keepWithNext=False)),Spacer(1,5)]))


def header(canvas,doc):
    canvas.saveState();canvas.setFillColor(colors.black)
    if doc.page==1:
        canvas.setFont('Times-Roman',19)
        canvas.drawCentredString(W/2,H-47,'TrueTone: Multimodal Sarcasm Detection')
        canvas.drawCentredString(W/2,H-69,'Using Text and Speech Prosody on MUStARD++')
        lines=[('Times-Roman',10,'Manas Sharma, Raghav Vats, Vaibhav Singh, and Sumit Bhati'),
               ('Times-Roman',9,'CSE Artificial Intelligence, Bennett University, Greater Noida, India'),
               ('Times-Roman',8.3,'Enrolments: S24CSEU2420, S24CSEU1665, S24CSEU1668, S24CSEU2870'),
               ('Times-Roman',9,'PID-02 | Instructor: Dr. Saurabh Kumar Srivastava')]
        for i,(font,size,text) in enumerate(lines):
            canvas.setFont(font,size);canvas.drawCentredString(W/2,H-91-i*13,text)
        canvas.setFont('Times-Roman',8.4)
        canvas.drawCentredString(W/2,H-148,'Code, models, presentation and paper: github.com/raghavvxts/TrueTone')
        canvas.linkURL('https://github.com/raghavvxts/TrueTone',(112,H-151,W-112,H-139),relative=0,thickness=0)
    canvas.setFont('Times-Roman',8)
    canvas.drawString(M,25,'TrueTone | PID-02 | MUStARD++ final project')
    canvas.drawRightString(W-M,25,str(doc.page))
    canvas.restoreState()


def build():
    story.clear()
    build_figures()
    summary=pd.read_csv(RESULTS/'summary.csv');held=pd.read_csv(RESULTS/'held_out_show.csv')
    def metric(training,mode,name='f1'):
        return float(summary.loc[(summary.training==training)&(summary['mode']==mode),name].iloc[0])
    baseline=metric('MUStARD','Text + speech');final=metric('MUStARD++','Text + speech')
    ci=pd.read_csv(RESULTS/'paired_bootstrap.csv').query("mode == 'Text + speech' and comparison == 'MUStARD++ minus MUStARD'").iloc[0]
    p(f'''Abstract - A transcript can preserve the words of an utterance while losing the delivery that changes their meaning. TrueTone combines frozen text and speech representations with prosody and text-voice valence features for sarcasm detection. The final application uses logistic-regression classifiers fitted on 1,202 balanced MUStARD++ targets; the original 690-target MUStARD release is retained as a training baseline. A controlled comparison evaluates both training conditions on the same 688 shared targets using repeated grouped cross-validation. Averaging each target's three out-of-fold probabilities gives sarcastic-class F1 of {final:.3f} for MUStARD++ and {baseline:.3f} for MUStARD. The observed difference is {ci.delta_f1:+.3f}, with a 95% paired group-bootstrap interval of [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}]. This interval includes zero. The results support using speech alongside text on this test population, while leaving the benefit of dataset expansion uncertain. Separate held-out-show tests expose substantial domain and class-balance limitations.''','abstract')
    p('<i>Index Terms - sarcasm detection, MUStARD++, multimodal learning, speech prosody, logistic regression.</i>', 'small')
    sec('I. INTRODUCTION')
    p("""The words "Wonderful, another meeting" can express approval or frustration. Pitch, timing and emphasis may change their interpretation, and the surrounding conversation can matter just as much. A system that reads only the transcript loses part of that evidence. A system that listens to the voice still cannot directly observe the speaker\'s intent.""")
    p("""TrueTone studies two practical questions. First, does adding speech improve detection over a text reference? Second, does training on MUStARD++ change performance relative to the original MUStARD corpus when the test targets are held fixed? Keeping the test population constant makes the dataset comparison easier to interpret than comparing scores from unrelated splits.""")
    p("""The final implementation freezes its pretrained encoders and trains small logistic-regression classifiers. This keeps the feature pipeline identical across training conditions and allows the same features to serve a local recording/upload application. We retain the original MUStARD notebook and weights as historical baselines; its gated-network and mismatch-ablation results are not presented as new MUStARD++ findings.""")
    p("""The contribution is a controlled dataset comparison and a local implementation connected to the MUStARD++ refit. We do not propose a new encoder or claim state-of-the-art performance. Published scores from different modalities or splits are not directly comparable to this matched shared-target protocol. We report observed differences and their uncertainty; final training fits are not accuracy estimates.""")
    sec('II. DATA AND RELATED WORK')
    p("""Castro et al. introduced MUStARD for multimodal sarcasm detection [1]. Ray et al. extended this research direction through a corpus with emotion, valence, arousal and sarcasm-type annotations [2]. We obtain MUStARD++ target annotations from the official repository [3] and audiovisual files from a separately verified mirror. The experiment uses dataset-provided target transcripts and extracted audio; it does not use video frames or dialogue context. Whisper is used for optional application transcription, not to generate the primary benchmark inputs. The benchmark therefore does not measure an end-to-end audio-to-ASR-to-sarcasm system.""")
    p("""The selected official MUStARD++ target rows contain 1,202 utterances, balanced at 601 sarcastic and 601 non-sarcastic. Relative to the original 690 IDs, 688 are shared, 514 are added and two are absent. Six shared transcripts differ after whitespace normalization. Shared test features use the MUStARD++ transcripts for both training conditions, so the test inputs remain identical. Original MUStARD training features preserve their original transcripts and labels.""")
    table('TABLE I<br/>MUStARD++ TARGET DISTRIBUTION',[
        ['Show','Targets','Sarcastic'],['Big Bang Theory','716','360'],['Friends','354','151'],['Silicon Valley','78','37'],['Golden Girls','40','39'],['Sarcasmoholics','14','14'],['Total','1,202','601']], [COL*.60,COL*.19,COL*.21])
    p("""Overall class balance hides strong show-level imbalance (Table I). In particular, nearly all Golden Girls targets and every Sarcasmoholics target are sarcastic. A high positive-class F1 on these small subsets can be misleading. The show name "Sarcasmoholics" follows the annotation label.""")
    p("""Audio is converted to 16 kHz mono PCM. Feature functions use at most the first 30 seconds and pad short waveforms as needed. Shared targets reuse original decoded audio for exact comparison. Official and mirror annotations are checked byte-for-byte; archive checksums and the annotation digest are recorded with the experiment. Raw clips, private caches and personal test recordings are excluded from Git.""")
    sec('III. REPRESENTATIONS AND TRAINING')
    sub('A. Frozen feature extraction')
    p("""RoBERTa-large [4] supplies a 1,024-dimensional mean-pooled text embedding. A separate CardiffNLP sentiment checkpoint supplies negative, neutral and positive probabilities [5]. WavLM-large [6] supplies a 2,048-dimensional speech representation by averaging middle-layer hidden states and concatenating temporal means and standard deviations. A SUPERB speech-emotion checkpoint [7] supplies four probabilities: neutral, happy, angry and sad. None of these encoders is fine-tuned.""")
    p("""We extract 58 acoustic and prosodic features, including pitch statistics, energy variation, pauses, spectral measures, MFCC means and standard deviations. Pitch estimation uses pYIN [8]. Speaking and articulation rates use transcript word counts. For that reason, the model named "Speech reference" is not fully independent of text.""")
    table('TABLE II<br/>FEATURE BLOCKS AND CLASSIFIER INPUTS',[
        ['Feature block','Dimensions'],['Target text embedding','1,024'],['Text sentiment probabilities','3'],['Speech embedding','2,048'],['Acoustic/prosodic statistics','58'],['Speech emotion probabilities','4'],['Valence mismatch','6']], [COL*.76,COL*.24])
    sub('B. Explicit valence features')
    p("""Text valence v_t is P(positive) minus P(negative). Voice valence v_s is P(happy) minus P(angry) minus P(sad). The six mismatch features are v_t, v_s, their difference, its absolute value, their product, and a polarity-flip indicator. The indicator is one when the product is negative and both absolute valences exceed 0.1. Ground-truth MUStARD++ emotion, valence and arousal annotations are never used as prediction inputs.""")
    p("""These features offer a compact way to represent opposing cues. The current MUStARD++ study does not ablate mismatch, so its results do not establish an independent benefit from these six features. The earlier MUStARD ablations remain available in the baseline archive.""")
    sub('C. Logistic-regression classifiers')
    p("""The text reference uses text embeddings and sentiment. Speech reference uses speech embeddings, prosody and speech emotion. Text + speech combines all six blocks. Each block is standardized, clipped to [-5, 5] and divided by the square root of its dimension before concatenation. Class weights are balanced. C is selected by three-fold log-loss validation from {0.1, 0.5, 2, 10, 50, 300, 1000}; optimization permits up to 3,000 iterations. Predictions use a fixed 0.5 threshold.""")
    p("""Evaluation fits separate classifiers for every split and training condition. The serving bundle is then refitted on all 1,202 MUStARD++ targets and contains Text + speech and Speech reference models. Its three-fold stratified C selection is a training step, not a reported held-out performance estimate. The final application uses no gated network.""")
    p("""The saved final MUStARD and MUStARD++ serving artifacts both selected C=2.0 for Text + speech and Speech reference. In the shared-target outer folds, baseline and size-matched training sets contain 552-553 examples, while full MUStARD++ sets contain 1,063-1,065 after group exclusions. The final MUStARD++ refit uses all 1,202 targets. These are training settings and sample counts, not training-set accuracy. Epoch-by-epoch loss histories were not saved for this logistic-regression study, so we report cross-validation outcomes instead of drawing unsupported learning curves.""")
    sec('IV. EVALUATION DESIGN')
    sub('A. Matched shared-target evaluation')
    p("""The primary test population is the same 688 shared targets, balanced at 344 per class. We use three repetitions of five-fold stratified grouped cross-validation with seeds 42, 43 and 44. Groups are formed jointly across both corpora by shared ID, identical normalized target text and identical decoded audio. Every held-out group is excluded from both training conditions. The audit finds 21 repeated normalized target rows within MUStARD++.""")
    p("""MUStARD training draws from its original 690 targets; MUStARD++ training draws from the expanded 1,202-target pool. Added targets can be used for training only when their group is absent from that fold\'s test set. A third, size-matched condition samples expanded-pool training rows to match the baseline training class counts. The same encoders, feature blocks, tuning grid and threshold are used throughout.""")
    p("""Inner C selection also uses three stratified grouped folds. Scaling is fitted on each outer training set and reused for its inner folds. Thus the outer test set remains excluded, but preprocessing is not fully nested inside C selection. This is a methodological limit of the reported comparison. Exact grouping also cannot remove every related dialogue, episode or speaker effect.""")
    p("""Each target receives one out-of-fold probability per repetition. We average its three probabilities and then compute sarcastic-class F1, macro-F1, accuracy, precision and recall. This pooled metric differs from the mean of fold-level F1 values. Fold summaries and predictions are saved separately to make that distinction explicit.""")
    p('For sarcasm as the positive class, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is 2PR/(P+R). Accuracy is (TP+TN)/N; macro F1 is the mean of sarcastic- and non-sarcastic-class F1. TP and TN are correct positive and negative predictions; FP and FN are their respective errors. Undefined class scores are set to zero.')
    p('Averaging three held-out probabilities forms a repeated-split estimate; it is not a measured score for the single final classifier refitted on all targets. The latter has no untouched test set after refitting. We keep both objects separate when interpreting performance.')
    sub('B. Uncertainty and unseen-show tests')
    p("""We estimate paired F1 differences with 2,000 bootstrap samples of target groups, using seed 42. Both training conditions are scored on the same resampled targets. Repeated-CV predictions are averaged before resampling; they are not treated as independent observations. The interval describes uncertainty within this development population and does not capture all domain or model-selection uncertainty.""")
    p("""A separate experiment leaves each of the five shows out in turn. Training excludes the held-out show, overlapping target IDs and duplicate groups. All 1,202 targets participate in these show-specific tests. The uneven class distributions make individual positive-class F1 scores difficult to compare directly across shows.""")
    sec('V. RESULTS AND INTERPRETATION')
    rows=[['Model','MUStARD','MUStARD++']]
    for mode in ['Text only','Speech reference','Text + speech']:
        rows.append([mode,f'{metric("MUStARD",mode):.3f}',f'{metric("MUStARD++",mode):.3f}'])
    table('TABLE III<br/>POOLED SHARED-TARGET SARCASTIC-CLASS F1 (688 TARGETS)',rows,[COL*.48,COL*.25,COL*.27])
    figure('f1_comparison.png','Fig. 1. Sarcastic-class F1 on the same 688 shared targets. Each score uses the mean of three out-of-fold probabilities. Size-matched training controls the number of training examples.',169)
    metric_rows=[['Training / input','Acc.','Prec.','Recall','F1','Macro F1']]
    for training,short in [('MUStARD','M'),('MUStARD++ size-matched','SM'),('MUStARD++','M++')]:
        for mode,label in [('Text only','Text'),('Speech reference','Speech'),('Text + speech','T + S')]:
            metric_rows.append([f'{short} / {label}',*[f'{metric(training,mode,key):.3f}' for key in ['accuracy','precision','recall','f1','macro_f1']]])
    table('TABLE IV<br/>FULL POOLED SHARED-TARGET METRICS',metric_rows,[COL*.31,*[COL*.138]*5])
    p("""In Table IV, M denotes MUStARD, M++ denotes full MUStARD++, and SM denotes size-matched MUStARD++ training. Precision, recall and F1 refer to the sarcastic class; macro F1 averages both classes. All rows use the same 688 test targets. A stronger text-only score in the size-matched condition than in the full condition also shows that dataset expansion does not improve every model consistently.""")
    figure('confusion_matrices.png','Fig. 2. Text + speech confusion matrices from averaged held-out predictions. Each reference class has 344 targets; counts are not final-refit training predictions.',141)
    p(f'Text + speech achieves F1 {final:.3f} with MUStARD++ training, compared with {metric("MUStARD++","Text only"):.3f} for text only and {metric("MUStARD++","Speech reference"):.3f} for speech reference (Table III). The multimodal result has accuracy {metric("MUStARD++","Text + speech","accuracy"):.3f}, macro-F1 {metric("MUStARD++","Text + speech","macro_f1"):.3f}, precision {metric("MUStARD++","Text + speech","precision"):.3f} and recall {metric("MUStARD++","Text + speech","recall"):.3f}. F1 should not be described as accuracy.')
    p(f'The multimodal dataset difference is {ci.delta_f1:+.3f} F1. Its 95% interval [{ci.ci_low:+.3f}, {ci.ci_high:+.3f}] includes zero, so the observed advantage does not conclusively demonstrate that MUStARD++ is better. Size-matched MUStARD++ training reaches F1 {metric("MUStARD++ size-matched","Text + speech"):.3f}, close to the original baseline. The full expansion\'s gain may therefore depend partly on having more training examples; these controls do not prove a causal explanation.')
    figure('paired_intervals.png','Fig. 3. Paired 95% group-bootstrap intervals for F1 differences (2,000 resamples). Every full-expansion interval crosses zero.',130)
    rows=[['Held-out show','MUStARD','MUStARD++']]
    for key,label in [('BBT','Big Bang Theory'),('FRIENDS','Friends'),('SV','Silicon Valley'),('GOLDENGIRLS','Golden Girls'),('SARCASMOHOLICS','Sarcasmoholics')]:
        vals=[held.loc[(held.show==key)&(held.training==t)&(held['mode']=='Text + speech'),'f1'].iloc[0] for t in ['MUStARD','MUStARD++']]
        rows.append([label,*[f'{v:.3f}' for v in vals]])
    table('TABLE V<br/>HELD-OUT-SHOW F1: TEXT + SPEECH',rows,[COL*.48,COL*.25,COL*.27])
    p("""Expanded training improves the observed Friends and Silicon Valley scores, while the Big Bang Theory score is slightly lower. The small, skewed Golden Girls and Sarcasmoholics subsets do not support broad generalization claims. These results show why one shared-target score cannot establish reliable behavior on all unseen recordings.""")
    sec('VI. SEPARATE SURFACE-EMOTION EXPERIMENT')
    emotion=json.loads((ROOT/'results/emotion/evaluation.json').read_text())
    p("""The original voice output comes from a frozen four-class speech-emotion encoder. It was not trained on MUStARD++ emotion labels. We later trained a separate nine-class head on the corpus's Explicit_Emotion annotations, which describe perceived surface emotion [3]. We do not infer a speaker's actual feelings from these labels. This experiment is separate from the binary sarcasm comparison above.""")
    p("""The new head uses frozen WavLM embeddings, the original four emotion probabilities and 55 acoustic/prosodic features. The three word-count-dependent rate features are excluded, making this head audio-only. A StandardScaler and logistic regression with fixed C=0.1 are fitted inside each leave-one-show-out split; optimization allows 2,000 iterations. Normalized duplicate test utterances are removed from training. There is no test-set hyperparameter search. The final candidate head is refitted on all 1,202 surface-emotion labels.""")
    counts=emotion['counts']
    p('Surface labels are imbalanced: ' + ', '.join(f'{label.lower()} {count}' for label,count in counts.items()) + '. This imbalance makes overall accuracy alone a poor description of coverage across emotions.')
    emotion_rows=[['Model','Accuracy','Macro F1']]
    for key,label in [('frozen_four_class','Frozen four-class'),('trained','MUStARD++ nine-class'),('majority','Train-fold majority')]:
        values=emotion['metrics'][key];emotion_rows.append([label,f"{values['accuracy']:.3f}",f"{values['macro_f1']:.3f}"])
    table('TABLE VI<br/>HELD-OUT-SHOW SURFACE-EMOTION RESULTS (1,202 TARGETS)',emotion_rows,[COL*.56,COL*.22,COL*.22])
    p("""The candidate reaches accuracy 0.310 and macro F1 0.146, versus 0.302 and 0.113 for the frozen detector. However, the training-fold majority predictor reaches accuracy 0.364. The old detector cannot output five of the nine labels, so this comparison reflects label coverage as well as classification quality. The candidate's pooled log loss is 2.720. Its confusion matrix gives zero recall for disgust (0/31) and fear (0/23), while only 1/48 frustration targets is correctly recognized. Thus expanding the label set has not produced useful recognition for several rare classes. These results do not justify calling it a reliable replacement.""")
    figure('emotion_confusion.png','Fig. 4. Nine-class surface-emotion confusion counts from leave-one-show-out predictions. Rows are source annotations, not labels inferred from the model itself.',227)
    p("""The app exposes the nine-class head as an experimental option and keeps the four-class detector as default. Switching this option changes the displayed voice scores only. The original emotion features and mismatch computation inside the evaluated sarcasm classifier remain unchanged. Replacing those features would require retraining and evaluating the sarcasm pipeline again.""")
    sec('VII. APPLICATION CHECKS AND LIMITATIONS')
    p("""The macOS application starts a localhost Python server when opened and stops it when its window closes or the app quits. A watchdog handles unexpected parent termination. The interface accepts a microphone recording or uploaded audio of 1-30 seconds, transcribes locally with Whisper-tiny.en [9] when needed, and reports a sarcasm score, transcript and voice-delivery estimate. Users can correct the transcript before analysis.""")
    p("""Frozen models and tokenizers are retained in memory during an app session, avoiding repeated checkpoint loading for each clip. The first request can still be slower because it initializes the encoders and audio-processing code. Closing the app releases these resources. Model scores are uncalibrated; a short positive-sounding phrase with frustrated delivery can still be classified incorrectly.""")
    p("""Default voice output has four supported classes and cannot label frustration. The experimental head adds frustration and four other surface labels, but neither detector is calibrated for live microphone speakers. A conflicting-cues flag marks strongly opposing text and voice polarities using thresholds of 0.5. It does not override the sarcasm score and has not been validated as another classifier.""")
    p("""The dataset contains scripted television speech rather than a representative sample of everyday conversations. Shared-show folds retain recurring speakers and recording conditions. ASR errors and sentiment/emotion domain mismatch can affect live results. No labeled live-microphone benchmark or accuracy improvement from the cue flag is claimed. Personal demonstration recordings were not used to select thresholds, train weights or compute research scores.""")
    p("""We also verified the application through generated-file uploads. Two macOS speech voices supplied eight sentences each, with MP3 and M4A versions of each voice's first WAV: 20 files in total. With transcripts left blank, both emotion modes completed all 40 upload requests. Comparing ASR output to the known words gives 9 word edits across 118 reference words (7.63% WER); one difference is the equivalent numeral spelling "nine" versus "9". Compressed variants repeat source sentences, so these are not 20 independent language examples. The figures describe processing and transcription, not sarcasm or emotion accuracy.""")
    p('Eighteen annotated dataset demos completed 36 inference requests in two emotion modes, with 14/18 sarcasm-label matches. These are training examples, so the count is a functional observation, not held-out accuracy. A synthetic noise check changed the top emotion on 10/16 pairs for the original detector and 4/16 for the candidate. Neither result supplies human emotion ground truth.')
    p('Microphone checks confirmed non-silent capture but also exposed empty or repetitive ASR output. Trimming and bounded gain did not reliably resolve that failure and were not adopted. Private captures are excluded from training and public research results. A speaker-separated adaptation and calibration workflow exists, but its labelled-recording manifest is empty: no calibrated microphone model or new-speaker accuracy gain is reported.')
    sub('Reproduction details')
    p('The serving manifest and committed CSVs identify the trained artifact and evaluation evidence. The notebook defaults to review mode and displays these saved results; full reproduction requires its explicit retraining switch. Dataset preparation and evaluation are implemented in experiment/prepare.py and experiment/compare.py. The original baseline is preserved separately.')
    p('Exact encoder identifiers are roberta-large, cardiffnlp/twitter-roberta-base-sentiment-latest, microsoft/wavlm-large and superb/wav2vec2-base-superb-er. Application ASR uses openai/whisper-tiny.en. Recorded experiment versions are PyTorch 2.15.0, Transformers 5.23.0, NumPy 2.5.5, pandas 3.1.1, librosa 1.0.1, SoundFile 0.14.0 and scikit-learn 1.9.1 [10]. Classifier and feature signatures are checked; upstream checkpoint revisions were not pinned, which limits exact future reproduction. Pretraining overlap with the television material was not audited.')
    sec('VIII. CONCLUSION')
    p("""TrueTone connects a reproducible MUStARD++ feature and training pipeline to a local speech application. Text and speech together outperform the text reference on the matched test population. MUStARD++ yields a modest observed gain over the original MUStARD training baseline, but its uncertainty interval includes zero. We adopt MUStARD++ as the final project dataset and preserve the baseline evidence. Evaluation on diverse, independently labeled recordings remains necessary before making stronger claims about real-world sarcasm detection.""")
    sec('REFERENCES')
    refs=[
        '[1] S. Castro et al., "Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)," in Proc. ACL, 2019, pp. 4619-4629. https://aclanthology.org/P19-1455/',
        '[2] A. Ray, S. Mishra, A. Nunna, and P. Bhattacharyya, "A Multimodal Corpus for Emotion Recognition in Sarcasm," in Proc. LREC, 2022, pp. 6992-7003. https://aclanthology.org/2022.lrec-1.756/',
        '[3] CFILT, "MUStARD_Plus_Plus," official dataset annotations. https://github.com/cfiltnlp/MUStARD_Plus_Plus (accessed Nov. 26, 2026).',
        '[4] Y. Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.',
        '[5] J. Camacho-Collados et al., "TweetNLP: Cutting-Edge Natural Language Processing for Social Media," in Proc. EMNLP: System Demonstrations, 2022. Sentiment checkpoint: cardiffnlp/twitter-roberta-base-sentiment-latest.',
        '[6] S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE J. Sel. Topics Signal Process., vol. 16, no. 6, pp. 1505-1518, 2022.',
        '[7] S.-w. Yang et al., "SUPERB: Speech Processing Universal PERformance Benchmark," in Proc. Interspeech, 2021, pp. 1194-1198. Checkpoint: superb/wav2vec2-base-superb-er.',
        '[8] M. Mauch and S. Dixon, "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions," in Proc. ICASSP, 2014, pp. 659-663.',
        '[9] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in Proc. ICML, PMLR, vol. 202, pp. 28492-28518, 2023. https://proceedings.mlr.press/v202/radford23a.html',
        '[10] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011. https://www.jmlr.org/papers/v12/pedregosa11a.html'
    ]
    for text in refs:p(text,'ref')
    first_top=H-170
    def frames(top):
        return [Frame(M,42,COL,top-42,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),Frame(M+COL+GAP,42,COL,top-42,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
    OUT.parent.mkdir(exist_ok=True)
    doc=BaseDocTemplate(str(OUT),pagesize=letter,title='TrueTone: Multimodal Sarcasm Detection on MUStARD++',author='Manas Sharma; Raghav Vats; Vaibhav Singh; Sumit Bhati',subject='PID-02: MUStARD++ final training and paired evaluation')
    doc.addPageTemplates([PageTemplate(id='first',frames=frames(first_top),onPage=header,autoNextPageTemplate='later'),PageTemplate(id='later',frames=frames(H-42),onPage=header)])
    doc.build(story)
    print(OUT)


if __name__=='__main__':build()
