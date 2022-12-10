### Курсы
Теоретические широкие:
1. Курс на русском от Самсунга (https://github.com/Samsung-IT-Academy/stepik-dl-nlp)
2. Текстовый курс Е. Войта (https://lena-voita.github.io/nlp_course.html)
3. Курс ШАДа (https://github.com/yandexdataschool/nlp_course)
4. Стэнфордский курс CS224n (http://web.stanford.edu/class/cs224n/)
5. Стэнфордский курс CS224u (http://web.stanford.edu/class/cs224u/). 
6. Курс на русском от Хуавея (https://ods.ai/tracks/nlp-course)
Узкоспециализированные:
1. Курс от Hugging Face по библиотеке transformers (https://huggingface.co/course/) 
Прикладные:
1. https://github.com/yandexdataschool/nlp_course
2. https://github.com/DanAnastasyev/DeepNLP-Course

### Основные статьи:
- Word2Vec: Mikolov et al., Efficient Estimation of Word Representations in Vector Space https://arxiv.org/pdf/1301.3781.pdf
- FastText: Bojanowski et al., Enriching Word Vectors with Subword Information https://arxiv.org/pdf/1607.04606.pdf
- Attention: Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
- Transformers: Vaswani et al., Attention Is All You Need https://arxiv.org/abs/1706.03762
- BERT: Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding https://arxiv.org/abs/1810.0480


### NER

CRF - https://medium.com/data-science-in-your-pocket/named-entity-recognition-ner-using-conditional-random-fields-in-nlp-3660df22e95c
CRF https://medium.com/@phylypo/nlp-text-segmentation-using-conditional-random-fields-e8ff1d2b6060
External knowledge https://aclanthology.org/P18-2039.pdf
Few Shot https://habr.com/ru/company/sberbank/blog/649609/
NER example https://github.com/Erlemar/Erlemar.github.io/blob/master/Notebooks/ner_sberloga.ipynb
Использование QA для few-shot NER: QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition (https://arxiv.org/abs/2203.01543), реализация - https://github.com/dayyass/QaNER

О том, как делать NER модели:
- важно иметь достаточно разметки для начала работы. Можно ориентироваться на 1 000 документов
- хороший безйлайн - регулярки. Если что-то можно цеплять регулярками, лучше воспользоваться ими. Регулярки хорошо подходят для единообразных сущностей, тогда достаточно не так много примеров.
- Если тексты сущностей разнообразны, необходима модель. Для BiLSTM + CharCNN + CRF нужно имного данных, больше чем для БЕРТа.
- Очень хорошо в NER работает CRF - сглаживает острые углы (например, не дает сущностям прерываться)
- В качестве body лучше использовать BiLSTM
- При обучении обычно сначала делают файн-тьюн головы при замороженном эмбеддере, а потом дообучают все вместе (на каждую итерацию 5-10 эпох)
- Если для сущностей есть иерархия (например, сущность Тип организации, подтип ЮЛ, ИП, кооператив), можно использовать two-level NER: У модели будет две головы. Первая --- стандартная, как для обычного NER, на типы (O, B/I_Entity, ...). Вторая --- только для подтипов. Вторая состоит из одного слоя для multihot-подтипов и одного для single-hot. Они отличаются только лоссом, поэтому разберем на примере single-hot. Пусть есть сущность A с подтипами a1, a2, a3, C с подтипами c1, c2 и B без подтипов. Размер слоя будет 2 (количество сущностей, имеющих подтипы) x 3 (максимальное число подтипов среди таких сущностей). Во время обучения, когда встречается метка A:a2, считается лосс для A в первой голове и лосс по a2 в первой строке тензора во второй голове (софтмакс по трем значениям). Во время предсказания вторая голова активируется, только если первая выбрала соответствующий тип.
- Очень хорошо работает Negative Sampling  при большом дисбалансе классов: если текст сущностей занимает 1-5% от всего текста, необходимо повысить баланс классов для модели NER, например, разбив текст на куски и обучивклассификатор наличия сущностей в тексте. Таким образом, на этапе обуения мы обучаем NER только на сегментах, предсказанных классификатором

### Суммаризация:
1. Цикл статей на Хабре https://habr.com/ru/post/596481/, https://habr.com/ru/post/595517/, https://habr.com/ru/post/595597/
Extractive BERT https://github.com/dmmiller612/bert-extractive-summarizer

### NLI и zero-shot классификации: 
русскоязычные модели https://habr.com/ru/post/582620

### Speech and Language Processing: 
https://web.stanford.edu/~jurafsky/slp3/

### OCR:

- Tesseract https://github.com/tesseract-ocr/tesseract
- EasyOCR https://github.com/JaidedAI/EasyOCR
- Ocropy https://github.com/ocropus/ocropy https://github.com/jze/ocropus-model_cyrillic
- Asprise http://asprise.com/royalty-free-library/python-ocr-api-overview.html

- FineReader / ABBYY Cloud OCR SDK https://cloud.ocrsdk.com
- Yandex Cloud Vision API https://cloud.yandex.ru/docs/vision/operations/ocr/text-detection
- Google Cloud Vision API https://cloud.google.com/vision/docs/ocr

- Microsoft Office Document Imaging https://t.me/natural_language_processing/15431
- Cuneiform https://launchpad.net/cuneiform-linux

### Чат-боты:
- Разработка навыков Алисы в Python https://youtu.be/VlkCJ26Gd60

### Spacy
Версия 2 - https://www.youtube.com/watch?v=sqDHBH9IjRU, 
Spacy NER -  https://arxiv.org/pdf/1603.01360.pdf 
Модели spacy ru: https://docs.google.com/spreadsheets/u/0/d/1laE3m3KmNlNk6HN3dC63IOs6aB8qyGnqusc1wX2SGdQ/edit

### Topic Models
Add prior https://aclanthology.org/E12-1021.pdf
GuidedLDA https://github.com/vi3k6i5/GuidedLDA

### Dialog Systems
In conversations https://arxiv.org/pdf/2002.02353.pdf
Conversational data Tasks https://arxiv.org/pdf/2103.03125.pdf
Generative dialog system https://habr.com/ru/company/sberdevices/blog/589969/
GPT https://colab.research.google.com/drive/1sD_hQJOi3CrHn7Ba-XuKkHRToxDRRSof?usp=sharing&hl=ru
Select response https://arxiv.org/pdf/2009.12539.pdf
Dialog segmentation http://www.iro.umontreal.ca/~lisa/pointeurs/Article_NLPRS.pdf
DailyDialog https://aclanthology.org/I17-1099.pdf

### Search Engines
Retrieval Transformer https://habr.com/ru/post/648705/

### Sentence Similarity
SBERT https://www.sbert.net/docs/quickstart.html#comparing-sentence-similarities
in Information Retrieval / Semantic Search scenarios: First, you use an efficient Bi-Encoder to retrieve e.g. the top-100 most similar sentences for a query. Then, you use a Cross-Encoder to re-rank these 100 hits by computing the score for every (query, hit) combination.
Перед использованием энкодеров - дообучение через sentence-transformers методом TSDAE модельки rubert-tiny2 на своих 1.5 млн предложениях.


### Aspect Extraction
Aspects - simple CNN https://www.sentic.net/aspect-extraction-for-opinion-mining.pdf
Classification of sentiment of aspect https://aclanthology.org/D16-1058.pdf
https://github.com/ruidan/Unsupervised-Aspect-Extraction
https://aclanthology.org/2021.emnlp-main.528/
https://github.com/declare-lab/awesome-sentiment-analysis

### Sentiment Analysis
Emotion Recognition in Conversations https://github.com/declare-lab/conv-emotion#cosmic-commonsense-knowledge-for-emotion-identification-in-conversations
https://github.com/declare-lab/awesome-sentiment-analysis


### Text Segmentaion

Supervised
- https://arxiv.org/pdf/1803.09337.pdf предсказываем конец сегмента
- https://arxiv.org/pdf/2012.03619.pdf
- https://medium.com/doctrine/structuring-legal-documents-with-deep-learning-4ad9b03fb19
- http://ceur-ws.org/Vol-710/paper23.pdf
Topic Modeling or graph-based
Based on word embeddings similarity
 https://arxiv.org/pdf/1503.05543v1.pdf

### Text Augmentation

Mixup-transformer https://aclanthology.org/2020.coling-main.305/
Synonims, translation-based https://github.com/dsfsi/textaugment
Weighting word embeddings https://math.mit.edu/research/highschool/primes/materials/2020/Zhao-Lialin-Rumshisky.pdf
Mixup on sentence classification https://arxiv.org/pdf/1905.08941.pdf


### transformers
Примеры использования - https://huggingface.co/transformers/v3.2.0/notebooks.html
transformers - 6 encoder + 6 decoder (https://arxiv.org/pdf/1706.03762.pdf)
BERT - only encoders + segment type id + trained on MLM and NSP (CLS+SEP). base - 12 encoder blocks, 12 heads, 768 emb size, 512 seq len (https://arxiv.org/pdf/1810.04805.pdf)
ALBERTa - shared  weights in all 12 encoders
DistilBERT - 6 encoders instead of 12 encoders with teaching from BERT
RoBERTa - optimized hyperparameters and no NSP
ELECTRA - pretrained on the Replaced Token Detection (RTD) task (there is small LM model which replaces [MASK] token with prediction and this input is given to ELECTRA which predicts correctness of of the predicted masked tokens) 
LaBSE - multilingual sentence embedder. The goal of training is to get similar CLS embeddings of pairs of sentences on different languages with the same meaning while using in-batch negative sampling with margin loss (https://arxiv.org/pdf/2007.01852.pdf)
Difference between BERT architectures:
https://towardsdatascience.com/everything-you-need-to-know-about-albert-roberta-and-distilbert-11a74334b2da
https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/#electra
Tokenization (BPE, WordPiece) https://towardsdatascience.com/a-comprehensive-guide-to-subword-tokenisers-4bbd3bad9a7c
