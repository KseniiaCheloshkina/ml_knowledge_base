Задача:
Поиск ответа на вопрос в тексте статьи или поиск документа с ответом на вопрос среди множества документов. 
Выделяют open domain QA и knowledge based QA. Вторые в качестве источника имеют под собой Википедию или граф знаний для поиска ответа

Datasets:
http://nlpprogress.com/english/question_answering.html
Особенность датасета SQUAD - the answer to every question is a segment of text (a span) from the corresponding reading passage. 
https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
FAQ dialog datasets https://paperswithcode.com/dataset/doqa

Survey:
https://arxiv.org/abs/2206.15030

Применение:
Онлайн(автоматический)-консультант на сайте - а-ля выдача ответа пользователю поиском по faq
https://in.springboard.com/blog/nlp-project-automated-question-answering-model/



FAQ retrieval https://arxiv.org/abs/1905.02851
Multilingual dataset and approach https://www.researchgate.net/publication/354889556_MFAQ_a_Multilingual_FAQ_Dataset
Multi domain https://www.researchgate.net/publication/341148105_DoQA_-_Accessing_Domain-Specific_FAQs_via_Conversational_QA


https://developers.sap.com/tutorials/conversational-ai-faq-chatbot-beginner.html
https://research.aimultiple.com/faq-chatbot/
https://www.intercom.com/automated-answers?on_pageview_event=automated_answers_footer

Русские системы
https://bothelp.io/ru/templates/faq-ru
https://www.helpdeski.ru/tags/chat_boty_dlja_podderzhki/


Вариант1
дообучить языковую модель так, чтобы  вектор диалога и вектор названия категории были близки (что-то типа LABSE)
NLU (category sep dialog)
Ещё если есть большая обучающая выборка, можно просто двумя несвязанными энкодерами получить представления вопроса и ответа и обучать модель минимизировать косинусное расстояние между этими представлениями. Из-за того, что энкодеры не связаны, непохожесть ответа на вопрос никак не помешает

Вариант2
В начале всё равно будет поиск tf-idf/bm25, а затем поиск ответа в найденной статье (odqa/kbqa). Статьи можно разбивать на блоки, обычно это улучшает качество ответа.
разбивал на блоки разного размера и с перекрытием, эмбеддил каждый блок бертом (в один вектор), и потом искал ответ среди более длинных блоков, включавших в себя блоки, оказавшиеся ближайшими соседями к эмбеддингу запроса.
да, если документы большие, то разбивать с оверлепом и матчить топ кандидатов, через faiss например (cosine similarity in most bearest cluster)
BART
Можно взять мою T5, я её обучал отвечать на вопросы на данных SberQUAD. 

Датасет всегда должен выглядеть так, как будет выглядеть при применении модели. 
В нашем случае, наверное, так: контекст + вопрос + ответ

Можно попробовать предобученные сетки для такого, например: https://demo.allennlp.org/open-information-extraction
Про разные подходы можно прочитать здесь:
https://paperswithcode.com/task/open-information-extraction

Перед использованием можно ещё пройтись вот этой сеткой: https://github.com/google-research/language/tree/master/language/decontext

