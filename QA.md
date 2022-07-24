## Задача:
Поиск ответа на вопрос в тексте статьи или поиск документа с ответом на вопрос среди множества документов. 
Выделяют open domain QA и knowledge based QA. Вторые в качестве источника имеют под собой Википедию или граф знаний для поиска ответа

## Примеры имплементации:
Лонгформер https://github.com/patil-suraj/Notebooks/blob/master/longformer_qa_training.ipynb

## Datasets:
- Список англоязычных датасетов: http://nlpprogress.com/english/question_answering.html
- Самый популярный корпус для обучения QA моделей - SQuAD. Его особенность - ответ на вопрос представляет собой спан исходного текста
- Датасеты для чат-ботов https://analyticsindiamag.com/10-question-answering-datasets-to-build-robust-chatbot-systems/
- FAQ dialog datasets https://paperswithcode.com/dataset/doqa

Русскозычные
- SberQUAD

## Методы

Обзор методов и датасетов: https://arxiv.org/abs/2206.15030

1. Самое простое решение - BERT-QA&. Данные подаются в формате [CLS] QUESTION [SEP] context [SEP] -> (start_pos, end_pos)
2. Дообучить языковую модель так, чтобы вектор вопроса и вектор ответа были близки (что-то типа LABSE). Ещё если есть большая обучающая выборка, можно просто двумя несвязанными энкодерами получить представления вопроса и ответа и обучать модель минимизировать косинусное расстояние между этими представлениями. Из-за того, что энкодеры не связаны, непохожесть ответа на вопрос никак не помешает -  contrastive learning
3. Если ответ нужно искать в нескольких документах, то нужный документ можно искать по tf-idf/bm25, а затем поиск ответа в найденной статье (odqa/kbqa). Статьи можно разбивать на блоки (разного размера и с перекрытием), обычно это улучшает качество ответа. Каждый блок эмбеддится бертом (в один вектор), и потом ищется ответ среди более длинных блоков, включавших в себя блоки, оказавшиеся ближайшими соседями к эмбеддингу запроса.
Матчить по близости можно через faiss например (cosine similarity in most nearest cluster)
4. Генерация ответа с помощью BART или T5

Тулзы:

- Можно попробовать предобученные сетки для такого, например: https://demo.allennlp.org/open-information-extraction
- Про разные подходы можно прочитать здесь: https://paperswithcode.com/task/open-information-extraction
- Перед использованием можно ещё пройтись вот этой сеткой: https://github.com/google-research/language/tree/master/language/decontext

### Few Shot



## Применение:
Онлайн(автоматический)-консультант на сайте - а-ля выдача ответа пользователю поиском по faq
https://in.springboard.com/blog/nlp-project-automated-question-answering-model/


### FAQ retrieval
О задаче https://arxiv.org/abs/1905.02851
Multilingual dataset and approach https://www.researchgate.net/publication/354889556_MFAQ_a_Multilingual_FAQ_Dataset
Multi domain https://www.researchgate.net/publication/341148105_DoQA_-_Accessing_Domain-Specific_FAQs_via_Conversational_QA

Существующие разработки:
- https://developers.sap.com/tutorials/conversational-ai-faq-chatbot-beginner.html
- https://research.aimultiple.com/faq-chatbot/
- https://www.intercom.com/automated-answers?on_pageview_event=automated_answers_footer

Русские системы:
- https://bothelp.io/ru/templates/faq-ru
- https://www.helpdeski.ru/tags/chat_boty_dlja_podderzhki/



