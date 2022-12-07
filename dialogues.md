DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances
URL: https://arxiv.org/pdf/2012.01775.pdf

Дообучение БЕРТ на двух задачах:

1) Masked utterance regression - предсказываем эмбеддинг маскированной реплики
2) Distributed Utterance Order Ranking - перемешиваем реплики и предсказываем порядковый номер каждой реплики (для получения итоговой последовательности сортируем реплики по выходному скору - чем больше значение, тем позже в ряду реплика)


