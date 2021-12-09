# RS-Solver

## О программе

RS-Solver создан специально для решения задач контрольной работы по НИС. Он умеет решать пять типов задач: расчет индекса Хирша, задачи на комплексную оценку, метод иерархий, парную оценку и непосредственную оценку. Программа тестировалось на двух подборках задач и всё решила верно.

Несмотря на то что задачи решает машина, а она не ошибается в расчетах, настоятельно рекомендуется проверять все решения программы. Эта программа по большой части нацелена на совместное решение человека и компьютера, при этом каждый должен проверять друг-друга. Именно такой подход создатель данной программы считает наиболее эффективным и результативным.

## Установка

1. Скачать архив с файлами программы (кнопка **Code** -- > **Download ZIP**):

![installation](https://user-images.githubusercontent.com/83603595/145408973-c6b19f5d-c54c-4593-908e-48612e5fede3.png)

2. Распаковать архив в любую удобную папку

## Как пользоваться

Данная программа работает следующим образом: ей нужно "скормить" Excel файл, который называется "research_seminar_problems.xlsx". Во избежание различных ошибок критически важно сфокусироваться на предобработке данных в Excel, а уже потом перейти к использованию программы.

### Подготовка Excel документа

Данный файл включает в себя 5 таблиц, соответствующие типу задания.

![installation1](https://user-images.githubusercontent.com/83603595/145411953-5bba100c-3035-4a08-bf01-fa3a60f0b80e.png)

Ни в коем случае не следует менять их местами, либо вставлять какие-то другие таблицы, не соответствующие типу задания! Также во ВСЕХ таблицах проверьте, чтобы у вас не было пробелов после цифр и чтобы все дроби были написаны через точку (например, 0.5, а не 0,5), иначе программа может выдать ошибку.

#### Индекс Хирша

Вы увидите такую таблицу:

![installation2](https://user-images.githubusercontent.com/83603595/145412595-46024aec-90b1-4510-8ffe-44268b79ff51.png)

Всё, что вам нужно сделать - скопировать и вставить значения в таблицу. НО НЕ МЕНЯЙТЕ НАЗВАНИЯ КОЛОНОК ("публикация" и "цитирование"), иначе программа работать не будет!

#### Непосредственная оценка

Здесь насчет названий колонок можно не беспокоиться, просто скопируйте и вставьте вашу таблицу с непосредственными оценками заместо старой таблицы. 

#### Парная оценка

Вставьте все ваши матрицы оценок в соответствии с шаблоном. ВНИМАНИЕ! Матрицы должны идти через одну строчку, иначе программа выдаст неправильные значения. Например, как здесь:

![installation3](https://user-images.githubusercontent.com/83603595/145416097-962d2e1f-2ec8-4e1f-8a33-398f83cd4314.png)

В данном примере матриц всего четыре, поскольку там 4 эксперта, но их может быть больше, а может быть меньше. Главное, чтобы они шли ЧЕРЕЗ 1 строку.

#### Анализ иерархий

Аналогично с этой таблицой. Неважно сколько критериев и сколько объектов (колонок A, B, C и т.д.), главное, чтобы все эти матрицы были разделены ОДНОЙ строкой.

#### Комплексная оценка

Категориальные переменные, которые представлены в виде текста необходимо закодировать прямо в Excel, т.е. перевести их в числовые показатели! Также необходимо вручную убрать предметы, которые неэффективны (например, как это было с 3-м учебником в примере презентации). За остальное не стоит беспокоиться:)

### Туториал по программе

#### Запуск

Запустить программу можно через jupyter-notebook, если вы знаете как это делать (файл `Program for Research Seminar.ipynb`). Лично для меня это наиболее удобный вариант. Но можно запустить и как python-файл.

Зажмите сочетание клавиш `Win+R`, у вас откроется окно "выполнить". Туда введите "cmd".

![installation4](https://user-images.githubusercontent.com/83603595/145417635-ece31496-5440-4247-b882-f541e58e7722.png)

Нажмите `Enter`. У вас откроется терминал.

![installation5](https://user-images.githubusercontent.com/83603595/145417988-480219e9-e46b-418f-bd01-31719f185d78.png)

Откройте вашу папку с файлами, она должна выглядеть так:

![installation7](https://user-images.githubusercontent.com/83603595/145420663-c1260b19-26ea-4d18-8fcc-ecdaa479950a.png)

Скопируйте путь данной папки (включая саму папку). Для этого нажмите правой кнопкой мыши -- > Свойства (Properties) и скопируйте путь.

![installation6](https://user-images.githubusercontent.com/83603595/145421622-905d7efa-88a6-4481-aeb3-0e54500f3ba0.png)

Зайдите снова в терминал. И пропишите `cd ваш-путь-к-файлу-без-кавычек`.

![installation8](https://user-images.githubusercontent.com/83603595/145421962-b7c486a8-ff7a-4861-b68c-f038422769c7.png)

Отлично, вы в нужной директории. Теперь введите `python RS_Solver.py`, чтобы запустить программу.

#### Что нужно делать дальше?

Если вы всё сделали правильно, у вас появится сообщение:

> Введите путь к файлу, либо название файла с расширением (если он в той же папке, что и скрипт): 

Если ваш файл Excel находится в той же папке, что и .py файл и вы не меняли у него название, то просто скопируйте и вставьте `research_seminar_problems.xlsx`. Если нет - укажите название файла и путь к нему. Нажмите `Enter`.

При условии, что нигде не возникло ошибок, программа сразу же рассчитает вам два задания (Хирш и непосредственная оценка).

![installation9](https://user-images.githubusercontent.com/83603595/145424396-2915e28f-b01b-4368-a4ca-85cff9512e36.png)


Для следующего задания "парная оценка" вам потребуется ввести доверительный интервал в виде доли (т.е. если в условии задачи он равен 99%, то вам надо ввести 0.99) и количество экспертов (сколько всего матриц с оценками экспертов). В примере файла их 4, но в контрольной может быть другое число.

## Возможные ошибки и как их исправить

## Благодарности
