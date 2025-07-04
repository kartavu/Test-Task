# Test-Task

# Генератор саммари на русском языке 

Мини-приложение на Streamlit для генерации краткого содержания текста на русском языке с использованием LLM-моделей

## Запуск 

Установление зависимостей 

```bash
pip install -r requirements.txt
```

Запуск приложения
```bash
streamlit run test_task.py
```

## Используемые модели 

В процессе работы были протестированы: 
1. facebook/bart-large-cnn
- Поддерживает только английский язык
- Использовал перевод RU - EN - RU, но результат неточный

2. ai-forever/FRED-T5-large
- Модель просто переформулировала оригинальные предложения 

В итоговой версии использовал:
3. IlyaGusev/rut5_base_sum_gazeta
- дает сжатое и логичное саммари
- иногда делает обобщения, но остаётся по смыслу

## Фильтрация

После получения саммари происходит:

1. ai-forever/sbert_large_nlu_ru
С его помощью я удаляю дублирующие по смыслу фразы 

2. проверка на фактическое соответсвие n-граммы 
т.е выбераю только те предложения, которые имеют фразы, встречающиеся в первоначальном тексте 

## Содержимое проекта
test_task.py — основной код Streamlit-приложения
requirements.txt — зависимости
test task/ — папка, в которой храняться файлы для саммари 
summary_results.txt - файл, в котором храниться результат саммари 

## Возможности
- Ввод текста вручную или через файл .txt, .pdf, .docx
- Разбиение длинных текстов на части
- Генерация саммари с учётом ограничений модели
- Семантическая фильтрация дубликатов
- Сохранение результата в summary_results.txt

## Пример работы

### Исходный текст №1

> В последние десятилетия информационные технологии стремительно развиваются. Особенно значимыми становятся решения в области искусственного интеллекта, машинного обучения и анализа больших данных. Компании, работающие в этих отраслях, активно внедряют технологии автоматизации процессов, повышая эффективность и снижая затраты. Одной из наиболее заметных тенденций является развитие генеративных моделей, таких как GPT, которые позволяют создавать осмысленный текст, имитирующий человеческую речь.  
> Кроме того, активно развиваются направления в области компьютерного зрения, что находит применение в медицине, автомобильной промышленности и безопасности. Например, системы распознавания лиц и объектов уже применяются в реальной жизни.  
> В то же время, такие технологии вызывают серьёзные вопросы в области этики и защиты персональных данных. Государственные органы и международные организации начинают разрабатывать нормативные акты, регулирующие использование ИИ.  
> В перспективе предполагается, что ИИ будет использоваться в образовании, правосудии и других социальных сферах, где важно соблюдение этических норм и прозрачности алгоритмов. Однако для этого необходимо решить множество технических и социальных задач, в том числе обеспечить интерпретируемость моделей, достоверность данных и обучение специалистов.  
> Таким образом, развитие ИИ — это не только технический прогресс, но и вызов для общества, требующий комплексного подхода и междисциплинарного взаимодействия.

### Сгенерированное саммари

> В последние десятилетия информационные технологии стремительно развиваются. Это не только технический прогресс, но и вызов для общества, требующего комплексного подхода и междисциплинарного взаимодействия.  
> Однако для этого необходимо решить множество технических и социальных задач, в том числе обеспечить интерпретируемость моделей, достоверность данных и прозрачность алгоритмов.  
> Эксперты считают, что развитие ИИ — это вызов, который может вызвать проблемы в области этики и безопасности персональных данных.

---

### Исходный текст №2

> Каждый день в библиотеку MyBook поступают новинки. Это книги и аудиокниги всех жанров: классика и современная литература, переиздания и новые произведения популярных авторов. Наш каталог — это ветвистое дерево.  
> Детективы, любовные романы, фэнтези, фантастика, научно-популярные и бизнес-книги — самые популярные жанры.  
> Каждый раздел содержит в себе несколько категорий, которые распределяют произведения по темам и уточняют поиск нужных книг.  
> В разделе современной прозы вы встретите Несбё и Акунина, Барнса и Токареву, Пелевина и Сорокина — весь спектр писателей, которые создают литературу сегодня.  
> Шедевры Достоевского, Толстого, Бунина и Куприна — в разделе классической литературы. Многие из классических романов и рассказов можно читать бесплатно.  
> Если у вас есть вопросы к самому себе, если вы хотите измениться или наладить отношения с близкими — обратитесь к книгам по психологии. Тысячи читателей ежедневно находят в них информацию, которая помогает улучшить качество жизни.  
> О том, как привести дела в порядок, все успевать, запоминать важное, презентовать и продавать, планировать и считать, рассказывают книги лучших российских издательств деловой литературы: «Манн, Иванов и Фербер», «Альпина Паблишер», «Эксмо», «Олимп-Бизнес», «Питер». Эти книги помогают улучшить личную эффективность и работу всего коллектива.  
> Что почитать, чтобы развлечься? Заходите в раздел детективов, любовных романов, фантастики и фэнтези. Там вы найдете новинки и классику любимых жанров.  
> В MyBook каждый сможет выбрать книгу по душе, даже ребенок. Сказки, повести, обучающие пособия и поучительные рассказы подойдут для чтения в кругу семьи. Вы можете читать и слушать книги онлайн на сайте или в мобильном приложении даже без интернета.

### Сгенерированное саммари

> Каждый день в библиотеку MyBook поступают новинки: классика и современная литература, переиздания и новые произведения популярных авторов.  
> Если у вас есть вопросы к самому себе, обратитесь к книгам по психологии, о том, как улучшить качество жизни и создать отношения с близкими, советуют читать книги лучших издательств деловой литературы: «Манн, Иванов и Фербер», «Олимп-Бизнес», «Питер».  
> В MyBook каждый сможет выбрать книгу по душе, даже ребенок.  
> В разделе детективов, любовных романов, фантастики и фэнтези вы найдете классику любимых жанров, которые подойдут для чтения в кругу семьи.  
> Что почитать, чтобы развлечься, заходите в раздел MyBook и слушайте книги онлайн на сайте или в мобильном приложении.

## Пример интерфейса 
<img src="image/1.png" width="500"/>

