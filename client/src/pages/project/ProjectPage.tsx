import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const ProjectPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-project">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Архитектура проекта</p>
                    <h1 className="info__title" id="info-title-project">Как система принимает решение:</h1>
                </header>

                <div className="info__cards">
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 1: Поиск кандидатов</h2>
                    <p className="info-card__text">TF-IDF биграммы и триграммы ищут категории по ключевым словам.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 2: Вероятностная модель</h2>
                    <p className="info-card__text">Текстовые эмбеддинги трансформера и обучаемые эмбеддинги микрокатегорий формируют softmax-вероятности.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 3: Выдача черновиков</h2>
                    <p className="info-card__text">Категории выше порога становятся кандидатами для черновиков. Результат экспортируется в ожидаемой JSON-схеме.</p>
                    </article>
                </div>

                <section className="timeline" aria-labelledby="timeline-title">
                    <h2 className="timeline__title" id="timeline-title">Схема работы</h2>
                    <ol className="timeline__list">
                    <li className="timeline__item">Входное объявление попадает в рабочую панель.</li>
                    <li className="timeline__item">Ретривер сужает список кандидатов.</li>
                    <li className="timeline__item">Softmax-модель оценивает вероятность разделения.</li>
                    <li className="timeline__item">Черновики формируются и экспортируются.</li>
                    </ol>
                </section>
                </div>
            </section>
        </main>
    )
}

export default ProjectPage