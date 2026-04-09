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
                    <h1 className="info__title" id="info-title-project">Текущий pipeline принятия решения</h1>
                </header>

                <div className="info__cards">
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 1: Поиск кандидатов</h2>
                    <p className="info-card__text">TF-IDF + char n-grams формируют detected-кандидатов и первичный скор по микрокатегориям.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 2: Transformer + split head</h2>
                    <p className="info-card__text">Transformer оценивает вероятности категорий и отдельно бинарное shouldSplit через split-head.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Этап 3: Реранжирование и JSON-ответ</h2>
                    <p className="info-card__text">Пороги class/score и shouldSplit определяют splitCategories; при включенном режиме split может приравниваться к detected.</p>
                    </article>
                </div>

                <section className="timeline" aria-labelledby="timeline-title">
                    <h2 className="timeline__title" id="timeline-title">Схема работы</h2>
                    <ol className="timeline__list">
                    <li className="timeline__item">Входное объявление попадает в рабочую панель.</li>
                    <li className="timeline__item">Сервис возвращает detectedMcIds без sourceMcId и с пороговой фильтрацией.</li>
                    <li className="timeline__item">shouldSplit вычисляется по split-head и split_threshold.</li>
                    <li className="timeline__item">Если shouldSplit=true, генерируются splitCategories и затем черновики по WebSocket.</li>
                    </ol>
                </section>
                </div>
            </section>
        </main>
    )
}

export default ProjectPage