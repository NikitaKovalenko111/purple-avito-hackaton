import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const MetricsPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-metrics">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Оценка</p>
                    <h1 className="info__title" id="info-title-metrics">Метрики качества и интерпретация</h1>
                </header>

                <div className="info__cards">
                    <article className="info-card">
                    <h2 className="info-card__title">Micro-precision (Precision)</h2>
                    <p className="info-card__text">Сколько предсказанных категорий разделения оказались правильными.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Micro-recall (Recall)</h2>
                    <p className="info-card__text">Какую долю истинных категорий разделения находит модель.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Micro F1</h2>
                    <p className="info-card__text">Основная метрика, балансирующая точность и полноту.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">shouldSplit accuracy</h2>
                    <p className="info-card__text">Точность бинарного решения о необходимости разделения.</p>
                    </article>
                </div>

                <section className="kpi" aria-labelledby="kpi-title">
                    <h2 className="kpi__title" id="kpi-title">Рекомендуемые целевые диапазоны</h2>
                    <ul className="kpi__list">
                    <li className="kpi__item">Micro F1: 0.70+ для надежного пилота</li>
                    <li className="kpi__item">shouldSplit accuracy: 0.80+ для стабильного рабочего процесса</li>
                    <li className="kpi__item">Ограничение по точности: избегать шумных лишних черновиков</li>
                    </ul>
                </section>
                </div>
            </section>
        </main>
    )
}

export default MetricsPage