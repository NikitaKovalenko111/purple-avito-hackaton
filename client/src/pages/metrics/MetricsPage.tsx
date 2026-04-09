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
                    <h1 className="info__title" id="info-title-metrics">Текущие метрики (последний запуск)</h1>
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
                    <h2 className="kpi__title" id="kpi-title">Фактические значения</h2>
                    <ul className="kpi__list">
                    <li className="kpi__item">Val split: precision 0.816, recall 0.427, F1 0.561, shouldSplit accuracy 0.511</li>
                    <li className="kpi__item">Val detect: precision 0.519, recall 0.934, F1 0.667</li>
                    <li className="kpi__item">Test split: precision 0.780, recall 0.450, F1 0.571, shouldSplit accuracy 0.495</li>
                    <li className="kpi__item">Test detect: precision 0.473, recall 0.927, F1 0.626</li>
                    </ul>
                </section>
                </div>
            </section>
        </main>
    )
}

export default MetricsPage