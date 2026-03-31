import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const MetricsPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-metrics">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Evaluation</p>
                    <h1 className="info__title" id="info-title-metrics">Quality metrics and interpretation</h1>
                </header>

                <div className="info__cards">
                    <article className="info-card">
                    <h2 className="info-card__title">Precision micro</h2>
                    <p className="info-card__text">Out of predicted split categories, how many are correct.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Recall micro</h2>
                    <p className="info-card__text">Out of true split categories, how many the model finds.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">F1 micro</h2>
                    <p className="info-card__text">Main score that balances precision and recall.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">shouldSplit accuracy</h2>
                    <p className="info-card__text">Correctness of the binary split decision.</p>
                    </article>
                </div>

                <section className="kpi" aria-labelledby="kpi-title">
                    <h2 className="kpi__title" id="kpi-title">Suggested target bands</h2>
                    <ul className="kpi__list">
                    <li className="kpi__item">F1 micro: 0.70+ for reliable pilot</li>
                    <li className="kpi__item">shouldSplit accuracy: 0.80+ for stable operator flow</li>
                    <li className="kpi__item">Precision guardrail: avoid noisy extra drafts</li>
                    </ul>
                </section>
                </div>
            </section>
        </main>
    )
}

export default MetricsPage