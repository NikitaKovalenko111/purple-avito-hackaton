import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const DatasetPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-dataset">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Документация датасета</p>
                    <h1 className="info__title" id="info-title-dataset">Текущий датасет: markup_balanced</h1>
                </header>

                <div className="stat-grid">
                    <article className="stat-card">
                    <p className="stat-card__label">Строк</p>
                    <p className="stat-card__value">2,480</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Микрокатегорий</p>
                    <p className="stat-card__value">11 (целевые 10)</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Доля split=True</p>
                    <p className="stat-card__value">19.0%</p>
                    </article>
                </div>

                <div className="stat-grid">
                    <article className="stat-card">
                    <p className="stat-card__label">Train / Val / Test</p>
                    <p className="stat-card__value">1736 / 372 / 372</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Источник (sourceMcId)</p>
                    <p className="stat-card__value">101</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Семантика split</p>
                    <p className="stat-card__value">split==detected при shouldSplit=True</p>
                    </article>
                </div>

                <section className="table-block" aria-labelledby="table-title">
                    <h2 className="table-block__title" id="table-title">Основные колонки</h2>
                    <table className="table-block__table">
                    <thead>
                        <tr>
                        <th>Поле</th>
                        <th>Описание</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                        <td>description</td>
                        <td>Текст объявления для детекции и решения о разделении</td>
                        </tr>
                        <tr>
                        <td>targetDetectedMcIds</td>
                        <td>Все микрокатегории, присутствующие в тексте</td>
                        </tr>
                        <tr>
                        <td>targetSplitMcIds</td>
                        <td>Для markup_balanced совпадает с targetDetectedMcIds, если shouldSplit=True</td>
                        </tr>
                        <tr>
                        <td>shouldSplit</td>
                        <td>Бинарный флаг: создавать дополнительные черновики или нет</td>
                        </tr>
                        <tr>
                        <td>split</td>
                        <td>Разделение на train, val, test</td>
                        </tr>
                    </tbody>
                    </table>
                </section>
                </div>
            </section>
            </main>
    )
}

export default DatasetPage