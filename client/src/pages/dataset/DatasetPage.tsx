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
                    <h1 className="info__title" id="info-title-dataset">Детали синтетического бенчмарка</h1>
                </header>

                <div className="stat-grid">
                    <article className="stat-card">
                    <p className="stat-card__label">Строк</p>
                    <p className="stat-card__value">3,000</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Микрокатегорий</p>
                    <p className="stat-card__value">11</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Доля split=True</p>
                    <p className="stat-card__value">37.1%</p>
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
                        <td>Микрокатегории для создания дополнительных черновиков</td>
                        </tr>
                        <tr>
                        <td>shouldSplit</td>
                        <td>Бинарный флаг необходимости создания черновиков</td>
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