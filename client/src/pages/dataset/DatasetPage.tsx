import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const DatasetPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-dataset">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Dataset Documentation</p>
                    <h1 className="info__title" id="info-title-dataset">Synthetic benchmark details</h1>
                </header>

                <div className="stat-grid">
                    <article className="stat-card">
                    <p className="stat-card__label">Rows</p>
                    <p className="stat-card__value">3,000</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Microcategories</p>
                    <p className="stat-card__value">11</p>
                    </article>
                    <article className="stat-card">
                    <p className="stat-card__label">Split=True share</p>
                    <p className="stat-card__value">37.1%</p>
                    </article>
                </div>

                <section className="table-block" aria-labelledby="table-title">
                    <h2 className="table-block__title" id="table-title">Core columns</h2>
                    <table className="table-block__table">
                    <thead>
                        <tr>
                        <th>Field</th>
                        <th>Description</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                        <td>description</td>
                        <td>Listing text used for detection and split decision</td>
                        </tr>
                        <tr>
                        <td>targetDetectedMcIds</td>
                        <td>All microcategories present in text</td>
                        </tr>
                        <tr>
                        <td>targetSplitMcIds</td>
                        <td>Microcategories for additional draft creation</td>
                        </tr>
                        <tr>
                        <td>shouldSplit</td>
                        <td>Binary decision flag for draft creation</td>
                        </tr>
                        <tr>
                        <td>split</td>
                        <td>train, val, test partition</td>
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