import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const MainPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--workspace">
            <section className="workspace" aria-labelledby="workspace-title">
                <div className="workspace__container container">
                <header className="workspace__head">
                    <div className="workspace__intro">
                    <p className="workspace__eyebrow">Operator panel</p>
                    <h1 className="workspace__title" id="workspace-title">Ad processing workspace</h1>
                    <p className="workspace__lead">
                        Paste listing text, pick the source microcategory, run analysis, and review predicted split drafts in one flow.
                    </p>
                    </div>
                    <div className="workspace__status" aria-label="Model status">
                    <span className="workspace__status-dot"></span>
                    Model profile: TF-IDF (2-3 grams) + Transformer Softmax
                    </div>
                </header>

                <div className="workspace__grid">
                    <section className="panel panel--form" aria-labelledby="panel-form-title">
                    <h2 className="panel__title" id="panel-form-title">Input listing</h2>
                    <form className="ad-form" action="#" method="post">
                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="ad-source">Source microcategory</label>
                        <select className="ad-form__control" id="ad-source" name="source">
                            <option>101 - Turnkey Renovation</option>
                            <option>102 - Plumbing</option>
                            <option>103 - Electrical</option>
                            <option>104 - Stretch Ceilings</option>
                            <option>105 - Tile Laying</option>
                        </select>
                        </div>

                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="listing-description">Listing description</label>
                        <textarea className="ad-form__control ad-form__control--textarea" id="listing-description" name="description" rows={9}>Делаем ремонт под ключ, отдельно выполняем электрику, сантехнику и укладку плитки. Работаем по договору, выезд и смета бесплатно.</textarea>
                        </div>

                        <div className="ad-form__field ad-form__field--row">
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-prob">Split threshold</label>
                            <input className="ad-form__control" id="ad-prob" name="threshold" type="number" min="0" max="1" step="0.01" value="0.30" />
                        </div>
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-topk">Candidate top K</label>
                            <input className="ad-form__control" id="ad-topk" name="topk" type="number" min="1" max="15" step="1" value="8" />
                        </div>
                        </div>

                        <div className="ad-form__actions">
                        <button className="ad-form__button ad-form__button--primary" type="button">Analyze</button>
                        <button className="ad-form__button ad-form__button--ghost" type="button">Reset</button>
                        </div>
                    </form>
                    </section>

                    <section className="panel panel--results" aria-labelledby="panel-results-title">
                    <h2 className="panel__title" id="panel-results-title">Model output</h2>

                    <article className="result-card">
                        <h3 className="result-card__title">Detected microcategories</h3>
                        <ul className="result-card__chips">
                        <li className="result-card__chip">102 Plumbing</li>
                        <li className="result-card__chip">103 Electrical</li>
                        <li className="result-card__chip">105 Tile Laying</li>
                        </ul>
                    </article>

                    <article className="result-card">
                        <h3 className="result-card__title">Split decision</h3>
                        <p className="result-card__decision result-card__decision--yes">shouldSplit: true</p>
                        <p className="result-card__text">3 draft microcategories exceed threshold and differ from source category.</p>
                    </article>

                    <article className="result-card">
                        <h3 className="result-card__title">Draft preview</h3>
                        <ul className="result-card__drafts">
                        <li className="result-card__draft">
                            <h4 className="result-card__draft-title">102 Plumbing</h4>
                            <p className="result-card__draft-text">Выполняем сантехнические работы отдельно: разводка труб, установка сантехники, замена смесителей.</p>
                        </li>
                        <li className="result-card__draft">
                            <h4 className="result-card__draft-title">103 Electrical</h4>
                            <p className="result-card__draft-text">Отдельно делаем электромонтаж: замена проводки, перенос розеток и установка освещения.</p>
                        </li>
                        </ul>
                    </article>
                    </section>

                    <aside className="panel panel--side" aria-labelledby="panel-side-title">
                    <h2 className="panel__title" id="panel-side-title">Quick tools</h2>

                    <article className="side-card">
                        <h3 className="side-card__title">Recent sessions</h3>
                        <ul className="side-card__list">
                        <li className="side-card__item">Item 5001 - split true</li>
                        <li className="side-card__item">Item 5002 - split false</li>
                        <li className="side-card__item">Item 5003 - split true</li>
                        </ul>
                    </article>

                    <article className="side-card">
                        <h3 className="side-card__title">Export options</h3>
                        <div className="side-card__buttons">
                        <button className="side-card__button" type="button">Copy JSON</button>
                        <button className="side-card__button" type="button">Download</button>
                        </div>
                    </article>

                    <article className="side-card">
                        <h3 className="side-card__title">Project pages</h3>
                        <nav className="side-card__links" aria-label="Secondary">
                        <a className="side-card__link" href="project.html">Pipeline overview</a>
                        <a className="side-card__link" href="dataset.html">Dataset structure</a>
                        <a className="side-card__link" href="metrics.html">Quality metrics</a>
                        </nav>
                    </article>
                    </aside>
                </div>
                </div>
            </section>
        </main>
    )
}

export default MainPage