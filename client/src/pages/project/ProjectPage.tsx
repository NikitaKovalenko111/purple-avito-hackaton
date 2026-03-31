import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const ProjectPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--info">
            <section className="info" aria-labelledby="info-title-project">
                <div className="info__container container">
                <header className="info__head">
                    <p className="info__eyebrow">Project Architecture</p>
                    <h1 className="info__title" id="info-title-project">How the system makes split decisions</h1>
                </header>

                <div className="info__cards">
                    <article className="info-card">
                    <h2 className="info-card__title">Stage 1: Retrieval</h2>
                    <p className="info-card__text">TF-IDF bi-grams and tri-grams detect microcategory candidates from key phrases and listing text.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Stage 2: Probability model</h2>
                    <p className="info-card__text">Transformer text embeddings and trainable microcategory embeddings form softmax probabilities.</p>
                    </article>
                    <article className="info-card">
                    <h2 className="info-card__title">Stage 3: Draft output</h2>
                    <p className="info-card__text">Categories above threshold become draft candidates. Results are exported in the expected JSON schema.</p>
                    </article>
                </div>

                <section className="timeline" aria-labelledby="timeline-title">
                    <h2 className="timeline__title" id="timeline-title">Operational flow</h2>
                    <ol className="timeline__list">
                    <li className="timeline__item">Input listing enters workspace panel.</li>
                    <li className="timeline__item">Retriever narrows candidates.</li>
                    <li className="timeline__item">Softmax model scores split probability.</li>
                    <li className="timeline__item">Drafts are prepared and exported.</li>
                    </ol>
                </section>
                </div>
            </section>
        </main>
    )
}

export default ProjectPage