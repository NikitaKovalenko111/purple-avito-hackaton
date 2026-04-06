import type { JSX } from "react";
import type React from "react";
import { Link } from "react-router-dom";

type PropsType = {}

const MainPage: React.FC<PropsType> = (): JSX.Element => {
    return (
        <main className="page__main page__main--workspace">
            <section className="workspace" aria-labelledby="workspace-title">
                <div className="workspace__container container">
                <header className="workspace__head">
                    <div className="workspace__intro">
                    <p className="workspace__eyebrow">Рабочая панель</p>
                    <h1 className="workspace__title" id="workspace-title">Панель обработки объявлений</h1>
                    <p className="workspace__lead">
                        Введите текст, выберите исходную категорию и запустите анализ.
                    </p>
                    </div>
                    <div className="workspace__status" aria-label="Статус модели">
                    <span className="workspace__status-dot"></span>
                    Модель: TF-IDF (2-3 grams) + Transformer Softmax
                    </div>
                </header>

                <div className="workspace__grid">
                    <section className="panel panel--form" aria-labelledby="panel-form-title">
                    <h2 className="panel__title" id="panel-form-title">Введите объявление</h2>
                    <form className="ad-form" action="#" method="post">
                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="ad-source">Исходная категория</label>
                        <select className="ad-form__control" id="ad-source" name="source">
                            <option>101 - Ремонт под ключ</option>
                            <option>102 - Сантехника</option>
                            <option>103 - Электрика</option>
                            <option>104 - Натяжные потолки</option>
                            <option>105 - Укладка плитки</option>
                        </select>
                        </div>

                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="listing-description">Описание объявления</label>
                        <textarea className="ad-form__control ad-form__control--textarea" id="listing-description" name="description" rows={9}>Делаем ремонт под ключ, отдельно выполняем электрику, сантехнику и укладку плитки. Работаем по договору, выезд и смета бесплатно.</textarea>
                        </div>

                        <div className="ad-form__field ad-form__field--row">
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-prob">Порог разделения</label>
                            <input className="ad-form__control" id="ad-prob" name="threshold" type="number" min="0" max="1" step="0.01" value="0.30" />
                        </div>
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-topk">Максимальное количество микрокатегорий</label>
                            <input className="ad-form__control" id="ad-topk" name="topk" type="number" min="1" max="15" step="1" value="8" />
                        </div>
                        </div>

                        <div className="ad-form__actions">
                        <button className="ad-form__button ad-form__button--primary" type="button">Анализ</button>
                        <button className="ad-form__button ad-form__button--ghost" type="button">Сбросить</button>
                        </div>
                    </form>
                    </section>

                    <section className="panel panel--results" aria-labelledby="panel-results-title">
                    <h2 className="panel__title" id="panel-results-title">Ответ модели</h2>

                    <article className="result-card">
                        <h3 className="result-card__title">Найденные микрокатегории</h3>
                        <ul className="result-card__chips">
                        <li className="result-card__chip">102 Сантехника</li>
                        <li className="result-card__chip">103 Электрика</li>
                        <li className="result-card__chip">105 Укладка плитки</li>
                        </ul>
                    </article>

                    <article className="result-card">
                        <h3 className="result-card__title">Следует разделить?</h3>
                        <p className="result-card__decision result-card__decision--yes">Нужно разделить: да</p>
                        <p className="result-card__text">3 микрокатегории превысили порог и отличны от исходной категории</p>
                    </article>

                    <article className="result-card">
                        <h3 className="result-card__title">Пример черновика</h3>
                        <ul className="result-card__drafts">
                        <li className="result-card__draft">
                            <h4 className="result-card__draft-title">102 Сантехника</h4>
                            <p className="result-card__draft-text">Выполняем сантехнические работы отдельно: разводка труб, установка сантехники, замена смесителей.</p>
                        </li>
                        <li className="result-card__draft">
                            <h4 className="result-card__draft-title">103 Электрика</h4>
                            <p className="result-card__draft-text">Отдельно делаем электромонтаж: замена проводки, перенос розеток и установка освещения.</p>
                        </li>
                        </ul>
                    </article>
                    </section>

                    <aside className="panel panel--side" aria-labelledby="panel-side-title">
                    <h2 className="panel__title" id="panel-side-title">Инструменты</h2>

                    <article className="side-card">
                        <h3 className="side-card__title">Недавние сессии</h3>
                        <ul className="side-card__list">
                        <li className="side-card__item">Объявление 5001 - разделение: да</li>
                        <li className="side-card__item">Объявление 5002 - разделение: нет</li>
                        <li className="side-card__item">Объявление 5003 - разделение: да</li>
                        </ul>
                    </article>

                    <article className="side-card">
                        <h3 className="side-card__title">Экспорт настроек</h3>
                        <div className="side-card__buttons">
                        <button className="side-card__button" type="button">Скачать JSON</button>
                        <button className="side-card__button" type="button">Скачать</button>
                        </div>
                    </article>

                    <article className="side-card">
                        <h3 className="side-card__title">Страницы проекта</h3>
                        <nav className="side-card__links" aria-label="Дополнительная навигация">
                        <Link className="side-card__link" to="/project">Показ пайплайна</Link>
                        <Link className="side-card__link" to="/dataset">Структура датасета</Link>
                        <Link className="side-card__link" to="/metrics">Метрики</Link>
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