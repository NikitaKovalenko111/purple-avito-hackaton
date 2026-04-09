import { useEffect, useState, type JSX } from "react";
import type React from "react";
import { useDispatch, useSelector } from "react-redux";
import { Link } from "react-router-dom";
import { type RootState } from "../../redux/store";
import { categoriesRefs, type Draft, type PredictionSession } from "../../types";
import { getPrediction } from "../../api/api";
import { setPrediction } from "../../redux/features/prediction/prediction";
import { loadPredictionSessions, subscribePredictionSessionsUpdated } from "../../utils/predictionSessions";

type PropsType = {}

const MainPage: React.FC<PropsType> = (): JSX.Element => {
    const dispatch = useDispatch()
    const prediction = useSelector((state: RootState) => state.prediction.prediction)
    const categories = useSelector((state: RootState) => state.prediction.prediction?.detectedMcIds)
    const shouldSplit = useSelector((state: RootState) => state.prediction.prediction?.shouldSplit)
    const drafts = useSelector((state: RootState) => state.prediction.prediction?.drafts)

    const [description, setDescription] = useState<string>("")
    const [sourceMcId, setSourceMcId] = useState<number>(101)
    const [isLoading, setIsLoading] = useState<boolean>(false)
    const [recentSessions, setRecentSessions] = useState<PredictionSession[]>([])

    useEffect(() => {
        const refreshSessions = () => {
            setRecentSessions(loadPredictionSessions())
        }

        refreshSessions()
        const unsubscribe = subscribePredictionSessionsUpdated(refreshSessions)
        return unsubscribe
    }, [])

    const handleReset = () => {
        setDescription("")
        setSourceMcId(101)
    }

    const handleAnalyze = async () => {
        if (!description.trim() || isLoading) {
            return
        }

        try {
            setIsLoading(true)
            await getPrediction({
                sourceMcId,
                sourceMcTitle: categoriesRefs[sourceMcId],
                description,
            })
        } catch (error) {
            console.error("Prediction request failed", error)
        } finally {
            setIsLoading(false)
        }
    }

    const openRecentSession = (session: PredictionSession) => {
        setSourceMcId(session.request.sourceMcId)
        setDescription(session.request.description)
        dispatch(setPrediction(session.response))
    }

    const formatSessionTime = (iso: string): string => {
        const date = new Date(iso)
        return Number.isNaN(date.getTime())
            ? ""
            : date.toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" })
    }

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
                    <form
                        className="ad-form"
                        onSubmit={(event) => {
                            event.preventDefault()
                            handleAnalyze()
                        }}
                    >
                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="ad-source">Исходная категория</label>
                        <select value={sourceMcId} onChange={(el) => {
                            setSourceMcId(parseInt(el.target.value))
                        }} className="ad-form__control" id="ad-source" name="source">
                            {Object.entries(categoriesRefs).map(([id, title]) => (
                                <option key={id} value={id}>
                                    {id} - {title}
                                </option>
                            ))}
                        </select>
                        </div>

                        <div className="ad-form__field">
                        <label className="ad-form__label" htmlFor="listing-description">Описание объявления</label>
                        <textarea
                            value={description}
                            onChange={(el) => {
                                setDescription(el.target.value)
                            }}
                            className="ad-form__control ad-form__control--textarea"
                            id="listing-description"
                            name="description"
                            rows={9}
                        />
                        </div>

                        {/*<div className="ad-form__field ad-form__field--row">
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-prob">Порог разделения</label>
                            <input className="ad-form__control" id="ad-prob" name="threshold" type="number" min="0" max="1" step="0.01" value="0.30" />
                        </div>
                        <div className="ad-form__field-item">
                            <label className="ad-form__label" htmlFor="ad-topk">Максимальное количество микрокатегорий</label>
                            <input className="ad-form__control" id="ad-topk" name="topk" type="number" min="1" max="15" step="1" value="8" />
                        </div>
                        </div>*/}

                        <div className="ad-form__actions">
                        <button
                            onClick={() => {
                                handleAnalyze()
                            }}
                            className="ad-form__button ad-form__button--primary"
                            type="button"
                            disabled={isLoading}
                        >
                            {isLoading ? "Анализ..." : "Анализ"}
                        </button>
                        <button onClick={handleReset} className="ad-form__button ad-form__button--ghost" type="button">Сбросить</button>
                        </div>
                    </form>
                    </section>

                    <section className="panel panel--results" aria-labelledby="panel-results-title">
                    <h2 className="panel__title" id="panel-results-title">Ответ модели</h2>
                    {
                        prediction != null ? (
                            <>
                            <article className="result-card">
                                <h3 className="result-card__title">Найденные микрокатегории</h3>
                                <ul className="result-card__chips">
                                {categories?.map((el) => (
                                    <li className="result-card__chip" key={el}>
                                        {el} {categoriesRefs[el] ?? "Неизвестная категория"}
                                    </li>
                                ))}
                                </ul>
                            </article>

                            <article className="result-card">
                                <h3 className="result-card__title">Следует разделить?</h3>
                                <p className={`result-card__decision ${shouldSplit ? 'result-card__decision--yes' : 'result-card__decision--no'}`}>Нужно разделить: {shouldSplit ? 'Да' : 'Нет'}</p>
                                {/*<p className="result-card__text">3 микрокатегории превысили порог и отличны от исходной категории</p>*/}
                            </article>

                            <article className="result-card">
                                <h3 className="result-card__title">Черновики</h3>
                                <ul className="result-card__drafts">
                                {
                                    drafts?.map((el: Draft) => {
                                        return (
                                            <li className="result-card__draft" key={`${el.mcId}-${el.mcTitle}`}>
                                                <h4 className="result-card__draft-title">{el.mcId} {el.mcTitle}</h4>
                                                <p className="result-card__draft-text">{el.text}</p>
                                            </li>
                                        )
                                    })
                                }
                                </ul>
                            </article>
                            </>
                        ) : (
                            <article className="result-card">
                                <h3 className="result-card__title">Ответа модели пока нет</h3>
                            </article>
                        )
                    }
                    </section>

                    <aside className="panel panel--side" aria-labelledby="panel-side-title">
                    <h2 className="panel__title" id="panel-side-title">Инструменты</h2>

                    <article className="side-card">
                        <h3 className="side-card__title">Недавние сессии</h3>
                        <ul className="side-card__list">
                        {recentSessions.length === 0 ? (
                            <li className="side-card__item">Сессий пока нет</li>
                        ) : (
                            recentSessions.slice(0, 8).map((session) => (
                                <li className="side-card__item" key={session.id}>
                                    <div className="side-card__session-head">
                                        <span>
                                            {session.request.sourceMcId} - {categoriesRefs[session.request.sourceMcId] ?? session.request.sourceMcTitle}
                                        </span>
                                        <span className="side-card__session-time"> {formatSessionTime(session.createdAt)}</span>
                                    </div>
                                    <div className="side-card__session-meta">
                                        Разделение: {session.response.shouldSplit ? "да" : "нет"}, черновиков: {session.response.drafts.length}
                                    </div>
                                    <button
                                        className="side-card__button side-card__button--small"
                                        type="button"
                                        onClick={() => openRecentSession(session)}
                                    >
                                        Открыть
                                    </button>
                                </li>
                            ))
                        )}
                        </ul>
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