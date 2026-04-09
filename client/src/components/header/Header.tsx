import type { JSX } from "react"
import type React from "react"
import { useState } from "react"
import { useSelector } from "react-redux"
import { Link } from "react-router-dom"
import type { RootState } from "../../redux/store"
import { upsertPredictionSession } from "../../utils/predictionSessions"

type PropsType = {}

const Header: React.FC<PropsType> = (): JSX.Element => {
    const prediction = useSelector((state: RootState) => state.prediction.prediction)
    const lastRequest = useSelector((state: RootState) => state.prediction.lastRequest)
    const [saveState, setSaveState] = useState<"idle" | "saved">("idle")

    const saveSessionHandler = () => {
      if (!prediction || !lastRequest) {
        return
      }

      upsertPredictionSession(lastRequest, prediction)
      setSaveState("saved")
      window.setTimeout(() => setSaveState("idle"), 1200)
    }

    return (
        <header className="page-header">
            <div className="page-header__container container">
              <Link className="page-header__brand" to="/">Category Split Assistant</Link>
              <nav className="page-header__nav" aria-label="Основная навигация">
                <Link className="page-header__link page-header__link--active" to="/">Рабочая панель</Link>
                <Link className="page-header__link" to="/project">Проект</Link>
                <Link className="page-header__link" to="/dataset">Датасет</Link>
                <Link className="page-header__link" to="/metrics">Метрики</Link>
              </nav>
              <button
                className="page-header__button"
                type="button"
                onClick={saveSessionHandler}
                disabled={!prediction || !lastRequest}
              >
                {saveState === "saved" ? "Сохранено" : "Сохранить сессию"}
              </button>
            </div>
        </header>
    )
}

export default Header