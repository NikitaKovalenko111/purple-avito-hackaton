import type { JSX } from "react"
import type React from "react"
import { Link } from "react-router-dom"

type PropsType = {}

const Header: React.FC<PropsType> = (): JSX.Element => {
    return (
        <header className="page-header">
            <div className="page-header__container container">
              <Link className="page-header__brand" to="/">Service Split Assistant</Link>
              <nav className="page-header__nav" aria-label="Primary">
                <Link className="page-header__link page-header__link--active" to="/">Рабочая панель</Link>
                <Link className="page-header__link" to="/project">Проект</Link>
                <Link className="page-header__link" to="/dataset">Датасет</Link>
                <Link className="page-header__link" to="/metrics">Метрики</Link>
              </nav>
              <button className="page-header__button" type="button">Сохранить сессию</button>
            </div>
        </header>
    )
}

export default Header