import type { JSX } from "react"
import type React from "react"

type PropsType = {}

const Header: React.FC<PropsType> = (): JSX.Element => {
    return (
        <header className="page-header">
            <div className="page-header__container container">
              <a className="page-header__brand" href="index.html">Service Split Assistant</a>
              <nav className="page-header__nav" aria-label="Primary">
                <a className="page-header__link page-header__link--active" href="index.html">Workspace</a>
                <a className="page-header__link" href="project.html">Project</a>
                <a className="page-header__link" href="dataset.html">Dataset</a>
                <a className="page-header__link" href="metrics.html">Metrics</a>
              </nav>
              <button className="page-header__button" type="button">Save Session</button>
            </div>
        </header>
    )
}

export default Header