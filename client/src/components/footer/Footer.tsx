import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const Footer: React.FC<PropsType> = (): JSX.Element => {
    return (
        <footer className="page-footer">
            <div className="page-footer__container container">
                <p className="page-footer__text">Интерфейс для операций разбиения объявлений. Семантическая HTML-разметка, нейминг BEM, SCSS-модули.</p>
            </div>
        </footer>
    )
}

export default Footer