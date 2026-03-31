import type { JSX } from "react";
import type React from "react";

type PropsType = {}

const Footer: React.FC<PropsType> = (): JSX.Element => {
    return (
        <footer className="page-footer">
            <div className="page-footer__container container">
                <p className="page-footer__text">Workspace UI for ad split operations. Semantic HTML, BEM naming, SCSS modules.</p>
            </div>
        </footer>
    )
}

export default Footer