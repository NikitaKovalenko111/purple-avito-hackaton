export const socketDraftReady = "draft_ready"
export const socketPredictDone = "predict_done"

export const categoriesRefs: Record<number, string> = {
    101: "Ремонт квартир и домов под ключ",
    102: "Сантехника",
    103: "Электрика",
    104: "Натяжные потолки",
    105: "Укладка плитки",
    106: "Поклейка обоев",
    107: "Малярные работы",
    108: "Штукатурные работы",
    109: "Напольные покрытия",
    110: "Гипсокартон",
    111: "Демонтажные работы",
}

export interface PredictionRequest {
    sourceMcId: number
    sourceMcTitle: string
    description: string
}

export interface Prediction {
    request_id: string
    detectedMcIds: Array<number>
    shouldSplit: boolean
    drafts: Array<Draft>
}

export interface Draft {
    mcId: number
    mcTitle: string
    text: string
}