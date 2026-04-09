import 'axios'
import axios from 'axios'
import { socketDraftReady, socketPredictDone, type Prediction, type PredictionRequest } from '../types'
import { socketService } from './socketManager'
import { store } from '../redux/store'
import { addDraft, setPrediction } from '../redux/features/prediction/prediction'

const instance = axios.create({
    baseURL: 'http://localhost:8000'
})

export const getPrediction = async (predictionData: PredictionRequest) => {
    const response = await instance.post<Prediction>('/predict/', predictionData)
    const prediction = response.data

    // Render base response immediately; drafts will arrive over websocket.
    store.dispatch(setPrediction(prediction))

    const socketUrl = `ws://localhost:8000/ws/predict/${prediction.request_id}/`
    socketService.connect(socketUrl, (event) => {
        let payload: any = null
        try {
            payload = JSON.parse(event.data)
        } catch {
            return
        }

        if (payload?.event === socketDraftReady && payload?.draft) {
            store.dispatch(addDraft(payload.draft))
        }

        if (payload?.event === socketPredictDone) {
            socketService.disconnect()
        }
    })

    return prediction
}