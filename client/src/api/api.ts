import 'axios'
import axios from 'axios'
import { socketDraftReady, socketPredictDone, type Prediction, type PredictionRequest } from '../types'
import { socketService } from './socketManager'
import { store } from '../redux/store'
import { addDraft } from '../redux/features/prediction/prediction'

const instance = axios.create({
    baseURL: 'http://localhost:8000'
})

export const getPrediction = async (predictionData: PredictionRequest) => {
    const res = await instance.post<Prediction>('/predict', predictionData).then(data => {
        const socketUrl = `ws://localhost:8000/ws/predict/${data.data.request_id}/`

        socketService.connect(socketUrl, (event) => {
            if (event.data.event = socketDraftReady) {
                store.dispatch(addDraft(event.data.draft))
            }

            if (event.data.event = socketPredictDone) {
                socketService.disconnect()
            }
        })

        return data.data
    })

    return res
}