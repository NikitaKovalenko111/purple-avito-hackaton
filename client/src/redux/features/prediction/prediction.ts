import { createSlice } from '@reduxjs/toolkit'
import type { PayloadAction } from '@reduxjs/toolkit'
import type { Draft, Prediction, PredictionRequest } from '../../../types'

export interface PredictionState {
  prediction: Prediction | null
  lastRequest: PredictionRequest | null
}

const initialState: PredictionState = {
  prediction: null,
  lastRequest: null,
}

export const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    setPrediction: (state, action: PayloadAction<Prediction>) => {
      state.prediction = action.payload
    },

    setLastRequest: (state, action: PayloadAction<PredictionRequest>) => {
      state.lastRequest = action.payload
    },

    addDraft: (state, action: PayloadAction<Draft>) => {
      state.prediction?.drafts.push(action.payload)
    }
  },
})

export const { setPrediction, setLastRequest, addDraft } = predictionSlice.actions

export default predictionSlice.reducer