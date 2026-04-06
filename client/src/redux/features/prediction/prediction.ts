import { createSlice } from '@reduxjs/toolkit'
import type { PayloadAction } from '@reduxjs/toolkit'
import type { Draft, Prediction } from '../../../types'

export interface PredictionState {
  prediction: Prediction | null
}

const initialState: PredictionState = {
  prediction: null
}

export const predictionSlice = createSlice({
  name: 'prediction',
  initialState,
  reducers: {
    setPrediction: (state, action: PayloadAction<Prediction>) => {
      state.prediction = action.payload
    },

    addDraft: (state, action: PayloadAction<Draft>) => {
      state.prediction?.drafts.push(action.payload)
    }
  },
})

export const { setPrediction, addDraft } = predictionSlice.actions

export default predictionSlice.reducer