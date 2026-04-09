import { configureStore } from '@reduxjs/toolkit'
import predictionReducer from './features/prediction/prediction'

export const store = configureStore({
  reducer: {
    prediction: predictionReducer
  },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch