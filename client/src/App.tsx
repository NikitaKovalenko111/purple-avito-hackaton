import { Route, Routes } from 'react-router-dom'
import { useEffect } from 'react'
import './assets/styles/main.css'
import Footer from './components/footer/Footer'
import Header from './components/header/Header'
import MainPage from './pages/main/MainPage'
import MetricsPage from './pages/metrics/MetricsPage'
import ProjectPage from './pages/project/ProjectPage'
import DatasetPage from './pages/dataset/DatasetPage'

function App() {
  useEffect(() => {
    document.title = 'Рабочая панель'
  }, [])

  return (
    <div className='page'>
      <Header />

      <Routes>
        <Route path='/dataset' element={<DatasetPage />} />
        <Route path='/project' element={<ProjectPage />} />
        <Route path='/metrics' element={<MetricsPage />} />
        <Route path='/' element={<MainPage />} />
      </Routes>

      <Footer />
    </div>
  )
}

export default App
