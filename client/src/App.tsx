import { Route, Routes } from 'react-router-dom'
import './assets/styles/main.css'
import Footer from './components/footer/Footer'
import Header from './components/header/Header'
import MainPage from './pages/main/MainPage'
import MetricsPage from './pages/metrics/MetricsPage'
import ProjectPage from './pages/project/ProjectPage'
import DatasetPage from './pages/dataset/DatasetPage'

function App() {
  return (
      <html lang="en">
        <head>
          <meta charSet="UTF-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0" />
          <title>Service Split Assistant Workspace</title>
          <meta
            name="description"
            content="Work panel for ad analysis, split decisions, and draft preparation."
          />
          <link rel="stylesheet" href="styles/main.css" />
        </head>
        <body className="page">
          <Header />

          <Routes>
            <Route path='/dataset' element={<DatasetPage />} />
            <Route path='/project' element={<ProjectPage />} />
            <Route path='/metrics' element={<MetricsPage />} />
            <Route path='/' element={<MainPage />} />
          </Routes>
          
          <Footer />
        </body>
      </html>
  )
}

export default App
