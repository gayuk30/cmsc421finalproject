import './App.css';
// import React, {useState} from 'react'; 
import {Route, Routes } from 'react-router-dom';
import Summarized from './components/summarized'; 
import MainPage from './components/mainPage';
import QuizPage from './components/QuizPage';

function App() {
  // const [url, setUrl] = useState(''); 
  // const [startTime, setStartTime] = useState('');
  // const [endTime, setEndTime] = useState('');
  // const [apiKey, setAPIkey] = useState('');

  // const handleSubmit = (e) => {
  //   e.preventDefault(); 
  //   console.log('URL:', url);
  //   console.log('Start Time:', startTime);
  //   console.log('End Time:', endTime);
  //   console.log('API Key:', apiKey); 
  // }
  return (
    <>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/summarized" element={<Summarized />} />
        <Route path="/quizPage" element={<QuizPage />} />
      </Routes>
    </>
      // <div className="Final">
      //   <header className="App-header">
      //     <h1>Welcome to our Video Summarizer!</h1>
      //       {/* <img src={logo} className="App-logo" alt="logo" /> */}
      //       <p>
      //         To start, fill out the fields below
      //       </p>
      //     <form onSubmit={handleSubmit}>
      //       <div className="info-container">
      //         <label>
      //           Please enter your <b>OpenAI API Key</b>:
      //           &nbsp;&nbsp;
      //           <input className="apiKey-input"
      //             type="text"
      //             value={apiKey}
      //             onChange={(e) => setAPIkey(e.target.value)}
      //             required
      //           />
      //         </label>
      //         <hr></hr>
      //         <label>
      //           Please enter the <b>YouTube Video URL</b>:
      //           &nbsp;&nbsp;
      //           <input className="url-input"
      //             type="text"
      //             value={url}
      //             onChange={(e) => setUrl(e.target.value)}
      //             required
      //           />
      //         </label>
      //         <hr></hr>
      //         <label>
      //           Please enter a <b>Timestamp (Optional)</b> if you only want part of the video summarized:
      //           <br></br>
      //           <br></br>
      //           <div className='inputs'>
      //             Start Time: 
      //             <input className="start-time"
      //               type="text"
      //               value={startTime}
      //               onChange={(e) => setStartTime(e.target.value)}
      //             />
      //             End Time: 
      //             <input className="end-time"
      //               type="text"
      //               value={endTime}
      //               onChange={(e) => setEndTime(e.target.value)}
      //             /> 
      //           </div>
      //         </label>
      //       </div> 
      //       <hr></hr>
      //       <div className="button-container">
      //         <button type="submit">Begin Generating Summary</button>
      //         Click this link below to see the generated summary and further features!
      //         <Link to="/summarized">Go to Summarized Page</Link>
      //       </div>
      //     </form>
      //   </header>
      // </div>
  );
}

export default App;
