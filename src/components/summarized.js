import React from 'react';
import {Link} from 'react-router-dom';
function Summarized() {
    return (
        <div className="Final">
        <header className="App-header">
          <h1> Summary Page</h1>
            <p>
                This is the summary of your video: 
            </p>
            <p>
                These are some key words from the video: 
            </p>
            <p>
                Here is a short quiz to test your learning!
            </p>
            <Link to="/quizPage">Go to Quiz!</Link>
        </header>
      </div>
    );
}

export default Summarized;